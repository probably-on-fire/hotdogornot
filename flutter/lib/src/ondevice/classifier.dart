/// On-device ResNet-18 inference. Loads the same .onnx weights the
/// server serves, runs the same ImageNet-normalized 224x224 forward
/// pass, returns a 6-class softmax. No rembg, no Hough, no TTA — this
/// is the Tier-1 spike that proves the standalone-on-phone path works.
/// Subsequent tiers can add preprocessing back in if the accuracy gap
/// is too wide on the field test.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

/// One inference result from the on-device classifier — kept shape-
/// compatible with the server's Prediction so the rest of the app
/// can consume it identically.
class OnDevicePrediction {
  OnDevicePrediction({
    required this.className,
    required this.confidence,
    required this.probabilities,
    required this.latencyMs,
  });

  final String className;
  final double confidence;
  final Map<String, double> probabilities;
  final int latencyMs;

  String get family {
    final i = className.lastIndexOf('-');
    return i < 0 ? className : className.substring(0, i);
  }

  String get gender {
    final i = className.lastIndexOf('-');
    return i < 0 ? '' : className.substring(i + 1);
  }
}

/// Lazy-initialised singleton. Loads the ONNX model on first call,
/// reuses the session across requests. Bundle size adds ~44 MB to
/// the app — same architecture (ResNet-18) we serve on aired.com,
/// same `weights.synth_20ep.onnx` file, same inverted-label mapping.
class OnDeviceClassifier {
  OnDeviceClassifier._();
  static final OnDeviceClassifier instance = OnDeviceClassifier._();

  OrtSession? _session;
  List<String>? _classNames;
  int _inputSize = 224;
  List<double> _mean = const [0.485, 0.456, 0.406];
  List<double> _std = const [0.229, 0.224, 0.225];
  bool _initStarted = false;
  Completer<void>? _initCompleter;

  /// Initialise once at app startup (background). Subsequent calls
  /// to [predict] await this if it's still in flight.
  Future<void> init() async {
    if (_session != null) return;
    if (_initStarted) {
      await _initCompleter?.future;
      return;
    }
    _initStarted = true;
    _initCompleter = Completer<void>();
    try {
      OrtEnv.instance.init();
      // Labels + preprocessing config.
      final labelsJson = await rootBundle.loadString(
        'assets/models/labels.json',
      );
      final meta = json.decode(labelsJson) as Map<String, dynamic>;
      _classNames = (meta['class_names'] as List).cast<String>();
      _inputSize = (meta['input_size'] as int?) ?? 224;
      _mean = ((meta['imagenet_mean'] as List?)?.cast<num>() ??
              [0.485, 0.456, 0.406])
          .map((n) => n.toDouble())
          .toList();
      _std = ((meta['imagenet_std'] as List?)?.cast<num>() ??
              [0.229, 0.224, 0.225])
          .map((n) => n.toDouble())
          .toList();

      // ONNX Runtime won't load directly from the asset bundle on
      // iOS, so copy the model to the app's temp dir on first run
      // and load by file path. ~44 MB so the copy is cheap and only
      // happens once per install (skipped on subsequent launches
      // when the file is already present at the same size).
      final bytes = await rootBundle.load(
        'assets/models/connector_classifier.onnx',
      );
      final tmp = await getTemporaryDirectory();
      final dst = File('${tmp.path}/connector_classifier.onnx');
      if (!await dst.exists() ||
          await dst.length() != bytes.lengthInBytes) {
        await dst.writeAsBytes(
          bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes),
        );
      }
      final options = OrtSessionOptions();
      _session = OrtSession.fromFile(dst, options);
      _initCompleter!.complete();
    } catch (e, st) {
      _initCompleter!.completeError(e, st);
      _initStarted = false;
      rethrow;
    }
  }

  /// Run inference on a JPEG/PNG byte buffer. Resizes to 224x224,
  /// ImageNet-normalises, returns the top class + 6-way softmax.
  Future<OnDevicePrediction> predict(Uint8List imageBytes) async {
    await init();
    final session = _session;
    final classNames = _classNames;
    if (session == null || classNames == null) {
      throw StateError('classifier not initialised');
    }
    final t0 = DateTime.now();

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      throw StateError('could not decode image');
    }
    final resized = img.copyResize(
      decoded,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Flatten into CHW Float32 with ImageNet normalisation.
    final n = _inputSize * _inputSize;
    final buf = Float32List(3 * n);
    var idx = 0;
    for (final c in [0, 1, 2]) {
      final m = _mean[c];
      final s = _std[c];
      for (var y = 0; y < _inputSize; y++) {
        for (var x = 0; x < _inputSize; x++) {
          final p = resized.getPixel(x, y);
          final v = (c == 0
                      ? p.r
                      : c == 1
                          ? p.g
                          : p.b)
                  .toDouble() /
              255.0;
          buf[idx++] = (v - m) / s;
        }
      }
    }

    final shape = [1, 3, _inputSize, _inputSize];
    final inputTensor = OrtValueTensor.createTensorWithDataList(buf, shape);
    final inputs = {'input': inputTensor};
    final outputs = await session.runAsync(OrtRunOptions(), inputs);
    inputTensor.release();
    if (outputs == null || outputs.isEmpty) {
      throw StateError('onnx returned no outputs');
    }
    final raw = outputs.first?.value as List<List<double>>;
    final logits = raw[0];

    // Numerically-stable softmax for human-readable confidences.
    var maxLogit = double.negativeInfinity;
    for (final v in logits) {
      if (v > maxLogit) maxLogit = v;
    }
    final exps = logits.map((v) => math.exp(v - maxLogit)).toList();
    final sumExp = exps.reduce((a, b) => a + b);
    final probs = exps.map((v) => v / sumExp).toList();
    var topIdx = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[topIdx]) topIdx = i;
    }
    final probabilities = <String, double>{
      for (var i = 0; i < classNames.length; i++)
        classNames[i]: probs[i].toDouble(),
    };
    final latency = DateTime.now().difference(t0).inMilliseconds;

    for (final o in outputs) {
      o?.release();
    }
    return OnDevicePrediction(
      className: classNames[topIdx],
      confidence: probs[topIdx].toDouble(),
      probabilities: probabilities,
      latencyMs: latency,
    );
  }
}
