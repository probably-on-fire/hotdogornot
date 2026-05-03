import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'settings.dart';

/// One detection from the /predict endpoint.
class Prediction {
  Prediction({
    required this.className,
    required this.confidence,
    required this.probabilities,
    required this.bbox,
  });

  final String className;
  final double confidence;
  final Map<String, double> probabilities;
  final Map<String, int> bbox;

  /// Convenience: family is everything before the last "-".
  String get family {
    final i = className.lastIndexOf('-');
    return i < 0 ? className : className.substring(0, i);
  }

  /// Convenience: gender is everything after the last "-" ("M" or "F").
  String get gender {
    final i = className.lastIndexOf('-');
    return i < 0 ? '' : className.substring(i + 1);
  }

  bool get isMale => gender == 'M';
  bool get isFemale => gender == 'F';

  factory Prediction.fromJson(Map<String, dynamic> j) {
    return Prediction(
      className: j['class_name'] as String,
      confidence: (j['confidence'] as num).toDouble(),
      probabilities: (j['probabilities'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
      bbox: (j['bbox'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toInt()),
      ),
    );
  }
}

class PredictResponse {
  PredictResponse({
    required this.imageWidth,
    required this.imageHeight,
    required this.predictions,
  });

  final int imageWidth;
  final int imageHeight;
  final List<Prediction> predictions;

  factory PredictResponse.fromJson(Map<String, dynamic> j) {
    return PredictResponse(
      imageWidth: j['image_width'] as int,
      imageHeight: j['image_height'] as int,
      predictions: (j['predictions'] as List)
          .map((p) => Prediction.fromJson(p as Map<String, dynamic>))
          .toList(),
    );
  }
}

class ApiClient {
  ApiClient(this.settings);
  final Settings settings;

  /// POST a JPEG to the /predict endpoint and return parsed predictions.
  /// Throws on non-2xx, network error, or invalid JSON.
  Future<PredictResponse> predict(File imageFile) async {
    final req = http.MultipartRequest('POST', Uri.parse(settings.predictUrl));
    req.headers['X-Device-Token'] = settings.deviceToken;
    req.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
    final streamed = await req.send().timeout(const Duration(seconds: 30));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return PredictResponse.fromJson(jsonDecode(resp.body));
  }

  /// POST a training photo to the labeler. cls is the canonical class
  /// e.g. "2.4mm-M".
  Future<String> uploadTrainingPhoto(File imageFile, String cls) async {
    return _uploadMultipart(
      url: settings.labelerUploadTrainUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      file: imageFile,
    );
  }

  /// POST a training video to the labeler. family is "2.4mm" / "2.92mm" /
  /// "3.5mm" / "SMA".
  Future<String> uploadTrainingVideo(File videoFile, String family) async {
    return _uploadMultipart(
      url: settings.labelerUploadVideoUrl(),
      fields: {
        'family': family,
        'fps': '5',
        'sensitivity': '2.0',
        'max_crops': '5',
      },
      fileField: 'file',
      file: videoFile,
    );
  }

  Future<String> _uploadMultipart({
    required String url,
    required Map<String, String> fields,
    required String fileField,
    required File file,
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse(url));
    final basic = base64Encode(utf8.encode(
      '${settings.labelerUser}:${settings.labelerPass}',
    ));
    req.headers['Authorization'] = 'Basic $basic';
    req.fields.addAll(fields);
    req.files.add(await http.MultipartFile.fromPath(fileField, file.path));
    final streamed = await req.send().timeout(const Duration(seconds: 120));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return resp.body;
  }
}

class _ApiError implements Exception {
  _ApiError(this.statusCode, this.body);
  final int statusCode;
  final String body;
  @override
  String toString() => 'ApiError $statusCode: ${body.length > 200 ? '${body.substring(0, 200)}...' : body}';
}
