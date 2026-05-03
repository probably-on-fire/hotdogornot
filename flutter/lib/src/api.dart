import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

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
  /// Native path — uses on-disk file, avoids reading bytes into memory.
  Future<PredictResponse> predict(File imageFile) async {
    return predictBytes(
      await imageFile.readAsBytes(),
      filename: imageFile.uri.pathSegments.isNotEmpty
          ? imageFile.uri.pathSegments.last
          : 'image.jpg',
    );
  }

  /// POST raw image bytes to /predict. Web-safe — works without dart:io
  /// path access (XFile.path is a blob URL in browser).
  Future<PredictResponse> predictBytes(
    Uint8List bytes, {
    String filename = 'image.jpg',
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse(settings.predictUrl));
    req.headers['X-Device-Token'] = settings.deviceToken;
    req.files.add(http.MultipartFile.fromBytes('image', bytes, filename: filename));
    final streamed = await req.send().timeout(const Duration(seconds: 30));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return PredictResponse.fromJson(jsonDecode(resp.body));
  }

  /// POST a video clip to /predict-video. Server samples at 1 fps and
  /// returns the highest-confidence single-frame prediction.
  Future<PredictResponse> predictVideo(File videoFile) async {
    return predictVideoBytes(
      await videoFile.readAsBytes(),
      filename: videoFile.uri.pathSegments.isNotEmpty
          ? videoFile.uri.pathSegments.last
          : 'video.mp4',
    );
  }

  /// Bytes variant of predictVideo for web/no-path-access environments.
  Future<PredictResponse> predictVideoBytes(
    Uint8List bytes, {
    String filename = 'video.mp4',
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse(settings.predictVideoUrl));
    req.headers['X-Device-Token'] = settings.deviceToken;
    req.files.add(http.MultipartFile.fromBytes('video', bytes, filename: filename));
    final streamed = await req.send().timeout(const Duration(seconds: 120));
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

  /// Bytes variant of uploadTrainingPhoto for web/no-path-access.
  Future<String> uploadTrainingPhotoBytes(
    Uint8List bytes,
    String cls, {
    String filename = 'photo.jpg',
  }) async {
    return _uploadMultipartBytes(
      url: settings.labelerUploadTrainUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      bytes: bytes,
      filename: filename,
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
    return _uploadMultipartBytes(
      url: url,
      fields: fields,
      fileField: fileField,
      bytes: await file.readAsBytes(),
      filename: file.uri.pathSegments.isNotEmpty
          ? file.uri.pathSegments.last
          : 'upload.bin',
    );
  }

  Future<String> _uploadMultipartBytes({
    required String url,
    required Map<String, String> fields,
    required String fileField,
    required Uint8List bytes,
    required String filename,
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse(url));
    final basic = base64Encode(utf8.encode(
      '${settings.labelerUser}:${settings.labelerPass}',
    ));
    req.headers['Authorization'] = 'Basic $basic';
    req.fields.addAll(fields);
    req.files.add(http.MultipartFile.fromBytes(fileField, bytes, filename: filename));
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
