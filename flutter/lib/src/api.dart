import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'auth.dart';
import 'settings.dart';

/// One file the server saved. The `path` round-trips through
/// `/labeler/delete` for the Undo flow.
class UploadRecord {
  UploadRecord({required this.cls, required this.path});
  final String cls;
  final String path;

  factory UploadRecord.fromJson(Map<String, dynamic> j) =>
      UploadRecord(cls: j['cls'] as String, path: j['path'] as String);
}

/// Parsed JSON from `/upload-train` and `/upload-test`.
/// Response from `/labeler/upload-video` when `with_predict=true`. Older
/// servers that don't know about the flag still return HTML; in that
/// case `bestPrediction` is null and `savedCrops` is null.
class VideoTrainingUploadResult {
  VideoTrainingUploadResult({
    this.savedCrops,
    this.framesScanned,
    this.bestPrediction,
  });
  final int? savedCrops;
  final int? framesScanned;
  final Prediction? bestPrediction;

  factory VideoTrainingUploadResult.fromResponseBody(String body) {
    try {
      final j = jsonDecode(body);
      if (j is! Map<String, dynamic>) {
        return VideoTrainingUploadResult();
      }
      final preds = (j['predictions'] as List?) ?? [];
      return VideoTrainingUploadResult(
        savedCrops: (j['saved_crops'] as num?)?.toInt(),
        framesScanned: (j['frames_scanned'] as num?)?.toInt(),
        bestPrediction: preds.isEmpty
            ? null
            : Prediction.fromJson(preds.first as Map<String, dynamic>),
      );
    } catch (_) {
      return VideoTrainingUploadResult();   // legacy HTML response
    }
  }
}

class UploadResult {
  UploadResult({required this.saved, required this.errors});
  final List<UploadRecord> saved;
  final List<String> errors;

  factory UploadResult.fromJson(Map<String, dynamic> j) {
    final saved = (j['saved'] as List? ?? [])
        .map((e) => UploadRecord.fromJson(e as Map<String, dynamic>))
        .toList();
    final errors = (j['errors'] as List? ?? [])
        .map((e) => e.toString())
        .toList();
    return UploadResult(saved: saved, errors: errors);
  }
}

/// Per-class capture counts returned by `/labeler/stats`.
class LabelerStats {
  LabelerStats({required this.train, required this.holdout});
  final Map<String, int> train;
  final Map<String, int> holdout;

  factory LabelerStats.fromJson(Map<String, dynamic> j) {
    Map<String, int> toIntMap(Map<String, dynamic> m) =>
        m.map((k, v) => MapEntry(k, (v as num).toInt()));
    return LabelerStats(
      train: toIntMap(j['train'] as Map<String, dynamic>),
      holdout: toIntMap(j['holdout'] as Map<String, dynamic>),
    );
  }
}

/// One detection from the /predict endpoint.
class Prediction {
  Prediction({
    required this.className,
    required this.confidence,
    required this.probabilities,
    required this.bbox,
    this.serverFamily,
    this.serverGender,
    this.familyConfidence,
    this.genderConfidence,
    this.spec,
  });

  final String className;
  final double confidence;
  final Map<String, double> probabilities;
  final Map<String, int> bbox;

  // Optional structured fields the upgraded /predict service emits
  // alongside the legacy `class_name`. Older servers return null/None
  // and the parser falls back to deriving from `className`.
  final String? serverFamily;
  final String? serverGender;
  final double? familyConfidence;
  final double? genderConfidence;
  final Map<String, dynamic>? spec;

  /// Family is server-provided when available, else derived from className.
  String get family {
    if (serverFamily != null && serverFamily!.isNotEmpty) return serverFamily!;
    final i = className.lastIndexOf('-');
    return i < 0 ? className : className.substring(0, i);
  }

  /// Gender is server-provided when available, else derived ("M"/"F").
  String get gender {
    if (serverGender != null && serverGender!.isNotEmpty) return serverGender!;
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
      serverFamily: j['family'] as String?,
      serverGender: j['gender'] as String?,
      familyConfidence: (j['family_confidence'] as num?)?.toDouble(),
      genderConfidence: (j['gender_confidence'] as num?)?.toDouble(),
      spec: j['spec'] is Map<String, dynamic>
          ? j['spec'] as Map<String, dynamic>
          : null,
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

/// Thrown when the labeler API returns 401. The Contribute screen
/// catches this to show the sign-in card.
class UnauthorizedException implements Exception {
  const UnauthorizedException();
  @override
  String toString() => 'not signed in';
}

class ApiClient {
  /// [auth] is optional — predict endpoints are anonymous. Pass an
  /// [AuthService] for any labeler write endpoints that require a token.
  ApiClient(this.settings, [this.auth]);
  final Settings settings;
  final AuthService? auth;

  Map<String, String> get _authHeaders => auth?.authHeaders ?? const {};

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
    // 60s: slow-cellular ~3-4MB JPEG upload can eat 20s+ before the
    // classifier even starts; 30s surfaced as "server too long" when the
    // real bottleneck was the user's uplink.
    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return PredictResponse.fromJson(jsonDecode(resp.body));
  }

  /// POST a video clip to /predict-video. Server samples at 3 fps,
  /// classifies each frame, and returns a softmax-averaged consensus
  /// across frames (more robust to single-frame variance than a single
  /// photo). On a 1.5s burst that's ~5 frames of consensus.
  Future<PredictResponse> predictVideo(File videoFile) async {
    // Force `.mp4` — the Android camera plugin writes recordings to a
    // .tmp file in cache, and the server's video-extension allowlist
    // rejects unknown suffixes. iOS happens to use .mov which is
    // accepted, but neither path's filename is meaningful to the server
    // (it renames on disk), so just send something legal.
    return predictVideoBytes(
      await videoFile.readAsBytes(),
      filename: 'video.mp4',
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
  /// e.g. "2.4mm-M". Optional [session] is prepended to the filename on
  /// the server; defaults to today's UTC date when omitted.
  Future<UploadResult> uploadTrainingPhoto(File imageFile, String cls, {String? session}) async {
    final fields = {'cls': cls};
    if (session != null && session.isNotEmpty) fields['session'] = session;
    final body = await _uploadMultipart(
      url: settings.labelerUploadTrainUrl(),
      fields: fields,
      fileField: 'images',
      file: imageFile,
      timeout: const Duration(seconds: 90),  // photo, not video
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// Bytes variant of uploadTrainingPhoto for web/no-path-access.
  Future<UploadResult> uploadTrainingPhotoBytes(
    Uint8List bytes,
    String cls, {
    String filename = 'photo.jpg',
    String? session,
  }) async {
    final fields = {'cls': cls};
    if (session != null && session.isNotEmpty) fields['session'] = session;
    final body = await _uploadMultipartBytes(
      url: settings.labelerUploadTrainUrl(),
      fields: fields,
      fileField: 'images',
      bytes: bytes,
      filename: filename,
      timeout: const Duration(seconds: 90),  // photo, not video
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// POST a held-out test photo to the labeler. cls is the canonical
  /// class. Held-out photos are not used in training — they live in
  /// data/test_holdout/[cls]/ and are scored against during retrain
  /// benchmarks. Use this when growing the test set with new phone
  /// shots whose label is known. Optional [session] is prepended to the
  /// filename on the server.
  Future<UploadResult> uploadTestHoldoutPhoto(File imageFile, String cls, {String? session}) async {
    final fields = {'cls': cls};
    if (session != null && session.isNotEmpty) fields['session'] = session;
    final body = await _uploadMultipart(
      url: settings.labelerUploadTestUrl(),
      fields: fields,
      fileField: 'images',
      file: imageFile,
      timeout: const Duration(seconds: 90),  // photo, not video
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// Bytes variant of uploadTestHoldoutPhoto for web/no-path-access.
  Future<UploadResult> uploadTestHoldoutPhotoBytes(
    Uint8List bytes,
    String cls, {
    String filename = 'photo.jpg',
    String? session,
  }) async {
    final fields = {'cls': cls};
    if (session != null && session.isNotEmpty) fields['session'] = session;
    final body = await _uploadMultipartBytes(
      url: settings.labelerUploadTestUrl(),
      fields: fields,
      fileField: 'images',
      bytes: bytes,
      filename: filename,
      timeout: const Duration(seconds: 90),  // photo, not video
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// POST a training video to the labeler. family is "2.4mm" / "2.92mm" /
  /// "3.5mm" / "SMA". gender is "M" or "F". with_predict=true asks the
  /// server to also classify the extracted crops and return the best
  /// prediction alongside the saved-crop count.
  Future<VideoTrainingUploadResult> uploadTrainingVideo(
      File videoFile, String family, String gender) async {
    // Bypass _uploadMultipart's path-derived filename — Android writes
    // the camera-plugin clip to `<cache>/CAPxxx.tmp`, which the server
    // rejects as an unknown video extension. Send as `video.mp4`; the
    // server renames on disk anyway based on family + gender.
    final body = await _uploadMultipartBytes(
      url: settings.labelerUploadVideoUrl(),
      fields: {
        'family': family,
        'gender': gender,
        'fps': '5',
        'sensitivity': '2.0',
        'max_crops': '5',
        'with_predict': 'true',
      },
      fileField: 'file',
      bytes: await videoFile.readAsBytes(),
      filename: 'video.mp4',
    );
    return VideoTrainingUploadResult.fromResponseBody(body);
  }

  /// GET /labeler/stats — per-class real-capture counts.
  Future<LabelerStats> fetchLabelerStats() async {
    final req = http.Request('GET', Uri.parse(settings.labelerStatsUrl()));
    req.headers.addAll(_authHeaders);
    final streamed = await req.send().timeout(const Duration(seconds: 30));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode == 401) throw const UnauthorizedException();
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return LabelerStats.fromJson(jsonDecode(resp.body));
  }

  /// POST /labeler/delete — unlink one server-side file by path.
  /// Used by the in-app Undo flow.
  Future<void> deleteLabelerFile(String path) async {
    final req = http.MultipartRequest(
      'POST', Uri.parse(settings.labelerDeleteUrl()),
    );
    req.headers.addAll(_authHeaders);
    req.fields['path'] = path;
    final streamed = await req.send().timeout(const Duration(seconds: 15));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode == 401) throw const UnauthorizedException();
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
  }

  Future<String> _uploadMultipart({
    required String url,
    required Map<String, String> fields,
    required String fileField,
    required File file,
    Duration? timeout,
  }) async {
    return _uploadMultipartBytes(
      url: url,
      timeout: timeout,
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
    Duration? timeout,
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse(url));
    req.headers.addAll(_authHeaders);
    req.fields.addAll(fields);
    req.files.add(http.MultipartFile.fromBytes(fileField, bytes, filename: filename));
    // Caller picks the timeout. Photo uploads need ~90s (slow cellular
    // for a 5MB JPEG); video uploads need ~600s (200MB phone clip +
    // server-side ffmpeg/extract). Default = 600s for back-compat.
    final t = timeout ?? const Duration(seconds: 600);
    final streamed = await req.send().timeout(t);
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode == 401) throw const UnauthorizedException();
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
