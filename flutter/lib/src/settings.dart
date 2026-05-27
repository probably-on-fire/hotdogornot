import 'package:shared_preferences/shared_preferences.dart';

/// Persistent app settings: predict-service URL and device token.
/// Defaults match the deployed aired.com setup so the app works out of the
/// box for the project owner. Edit via SettingsScreen.
class Settings {
  Settings._({
    required String relayBaseUrl,
    required String deviceToken,
    required this.devMode,
    required this.onDeviceMode,
  }) : _relayBaseUrl = _normalizeRelay(relayBaseUrl),
       _deviceToken = deviceToken.trim().isEmpty
           ? _defaultToken
           : deviceToken.trim();

  // Backing field for relayBaseUrl. We force every write through the
  // normalizing setter so a direct field mutation can't bypass it.
  String _relayBaseUrl;
  String get relayBaseUrl => _relayBaseUrl;
  set relayBaseUrl(String v) => _relayBaseUrl = _normalizeRelay(v);

  // deviceToken is similarly trimmed — an accidental newline / whitespace
  // would otherwise ship as part of the X-Device-Token header and the
  // server would 401 with a cryptic message.
  String _deviceToken;
  String get deviceToken => _deviceToken;
  set deviceToken(String v) {
    final t = v.trim();
    _deviceToken = t.isEmpty ? _defaultToken : t;
  }
  // When true, the Contribute tab and the Advanced (relay/token)
  // panel are visible. Toggled by 7-tap on the version string in About.
  // Default off so end users see only Identify + About.
  bool devMode;
  // When true, /predict calls bypass the server and run the bundled
  // ResNet-18 ONNX model on-device. Useful for offline / low-latency
  // identification. Default off (still uses aired.com). Toggle in the
  // Advanced section (dev-mode-gated).
  bool onDeviceMode;

  static const _kRelay = 'relay_base_url';
  static const _kToken = 'device_token';
  static const _kDevMode = 'dev_mode';
  static const _kOnDevice = 'on_device_mode';

  static const _defaultRelay = 'https://aired.com/rfcai';
  static const _defaultToken =
      '66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1';

  static Future<Settings> load() async {
    final prefs = await SharedPreferences.getInstance();
    return Settings._(
      relayBaseUrl: _normalizeRelay(prefs.getString(_kRelay)),
      deviceToken: prefs.getString(_kToken) ?? _defaultToken,
      devMode: prefs.getBool(_kDevMode) ?? false,
      onDeviceMode: prefs.getBool(_kOnDevice) ?? false,
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    // _relayBaseUrl is already normalized by the setter; no need to
    // re-normalize here.
    await prefs.setString(_kRelay, _relayBaseUrl);
    await prefs.setString(_kToken, deviceToken);
    await prefs.setBool(_kDevMode, devMode);
    await prefs.setBool(_kOnDevice, onDeviceMode);
  }

  /// Guard the relay URL: empty / whitespace-only / no scheme falls back
  /// to the default so the next HTTP request doesn't crash with
  /// `ArgumentError: No host specified in URI /predict`.
  static String _normalizeRelay(String? raw) {
    final s = (raw ?? '').trim();
    if (s.isEmpty) return _defaultRelay;
    // Must parse + have a scheme + host. Reject anything weird (e.g.
    // 'aired.com', 'http://' alone, just '/' etc.).
    Uri? u;
    try { u = Uri.parse(s); } catch (_) { return _defaultRelay; }
    if (!(u.hasScheme && u.host.isNotEmpty)) return _defaultRelay;
    // Strip any trailing slash so '$relayBaseUrl/predict' doesn't double.
    return s.endsWith('/') ? s.substring(0, s.length - 1) : s;
  }

  String get predictUrl => '$relayBaseUrl/predict';

  String get predictVideoUrl => '$relayBaseUrl/predict-video';

  String labelerUploadTrainUrl() => '$relayBaseUrl/labeler/upload-train';

  String labelerUploadTestUrl() => '$relayBaseUrl/labeler/upload-test';

  String labelerUploadVideoUrl() => '$relayBaseUrl/labeler/upload-video';

  String labelerStatsUrl() => '$relayBaseUrl/labeler/stats';

  String labelerDeleteUrl() => '$relayBaseUrl/labeler/delete';
}
