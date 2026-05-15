import 'package:shared_preferences/shared_preferences.dart';

/// Persistent app settings: predict-service URL and device token.
/// Defaults match the deployed aired.com setup so the app works out of the
/// box for the project owner. Edit via SettingsScreen.
class Settings {
  Settings._({
    required this.relayBaseUrl,
    required this.deviceToken,
    required this.devMode,
    required this.onDeviceMode,
  });

  String relayBaseUrl;
  String deviceToken;
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
      relayBaseUrl: prefs.getString(_kRelay) ?? _defaultRelay,
      deviceToken: prefs.getString(_kToken) ?? _defaultToken,
      devMode: prefs.getBool(_kDevMode) ?? false,
      onDeviceMode: prefs.getBool(_kOnDevice) ?? false,
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kRelay, relayBaseUrl);
    await prefs.setString(_kToken, deviceToken);
    await prefs.setBool(_kDevMode, devMode);
    await prefs.setBool(_kOnDevice, onDeviceMode);
  }

  String get predictUrl => '$relayBaseUrl/predict';

  String get predictVideoUrl => '$relayBaseUrl/predict-video';

  String labelerUploadTrainUrl() => '$relayBaseUrl/labeler/upload-train';

  String labelerUploadTestUrl() => '$relayBaseUrl/labeler/upload-test';

  String labelerUploadVideoUrl() => '$relayBaseUrl/labeler/upload-video';

  String labelerStatsUrl() => '$relayBaseUrl/labeler/stats';

  String labelerDeleteUrl() => '$relayBaseUrl/labeler/delete';
}
