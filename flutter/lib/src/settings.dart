import 'package:shared_preferences/shared_preferences.dart';

/// Persistent app settings: predict-service URL, device token, labeler creds.
/// Defaults match the deployed aired.com setup so the app works out of the
/// box for the project owner. Edit via SettingsScreen.
class Settings {
  Settings._({
    required this.relayBaseUrl,
    required this.deviceToken,
    required this.labelerUser,
    required this.labelerPass,
  });

  String relayBaseUrl;
  String deviceToken;
  String labelerUser;
  String labelerPass;

  static const _kRelay = 'relay_base_url';
  static const _kToken = 'device_token';
  static const _kUser = 'labeler_user';
  static const _kPass = 'labeler_pass';

  static const _defaultRelay = 'https://aired.com/rfcai';
  static const _defaultToken =
      '66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1';
  static const _defaultUser = 'admin';
  static const _defaultPass = '663800c2f2a8f2c4e33f2c43';

  static Future<Settings> load() async {
    final prefs = await SharedPreferences.getInstance();
    return Settings._(
      relayBaseUrl: prefs.getString(_kRelay) ?? _defaultRelay,
      deviceToken: prefs.getString(_kToken) ?? _defaultToken,
      labelerUser: prefs.getString(_kUser) ?? _defaultUser,
      labelerPass: prefs.getString(_kPass) ?? _defaultPass,
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kRelay, relayBaseUrl);
    await prefs.setString(_kToken, deviceToken);
    await prefs.setString(_kUser, labelerUser);
    await prefs.setString(_kPass, labelerPass);
  }

  String get predictUrl => '$relayBaseUrl/predict';

  String labelerUploadTrainUrl() => '$relayBaseUrl/labeler/upload-train';

  String labelerUploadVideoUrl() => '$relayBaseUrl/labeler/upload-video';
}
