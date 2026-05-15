import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

import 'settings.dart';

/// Manages the API token used for labeler writes.
///
/// First launch / after sign-out: `token` is null and `isSignedIn` is
/// false. Once the user signs in, the token is persisted to secure
/// storage and `isSignedIn` flips true. Notifies listeners on every
/// state change so Contribute can reactively swap the camera vs the
/// sign-in card.
class AuthService extends ChangeNotifier {
  AuthService(this._settings);

  final Settings _settings;
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  String? _token;
  String? _username;

  static const _kToken = 'rfcai_api_token';
  static const _kUsername = 'rfcai_username';

  String? get token => _token;
  String? get username => _username;
  bool get isSignedIn => _token != null && _token!.isNotEmpty;

  /// Read any persisted token into memory. Call once at app startup.
  Future<void> load() async {
    _token = await _storage.read(key: _kToken);
    _username = await _storage.read(key: _kUsername);
    notifyListeners();
  }

  /// Exchange username+password for a long-lived Bearer token.
  /// Throws [AuthFailed] on 401. Throws other errors as-is.
  Future<void> signIn({
    required String username,
    required String password,
    String deviceName = 'phone',
  }) async {
    final url = '${_settings.relayBaseUrl}/labeler/api-tokens/exchange';
    final req = http.MultipartRequest('POST', Uri.parse(url));
    req.fields['username'] = username;
    req.fields['password'] = password;
    req.fields['name'] = deviceName;
    final streamed = await req.send().timeout(const Duration(seconds: 20));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode == 401) {
      throw AuthFailed('invalid username or password');
    }
    if (resp.statusCode != 200) {
      throw Exception('sign-in failed: ${resp.statusCode} ${resp.body}');
    }
    final body = jsonDecode(resp.body) as Map<String, dynamic>;
    final t = body['token'] as String?;
    if (t == null || t.isEmpty) {
      throw Exception('sign-in response missing token');
    }
    await _storage.write(key: _kToken, value: t);
    await _storage.write(key: _kUsername, value: username);
    _token = t;
    _username = username;
    notifyListeners();
  }

  /// Clear the local token. Does NOT revoke server-side (admin can
  /// delete the token from the admin UI). On re-sign-in a new token
  /// is issued.
  Future<void> signOut() async {
    await _storage.delete(key: _kToken);
    await _storage.delete(key: _kUsername);
    _token = null;
    _username = null;
    notifyListeners();
  }

  /// Bearer-auth header for the labeler API. Empty map if not signed
  /// in — callers must check isSignedIn first to avoid 401 round-trips.
  Map<String, String> get authHeaders {
    final t = _token;
    if (t == null || t.isEmpty) return const {};
    return {'Authorization': 'Bearer $t'};
  }
}

class AuthFailed implements Exception {
  AuthFailed(this.message);
  final String message;
  @override
  String toString() => message;
}
