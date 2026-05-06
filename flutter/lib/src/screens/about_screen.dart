import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show HapticFeedback;
import 'package:package_info_plus/package_info_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import '../settings.dart';

/// "About" tab — visible to every user. Doubles as the dev-mode toggle:
/// tap the version string 7 times in quick succession to flip dev mode,
/// which reveals the Contribute tab + the Advanced (relay / token /
/// labeler) panel further down this screen.
class AboutScreen extends StatefulWidget {
  const AboutScreen({
    super.key,
    required this.settings,
    required this.onDevModeChanged,
  });
  final Settings settings;
  final ValueChanged<bool> onDevModeChanged;

  @override
  State<AboutScreen> createState() => _AboutScreenState();
}

class _AboutScreenState extends State<AboutScreen> {
  String _version = '';
  String _build = '';

  // Hidden dev-mode unlock — Android-style "tap version 7 times" gesture.
  int _devTapCount = 0;
  Timer? _devTapTimer;

  // Advanced panel controllers (only relevant when dev mode is on).
  late TextEditingController _relayCtl;
  late TextEditingController _tokenCtl;
  late TextEditingController _userCtl;
  late TextEditingController _passCtl;
  String? _saveStatus;

  @override
  void initState() {
    super.initState();
    _relayCtl = TextEditingController(text: widget.settings.relayBaseUrl);
    _tokenCtl = TextEditingController(text: widget.settings.deviceToken);
    _userCtl = TextEditingController(text: widget.settings.labelerUser);
    _passCtl = TextEditingController(text: widget.settings.labelerPass);
    _loadVersion();
  }

  @override
  void dispose() {
    _devTapTimer?.cancel();
    _relayCtl.dispose();
    _tokenCtl.dispose();
    _userCtl.dispose();
    _passCtl.dispose();
    super.dispose();
  }

  Future<void> _loadVersion() async {
    try {
      final info = await PackageInfo.fromPlatform();
      if (!mounted) return;
      setState(() {
        _version = info.version;
        _build = info.buildNumber;
      });
    } catch (_) {/* version stays blank, screen still works */}
  }

  void _onVersionTap() {
    HapticFeedback.selectionClick();
    _devTapCount++;
    _devTapTimer?.cancel();
    // Reset the counter if the user pauses for >2 seconds.
    _devTapTimer = Timer(const Duration(seconds: 2), () {
      _devTapCount = 0;
    });

    // Mid-progress hint — match Android's "X taps to enable developer mode".
    final remaining = 7 - _devTapCount;
    if (remaining > 0 && remaining <= 4) {
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(
          SnackBar(
            duration: const Duration(milliseconds: 800),
            content: Text(
              widget.settings.devMode
                  ? '$remaining more taps to disable developer mode'
                  : '$remaining more taps to enable developer mode',
            ),
          ),
        );
    }

    if (_devTapCount >= 7) {
      _devTapCount = 0;
      _devTapTimer?.cancel();
      final next = !widget.settings.devMode;
      widget.onDevModeChanged(next);
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(
          SnackBar(
            content: Text(next
                ? 'Developer mode enabled'
                : 'Developer mode disabled'),
          ),
        );
      HapticFeedback.mediumImpact();
    }
  }

  Future<void> _openAired() async {
    final uri = Uri.parse('https://aired.com');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> _saveAdvanced() async {
    widget.settings
      ..relayBaseUrl = _relayCtl.text.trim()
      ..deviceToken = _tokenCtl.text.trim()
      ..labelerUser = _userCtl.text.trim()
      ..labelerPass = _passCtl.text.trim();
    await widget.settings.save();
    if (!mounted) return;
    setState(() => _saveStatus = '✓ Saved');
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _saveStatus = null);
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 32, 24, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Hero brand block.
              Container(
                width: 96,
                height: 96,
                decoration: BoxDecoration(
                  color: const Color(0xFFE63946),  // aired.com red
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Center(
                  child: Text(
                    'ai',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 40,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -1,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Connector ID',
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 4),
              GestureDetector(
                onTap: _onVersionTap,
                behavior: HitTestBehavior.opaque,
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 6),
                  child: Text(
                    _version.isEmpty
                        ? 'version —'
                        : 'v$_version  ($_build)',
                    style: TextStyle(
                      color: theme.colorScheme.onSurface.withOpacity(0.55),
                      fontSize: 13,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 28),

              // Description
              Text(
                'Identify RF coaxial connectors from your phone camera. '
                'Supports SMA, 3.5mm, 2.92mm, and 2.4mm in male and female '
                'variants. Point, snap, get a class.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: theme.colorScheme.onSurface.withOpacity(0.85),
                ),
              ),

              const SizedBox(height: 28),

              // aired.com promo card.
              GestureDetector(
                onTap: _openAired,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        Color(0xFFE63946),
                        Color(0xFFB8232F),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Powered by aired.com',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.2,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'AI-driven RF tooling',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          height: 1.2,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        'Built and operated by aired.com — '
                        'tap to learn more.',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.92),
                          fontSize: 13,
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Row(
                        children: const [
                          Text(
                            'Visit aired.com',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          SizedBox(width: 4),
                          Icon(Icons.arrow_forward,
                              color: Colors.white, size: 16),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // App credits / fine print.
              Text(
                '© aired.com',
                style: TextStyle(
                  color: theme.colorScheme.onSurface.withOpacity(0.45),
                  fontSize: 11,
                ),
              ),

              // Advanced section — only visible in dev mode.
              if (widget.settings.devMode) ...[
                const SizedBox(height: 32),
                _AdvancedSection(
                  relayCtl: _relayCtl,
                  tokenCtl: _tokenCtl,
                  userCtl: _userCtl,
                  passCtl: _passCtl,
                  onSave: _saveAdvanced,
                  status: _saveStatus,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _AdvancedSection extends StatelessWidget {
  const _AdvancedSection({
    required this.relayCtl,
    required this.tokenCtl,
    required this.userCtl,
    required this.passCtl,
    required this.onSave,
    required this.status,
  });
  final TextEditingController relayCtl;
  final TextEditingController tokenCtl;
  final TextEditingController userCtl;
  final TextEditingController passCtl;
  final VoidCallback onSave;
  final String? status;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: const [
                Icon(Icons.developer_mode, size: 16, color: Color(0xFFE63946)),
                SizedBox(width: 6),
                Text(
                  'ADVANCED',
                  style: TextStyle(
                    color: Color(0xFFE63946),
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.2,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: relayCtl,
              decoration: const InputDecoration(
                labelText: 'Relay base URL',
                hintText: 'https://aired.com/rfcai',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: tokenCtl,
              decoration: const InputDecoration(
                labelText: 'Device token',
              ),
              obscureText: true,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: userCtl,
              decoration: const InputDecoration(
                labelText: 'Labeler user',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: passCtl,
              decoration: const InputDecoration(
                labelText: 'Labeler password',
              ),
              obscureText: true,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: onSave,
              child: const Text('Save'),
            ),
            if (status != null)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(
                  status!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Color(0xFF4ADE80),
                    fontSize: 12,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
