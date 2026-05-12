import 'dart:async';

import 'package:flutter/foundation.dart' show defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show HapticFeedback;
import 'package:package_info_plus/package_info_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import '../settings.dart';

/// "About" tab — visible to every user. The main interaction here is
/// **Request a new connector type** (a mailto form). Branding shrinks
/// to a small "Powered by aired.com" footer beneath. The version
/// string also doubles as the dev-mode unlock: 7 taps in quick
/// succession flips dev mode, which reveals the Contribute tab and
/// the Advanced (relay/token/labeler) panel.
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

// Where new-connector requests land. Editable via the Advanced section
// later if we want a separate inbox.
const _kRequestEmail = 'chris@aired.com';

class _AboutScreenState extends State<AboutScreen> {
  String _version = '';
  String _build = '';

  // Dev-mode unlock — Android-style "tap version 7 times" gesture.
  int _devTapCount = 0;
  Timer? _devTapTimer;

  // Request-a-connector form.
  final _requestNameCtl = TextEditingController();
  final _requestNotesCtl = TextEditingController();
  bool _requestSending = false;

  // Advanced (dev-mode-gated) panel.
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
    _requestNameCtl.dispose();
    _requestNotesCtl.dispose();
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
    _devTapTimer = Timer(const Duration(seconds: 2), () {
      _devTapCount = 0;
    });

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

  Future<void> _sendConnectorRequest() async {
    final name = _requestNameCtl.text.trim();
    if (name.isEmpty) {
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(const SnackBar(
          content: Text('Tell us which connector you want — name is required.'),
        ));
      return;
    }
    setState(() => _requestSending = true);
    try {
      final notes = _requestNotesCtl.text.trim();
      final platform =
          '${defaultTargetPlatform.name}  ($_version+$_build)';
      final body = StringBuffer()
        ..writeln('Hi,')
        ..writeln()
        ..writeln('I would like Connector ID to support: $name')
        ..writeln();
      if (notes.isNotEmpty) {
        body
          ..writeln('Notes:')
          ..writeln(notes)
          ..writeln();
      }
      body
        ..writeln('—')
        ..writeln('Sent from Connector ID v$_version ($_build) on $platform');

      final uri = Uri(
        scheme: 'mailto',
        path: _kRequestEmail,
        queryParameters: {
          'subject': 'Connector request: $name',
          'body': body.toString(),
        },
      );

      // Some platforms refuse to canLaunchUrl mailto unless an email app
      // is registered; just try launchUrl and surface the failure.
      final launched = await launchUrl(uri, mode: LaunchMode.externalApplication);
      if (!mounted) return;
      if (launched) {
        _requestNameCtl.clear();
        _requestNotesCtl.clear();
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(const SnackBar(
            content: Text('Opened your email app — review and send.'),
          ));
      } else {
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(const SnackBar(
            content: Text(
              'No email app available — copy the request and email '
              '$_kRequestEmail.',
            ),
          ));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(SnackBar(
          content: Text('Could not open email: $e'),
        ));
    } finally {
      if (mounted) setState(() => _requestSending = false);
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
          padding: const EdgeInsets.fromLTRB(20, 28, 20, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Compact hero — small mark, app name, tap-to-unlock version.
              Center(
                child: Column(
                  children: [
                    Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(
                        color: const Color(0xFFE63946),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Center(
                        child: Text(
                          'ai',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 28,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Connector ID',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 2),
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
                            color: theme.colorScheme.onSurface
                                .withOpacity(0.5),
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // One-line description, intentionally short.
              Center(
                child: Text(
                  'Identify RF coaxial connectors with your phone.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13,
                    color: theme.colorScheme.onSurface.withOpacity(0.7),
                  ),
                ),
              ),

              const SizedBox(height: 28),

              // ── Main feature: request a new connector type ──
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: const [
                          Icon(Icons.add_circle_outline,
                              size: 18, color: Color(0xFFE63946)),
                          SizedBox(width: 8),
                          Text(
                            'Request a new connector type',
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      Text(
                        'Don\'t see the connector you need? Tell us '
                        'which one to add next.',
                        style: TextStyle(
                          color: theme.colorScheme.onSurface.withOpacity(0.6),
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 14),
                      TextField(
                        controller: _requestNameCtl,
                        textInputAction: TextInputAction.next,
                        decoration: const InputDecoration(
                          labelText: 'Connector name',
                          hintText: 'e.g. BNC, N-type, TNC, MMCX',
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _requestNotesCtl,
                        minLines: 2,
                        maxLines: 4,
                        textInputAction: TextInputAction.newline,
                        decoration: const InputDecoration(
                          labelText: 'Notes (optional)',
                          hintText: 'Use case, frequency range, vendor…',
                        ),
                      ),
                      const SizedBox(height: 14),
                      ElevatedButton.icon(
                        onPressed:
                            _requestSending ? null : _sendConnectorRequest,
                        icon: _requestSending
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.send, size: 16),
                        label: const Text('Send request'),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 20),

              // ── Privacy disclosure ──
              const _PrivacySection(),

              const SizedBox(height: 24),

              // ── Small "Powered by aired.com" footer ──
              GestureDetector(
                onTap: _openAired,
                behavior: HitTestBehavior.opaque,
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          color: const Color(0xFFE63946),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: const Center(
                          child: Text(
                            'ai',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 9,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -0.5,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Powered by aired.com',
                        style: TextStyle(
                          color: theme.colorScheme.onSurface.withOpacity(0.65),
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 4),
                      Icon(
                        Icons.open_in_new,
                        size: 12,
                        color: theme.colorScheme.onSurface.withOpacity(0.45),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 6),
              Center(
                child: Text(
                  '© aired.com',
                  style: TextStyle(
                    color: theme.colorScheme.onSurface.withOpacity(0.4),
                    fontSize: 11,
                  ),
                ),
              ),

              // Advanced section — only visible in dev mode.
              if (widget.settings.devMode) ...[
                const SizedBox(height: 28),
                _OnDeviceToggle(
                  settings: widget.settings,
                  onChanged: () => setState(() {}),
                ),
                const SizedBox(height: 12),
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

class _PrivacySection extends StatefulWidget {
  const _PrivacySection();

  @override
  State<_PrivacySection> createState() => _PrivacySectionState();
}

class _PrivacySectionState extends State<_PrivacySection> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.colorScheme.onSurface.withOpacity(0.7);
    return AnimatedSize(
      duration: const Duration(milliseconds: 180),
      curve: Curves.easeOut,
      alignment: Alignment.topCenter,
      child: GestureDetector(
        onTap: () => setState(() => _expanded = !_expanded),
        behavior: HitTestBehavior.opaque,
        child: Container(
          padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: const Color(0xFF2A313E)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.lock_outline,
                      size: 14,
                      color: theme.colorScheme.onSurface.withOpacity(0.6)),
                  const SizedBox(width: 8),
                  Text(
                    'Privacy',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: theme.colorScheme.onSurface.withOpacity(0.85),
                    ),
                  ),
                  const Spacer(),
                  Icon(
                    _expanded ? Icons.expand_less : Icons.expand_more,
                    size: 16,
                    color: theme.colorScheme.onSurface.withOpacity(0.5),
                  ),
                ],
              ),
              if (_expanded) ...[
                const SizedBox(height: 10),
                _bullet(
                  'Photos you take in Identify are sent to aired.com over '
                  'HTTPS for classification, processed in memory, and '
                  'discarded after the response. No image is stored on '
                  'our servers.',
                  muted,
                ),
                const SizedBox(height: 8),
                _bullet(
                  'Connector requests open your phone\'s email app — your '
                  'message is sent through your normal email account to '
                  'chris@aired.com. The app does not relay or intercept it.',
                  muted,
                ),
                const SizedBox(height: 8),
                _bullet(
                  'No accounts, no analytics, no advertising IDs. Local '
                  'preferences (server URL, device token) are stored '
                  'on-device only.',
                  muted,
                ),
                const SizedBox(height: 8),
                _bullet(
                  'Questions or removal requests: email '
                  'chris@aired.com.',
                  muted,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  static Widget _bullet(String text, Color muted) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 6, right: 8),
          child: Container(
            width: 4,
            height: 4,
            decoration: BoxDecoration(
              color: muted,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
        ),
        Expanded(
          child: Text(
            text,
            style: TextStyle(
              fontSize: 11.5,
              height: 1.45,
              color: muted,
            ),
          ),
        ),
      ],
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

/// Toggle for the on-device classifier path. Only shown in dev mode.
/// When on, /predict calls bypass aired.com and run the bundled
/// ResNet-18 ONNX locally.
class _OnDeviceToggle extends StatelessWidget {
  const _OnDeviceToggle({required this.settings, required this.onChanged});
  final Settings settings;
  final VoidCallback onChanged;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: SwitchListTile(
        dense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16),
        title: const Text('On-device inference',
            style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700)),
        subtitle: const Text(
          'Run the classifier locally instead of POSTing to aired.com. '
          'No network round-trip, works offline. Accuracy may differ — '
          'currently no rembg / Hough preprocessing on this path.',
          style: TextStyle(fontSize: 11),
        ),
        value: settings.onDeviceMode,
        onChanged: (v) async {
          settings.onDeviceMode = v;
          await settings.save();
          onChanged();
        },
      ),
    );
  }
}

