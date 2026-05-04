import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show HapticFeedback;
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../settings.dart';
import '../theme.dart';

const _kCanonicalClasses = [
  'SMA-M', 'SMA-F',
  '3.5mm-M', '3.5mm-F',
  '2.92mm-M', '2.92mm-F',
  '2.4mm-M', '2.4mm-F',
];

// Below this top-class confidence we treat the prediction as "no real
// connector found" rather than a verdict — the classifier always emits
// a softmax over 8 classes, so on background-only frames it returns a
// best-of-noise pick that we'd otherwise show as a confident answer.
const _kMinAcceptedConfidence = 0.40;

// Background frames produce many tiny spurious Hough detections (often
// well under 1% of image area, sometimes with very high softmax conf
// because the classifier has no "background" class). A real connector
// held up to the camera occupies a much larger fraction of the frame.
// Reject any single-detection bbox below this fraction of the image.
const _kMinBboxFractionOfImage = 0.02;

enum _Mode { photo, video }

class IdentifyScreen extends StatefulWidget {
  const IdentifyScreen({super.key, required this.settings});
  final Settings settings;

  @override
  State<IdentifyScreen> createState() => _IdentifyScreenState();
}

class _IdentifyScreenState extends State<IdentifyScreen>
    with WidgetsBindingObserver {
  CameraController? _cam;
  bool _camInitFailed = false;
  bool _camInitInFlight = false;   // re-entrancy guard for _initCamera
  String? _camInitError;

  _Mode _mode = _Mode.photo;
  bool _busy = false;       // capture / classify in flight
  bool _recording = false;  // video record in progress
  String? _error;
  PredictResponse? _result;
  File? _capturedFile;      // photo or video that produced _result (native)
  Uint8List? _capturedBytes;// bytes path used on web (no usable file path)
  String? _contributionStatus;
  bool _contributing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cam?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _cam;
    if (state == AppLifecycleState.inactive
        || state == AppLifecycleState.paused) {
      // Tear down the controller and forget it. Leaving _cam pointing
      // at a disposed controller would cause _buildPreview to call
      // .value.isInitialized on a dead object on the next frame.
      // Also reset _recording so that on resume the shutter doesn't
      // try to stop a recording that the new controller never started.
      if (cam != null) {
        cam.dispose();
        if (mounted) {
          setState(() {
            _cam = null;
            _recording = false;
          });
        } else {
          _cam = null;
          _recording = false;
        }
      }
    } else if (state == AppLifecycleState.resumed) {
      if (_cam == null && !_camInitInFlight) {
        _initCamera();
      }
    }
  }

  Future<void> _initCamera() async {
    if (_camInitInFlight) return;
    _camInitInFlight = true;
    if (kIsWeb) {
      // Browser camera access through this plugin is fiddly; fall back
      // to image_picker on web (tap shutter → OS file picker).
      _camInitInFlight = false;
      if (mounted) setState(() => _camInitFailed = true);
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          setState(() {
            _camInitFailed = true;
            _camInitError = 'No cameras found on this device.';
          });
        }
        return;
      }
      final rear = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        rear,
        // Sensor's native max — connector identification benefits from
        // the highest possible resolution at the central pin/socket.
        ResolutionPreset.max,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      // Lock focus + exposure to a single autofocus pass on init so the
      // preview doesn't pulse-hunt while the user composes the shot.
      try {
        await controller.setFocusMode(FocusMode.auto);
      } catch (_) {/* not all devices support manual focus mode */}
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _cam = controller;
        _camInitFailed = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _camInitFailed = true;
        _camInitError = '$e';
      });
    } finally {
      _camInitInFlight = false;
    }
  }

  Future<void> _onShutter() async {
    if (_busy) return;
    HapticFeedback.lightImpact();
    if (_mode == _Mode.video) {
      await _toggleRecording();
    } else {
      await _capturePhoto();
    }
  }

  Future<void> _capturePhoto() async {
    final cam = _cam;
    if (cam != null && cam.value.isInitialized) {
      try {
        final shot = await cam.takePicture();
        await _classifyPhotoFile(File(shot.path));
      } catch (e) {
        setState(() => _error = 'Capture failed: $e');
      }
      return;
    }
    // Web / no-camera fallback: use image_picker. On web XFile.path is
    // a blob URL we can't open as a File, so route through bytes.
    final pf = await ImagePicker().pickImage(
      source: ImageSource.camera,
      imageQuality: 92,
      maxWidth: 4032,
    );
    if (pf == null) return;
    if (kIsWeb) {
      final bytes = await pf.readAsBytes();
      await _classifyPhotoBytes(bytes, pf.name);
    } else {
      await _classifyPhotoFile(File(pf.path));
    }
  }

  Future<void> _toggleRecording() async {
    final cam = _cam;
    if (cam == null || !cam.value.isInitialized) {
      setState(() => _error = 'Camera not available for video on this platform.');
      return;
    }
    if (!_recording) {
      try {
        await cam.startVideoRecording();
        setState(() => _recording = true);
      } catch (e) {
        setState(() => _error = 'Record start failed: $e');
      }
    } else {
      try {
        final clip = await cam.stopVideoRecording();
        setState(() => _recording = false);
        await _classifyVideo(File(clip.path));
      } catch (e) {
        setState(() {
          _recording = false;
          _error = 'Record stop failed: $e';
        });
      }
    }
  }

  Future<void> _classifyPhotoFile(File f) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _capturedFile = f;
      _capturedBytes = null;
      _contributionStatus = null;
    });
    try {
      final r = await ApiClient(widget.settings).predict(f);
      if (!mounted) return;
      _hapticOnResult(r);
      setState(() => _result = r);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _friendlyError(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _classifyPhotoBytes(Uint8List bytes, String filename) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _capturedFile = null;
      _capturedBytes = bytes;
      _contributionStatus = null;
    });
    try {
      final r = await ApiClient(widget.settings)
          .predictBytes(bytes, filename: filename);
      if (!mounted) return;
      _hapticOnResult(r);
      setState(() => _result = r);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _friendlyError(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _classifyVideo(File f) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _capturedFile = f;
      _capturedBytes = null;
      _contributionStatus = null;
    });
    try {
      final r = await ApiClient(widget.settings).predictVideo(f);
      if (!mounted) return;
      _hapticOnResult(r);
      setState(() => _result = r);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _friendlyError(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _resetToLive() {
    setState(() {
      _result = null;
      _error = null;
      _capturedFile = null;
      _capturedBytes = null;
      _contributionStatus = null;
    });
  }

  Future<void> _contribute(String cls) async {
    // Video contribution would need a different endpoint; only photos
    // route to /labeler/upload-train today.
    if (_mode == _Mode.video) {
      setState(() => _contributionStatus =
          'Video contribution: open Contribute tab and use the Video uploader.');
      return;
    }
    final file = _capturedFile;
    final bytes = _capturedBytes;
    if (file == null && bytes == null) return;
    setState(() {
      _contributing = true;
      _contributionStatus = null;
    });
    try {
      final api = ApiClient(widget.settings);
      if (file != null) {
        await api.uploadTrainingPhoto(file, cls);
      } else {
        await api.uploadTrainingPhotoBytes(bytes!, cls);
      }
      if (!mounted) return;
      setState(() => _contributionStatus = '✓ Added to training as $cls');
    } catch (e) {
      if (!mounted) return;
      setState(() => _contributionStatus = 'Upload failed: ${_friendlyError(e)}');
    } finally {
      if (mounted) setState(() => _contributing = false);
    }
  }

  /// Buzz on prediction land. Confident detection => medium impact;
  /// no-detection / low-conf => short selection-style click.
  void _hapticOnResult(PredictResponse r) {
    final imageArea = r.imageWidth * r.imageHeight;
    final goodHit = r.predictions.any((p) {
      if (p.confidence < _kMinAcceptedConfidence) return false;
      if (imageArea <= 0) return true;
      final area = p.bbox['w']! * p.bbox['h']!;
      return area / imageArea >= _kMinBboxFractionOfImage;
    });
    if (goodHit) {
      HapticFeedback.mediumImpact();
    } else {
      HapticFeedback.selectionClick();
    }
  }

  /// Map raw exception/_ApiError chatter to short user-readable messages.
  String _friendlyError(Object e) {
    final s = e.toString();
    if (s.contains('SocketException') || s.contains('Failed host lookup')) {
      return 'No connection — check Wi-Fi and the relay URL in Settings.';
    }
    if (s.contains('TimeoutException') || s.contains('timed out')) {
      return 'Server took too long — try again or pick a smaller image.';
    }
    if (s.contains('HandshakeException') || s.contains('CERTIFICATE')) {
      return 'TLS handshake failed — relay URL may be wrong.';
    }
    if (s.contains('ApiError 401')) {
      return 'Auth failed — check device token in Settings.';
    }
    if (s.contains('ApiError 403')) {
      return 'Forbidden — token may have been rotated.';
    }
    if (s.contains('ApiError 413')) {
      return 'Image too large for the server.';
    }
    if (s.contains('ApiError 422')) {
      return 'Server rejected the image format.';
    }
    if (s.contains('ApiError 503')) {
      return 'Classifier not loaded yet on the server — wait and retry.';
    }
    if (s.contains('ApiError')) {
      return 'Server error — try again.';
    }
    return s.length > 200 ? '${s.substring(0, 200)}…' : s;
  }

  Future<void> _pickCorrectClass(String suggested) async {
    final picked = await showDialog<String>(
      context: context,
      builder: (_) => SimpleDialog(
        title: const Text('Add to training as…'),
        children: _kCanonicalClasses
            .map((c) => SimpleDialogOption(
                  onPressed: () => Navigator.pop(context, c),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Text(
                      c,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: c == suggested
                            ? FontWeight.bold
                            : FontWeight.normal,
                      ),
                    ),
                  ),
                ))
            .toList(),
      ),
    );
    if (picked != null) await _contribute(picked);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildPreview(),
          // Centered target reticle to help users frame the connector
          // — only visible on the live preview (hide when showing
          // a captured photo + result).
          if (_result == null && _error == null && !_busy && !_camInitFailed)
            const IgnorePointer(child: _ReticleOverlay()),
          // Mode toggle (photo / video) — hidden while showing a result.
          if (_result == null && _error == null)
            Positioned(
              top: MediaQuery.of(context).padding.top + 12,
              left: 0,
              right: 0,
              child: Center(child: _buildModeToggle()),
            ),
          // Bottom shutter (or back-to-live when result is showing).
          Positioned(
            left: 0,
            right: 0,
            bottom: MediaQuery.of(context).padding.bottom + 100,
            child: Center(child: _buildShutterArea()),
          ),
          // Result overlay slides up from the bottom over the frozen frame.
          if (_result != null || _error != null)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 96),
                  child: _buildResultPanel(),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    // When we have a result, freeze the captured photo on screen instead
    // of showing the live preview.
    if (_result != null && _mode == _Mode.photo) {
      if (_capturedBytes != null) {
        return Image.memory(_capturedBytes!,
            fit: BoxFit.cover, alignment: Alignment.center);
      }
      if (_capturedFile != null) {
        return Image.file(_capturedFile!,
            fit: BoxFit.cover, alignment: Alignment.center);
      }
    }
    final cam = _cam;
    if (cam != null && cam.value.isInitialized) {
      // Sensor returns landscape dimensions (width = sensor long edge);
      // for portrait UI we have to swap to get the right aspect, then
      // FittedBox.cover lets the image fill the screen without
      // squishing. Without this, CameraPreview alone shows a stretched
      // / wrong-orientation feed on most Androids.
      final preview = cam.value.previewSize;
      final pw = preview?.height ?? 1;  // swap intentional
      final ph = preview?.width ?? 1;
      return ClipRect(
        child: SizedBox.expand(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: pw,
              height: ph,
              child: CameraPreview(cam),
            ),
          ),
        ),
      );
    }
    if (_camInitFailed) {
      return Container(
        color: Colors.black,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.photo_camera_outlined,
                    size: 72, color: Colors.white38),
                const SizedBox(height: 16),
                Text(
                  kIsWeb
                      ? 'Live preview not available in browser.\nTap the shutter to pick a photo.'
                      : (_camInitError ?? 'Camera unavailable.'),
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white54, fontSize: 14),
                ),
              ],
            ),
          ),
        ),
      );
    }
    return const Center(
      child: CircularProgressIndicator(color: Colors.white),
    );
  }

  Widget _buildModeToggle() {
    Widget pill(_Mode m, IconData icon, String label) {
      final selected = _mode == m;
      return GestureDetector(
        onTap: _busy || _recording
            ? null
            : () => setState(() => _mode = m),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: selected ? Colors.white : Colors.transparent,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 16,
                  color: selected ? Colors.black : Colors.white),
              const SizedBox(width: 6),
              Text(label,
                  style: TextStyle(
                    color: selected ? Colors.black : Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  )),
            ],
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          pill(_Mode.photo, Icons.photo_camera, 'Photo'),
          const SizedBox(width: 4),
          pill(_Mode.video, Icons.videocam, 'Video'),
        ],
      ),
    );
  }

  Widget _buildShutterArea() {
    if (_result != null || _error != null) {
      // Result is up; primary action is "scan again".
      return _RoundIconButton(
        icon: Icons.refresh,
        label: 'Scan again',
        onTap: _resetToLive,
      );
    }
    if (_busy) {
      return const _ShutterButton(
        recording: false,
        videoMode: false,
        busy: true,
      );
    }
    return _ShutterButton(
      recording: _recording,
      videoMode: _mode == _Mode.video,
      busy: false,
      onTap: _onShutter,
    );
  }

  Widget _buildResultPanel() {
    if (_error != null) {
      return _ResultCard(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Identification failed',
                      style: TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text(_error!,
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 13)),
                ],
              ),
            ),
          ],
        ),
      );
    }
    final r = _result!;
    final imageArea = r.imageWidth * r.imageHeight;
    // Sort first so we know the unfiltered top, then apply the
    // bbox-fraction filter. Distinguishing "server returned nothing"
    // vs "we filtered out a tiny-bbox detection" gives the user better
    // guidance ("move closer" vs "no connector here").
    final sortedAll = [...r.predictions]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final sorted = sortedAll
        .where((p) {
          if (imageArea <= 0) return true;
          final area = p.bbox['w']! * p.bbox['h']!;
          return area / imageArea >= _kMinBboxFractionOfImage;
        })
        .toList();
    if (sorted.isEmpty || sorted.first.confidence < _kMinAcceptedConfidence) {
      final filteredOutTinyHigh = sorted.isEmpty
          && sortedAll.any((p) => p.confidence >= _kMinAcceptedConfidence);
      final String hint;
      final String subhint;
      if (sortedAll.isEmpty) {
        hint = 'No connector detected.';
        subhint = 'Hold the connector face-on, center it, fill the frame.';
      } else if (filteredOutTinyHigh) {
        hint = 'Connector too small in frame.';
        subhint = 'Move the camera closer so the connector face fills the frame.';
      } else {
        hint = 'No clear connector — best guess was '
               '${sortedAll.first.className} at '
               '${(sortedAll.first.confidence * 100).toStringAsFixed(0)}%.';
        subhint = 'Hold the connector face-on, center it, fill more of the frame.';
      }
      return _ResultCard(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Column(
            children: [
              Text(
                hint,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 8),
              Text(
                subhint,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 12, color: Colors.white70),
              ),
            ],
          ),
        ),
      );
    }
    final p = sorted.first;
    final hot = p.isMale;
    final color = hot ? kHotDogColor : kNotHotDogColor;
    final label = hot ? 'HOT DOG' : 'NOT HOT DOG';
    return _ResultCard(
      child: Column(
        children: [
          Text(label,
              style: TextStyle(
                color: color,
                fontSize: 38,
                fontWeight: FontWeight.w900,
                letterSpacing: 1.0,
              )),
          const SizedBox(height: 8),
          Text(p.family,
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w600,
              )),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              '${(p.confidence * 100).toStringAsFixed(0)}% confidence  ·  ${p.className}',
              style: TextStyle(
                  color: color, fontSize: 12, fontWeight: FontWeight.w600),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _contributing ? null : () => _contribute(p.className),
                  icon: const Icon(Icons.check, size: 18),
                  label: Text('Confirm ${p.className}'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF4ADE80),
                    foregroundColor: Colors.black,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _contributing
                      ? null
                      : () => _pickCorrectClass(p.className),
                  icon: const Icon(Icons.edit, size: 18),
                  label: const Text('Correct…'),
                ),
              ),
            ],
          ),
          if (_contributing)
            const Padding(
              padding: EdgeInsets.only(top: 10),
              child: Center(child: CircularProgressIndicator()),
            ),
          if (_contributionStatus != null)
            Padding(
              padding: const EdgeInsets.only(top: 10),
              child: Text(
                _contributionStatus!,
                style: TextStyle(
                  fontSize: 12,
                  color: _contributionStatus!.startsWith('✓')
                      ? const Color(0xFF4ADE80)
                      : Colors.redAccent,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.child});
  final Widget child;
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.78),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white24),
      ),
      child: child,
    );
  }
}

class _ShutterButton extends StatelessWidget {
  const _ShutterButton({
    required this.recording,
    required this.videoMode,
    required this.busy,
    this.onTap,
  });
  final bool recording;
  final bool videoMode;
  final bool busy;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final outerColor = Colors.white.withOpacity(0.9);
    final innerColor = videoMode ? Colors.redAccent : Colors.white;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 78,
        height: 78,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: outerColor, width: 4),
        ),
        child: Center(
          child: busy
              ? const SizedBox(
                  width: 28, height: 28,
                  child: CircularProgressIndicator(
                      strokeWidth: 3, color: Colors.white))
              : AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  width: recording ? 28 : 60,
                  height: recording ? 28 : 60,
                  decoration: BoxDecoration(
                    color: innerColor,
                    borderRadius: BorderRadius.circular(recording ? 6 : 30),
                  ),
                ),
        ),
      ),
    );
  }
}

class _ReticleOverlay extends StatelessWidget {
  const _ReticleOverlay();

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _ReticlePainter(),
      child: const SizedBox.expand(),
    );
  }
}

class _ReticlePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = math.min(size.width, size.height) * 0.28;
    final ringPaint = Paint()
      ..color = Colors.white.withOpacity(0.55)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    final dashPaint = Paint()
      ..color = Colors.white.withOpacity(0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    // Outer guide ring — where the connector face should fit.
    canvas.drawCircle(Offset(cx, cy), r, ringPaint);
    // Four corner ticks at the inscribed-square corners — gives a
    // sharper target than a single circle without cluttering the view.
    final tick = r * 0.18;
    final s = r / math.sqrt(2);   // half-side of inscribed square
    for (final sx in const [-1.0, 1.0]) {
      for (final sy in const [-1.0, 1.0]) {
        final ax = cx + sx * s;
        final ay = cy + sy * s;
        canvas.drawLine(Offset(ax, ay), Offset(ax - sx * tick, ay), dashPaint);
        canvas.drawLine(Offset(ax, ay), Offset(ax, ay - sy * tick), dashPaint);
      }
    }
    // Tiny crosshair in the dead center.
    final ch = r * 0.08;
    canvas.drawLine(Offset(cx - ch, cy), Offset(cx + ch, cy), dashPaint);
    canvas.drawLine(Offset(cx, cy - ch), Offset(cx, cy + ch), dashPaint);

    // "CENTER CONNECTOR" hint above the reticle.
    final tp = TextPainter(
      text: TextSpan(
        text: 'CENTER CONNECTOR',
        style: TextStyle(
          color: Colors.white.withOpacity(0.7),
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.5,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    tp.layout();
    tp.paint(canvas, Offset(cx - tp.width / 2, cy - r - tp.height - 8));
  }

  @override
  bool shouldRepaint(covariant _ReticlePainter oldDelegate) => false;
}

class _RoundIconButton extends StatelessWidget {
  const _RoundIconButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 64, height: 64,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.18),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white70, width: 2),
            ),
            child: Icon(icon, color: Colors.white, size: 28),
          ),
        ),
        const SizedBox(height: 6),
        Text(label,
            style: const TextStyle(
                color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
      ],
    );
  }
}
