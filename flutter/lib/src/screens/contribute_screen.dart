import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show HapticFeedback;
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../settings.dart';

const _kFamilies = ['SMA', '3.5mm', '2.92mm', '2.4mm', '1.85mm'];
const _kGenders = ['M', 'F'];

/// Camera-first capture screen. Set the class via chips, tap shutter,
/// upload happens in the background while the camera stays live so the
/// next shot is one tap away. No predict round-trip.
class ContributeScreen extends StatefulWidget {
  const ContributeScreen({
    super.key,
    required this.settings,
    this.isActive = true,
  });
  final Settings settings;
  // False when sitting in an IndexedStack but on a non-selected tab.
  // We tear the camera down so Identify can have it.
  final bool isActive;

  @override
  State<ContributeScreen> createState() => _ContributeScreenState();
}

class _ContributeScreenState extends State<ContributeScreen>
    with WidgetsBindingObserver {
  CameraController? _cam;
  bool _camInitFailed = false;
  bool _camInitInFlight = false;
  String? _camInitError;

  // Sticky class selection — set once, keep shooting.
  String _family = '2.4mm';
  String _gender = 'M';

  // When true, next captures land in data/test_holdout/<class>/ instead
  // of the training set. Default off because most contributions grow
  // training; toggling on is the explicit "this is for evaluation" act.
  bool _holdout = false;

  bool _shutterBusy = false;     // single-flight guard on capture
  int _uploadInFlight = 0;       // count of background uploads posting now
  int _uploadedCount = 0;        // session total successful uploads
  String? _toast;                // transient status pill above the chips
  bool _toastIsError = false;
  Timer? _toastTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    if (widget.isActive) _initCamera();
  }

  @override
  void didUpdateWidget(covariant ContributeScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.isActive && !widget.isActive) {
      final cam = _cam;
      if (cam != null) {
        cam.dispose();
        _cam = null;
      }
    } else if (!oldWidget.isActive && widget.isActive) {
      if (_cam == null && !_camInitInFlight) _initCamera();
    }
  }

  @override
  void dispose() {
    _toastTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    _cam?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _cam;
    if (state == AppLifecycleState.inactive
        || state == AppLifecycleState.paused) {
      if (cam != null) {
        cam.dispose();
        if (mounted) {
          setState(() => _cam = null);
        } else {
          _cam = null;
        }
      }
    } else if (state == AppLifecycleState.resumed && widget.isActive) {
      if (_cam == null && !_camInitInFlight) _initCamera();
    }
  }

  Future<void> _initCamera() async {
    if (_camInitInFlight) return;
    _camInitInFlight = true;
    if (kIsWeb) {
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
            _camInitError = 'No cameras found.';
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
        ResolutionPreset.max,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      try { await controller.setFocusMode(FocusMode.auto); } catch (_) {}
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

  String get _classLabel => '$_family-$_gender';

  Future<void> _onShutter() async {
    if (_shutterBusy) return;
    setState(() => _shutterBusy = true);
    HapticFeedback.lightImpact();
    final cam = _cam;
    try {
      if (cam != null && cam.value.isInitialized) {
        final shot = await cam.takePicture();
        // Fire-and-forget upload so the camera is ready for the next shot.
        unawaited(_uploadFile(File(shot.path)));
      } else {
        // Web / no-camera fallback — OS camera dialog.
        final pf = await ImagePicker().pickImage(
          source: ImageSource.camera,
          imageQuality: 92,
          maxWidth: 4032,
        );
        if (pf == null) return;
        if (kIsWeb) {
          final bytes = await pf.readAsBytes();
          unawaited(_uploadBytes(bytes, pf.name));
        } else {
          unawaited(_uploadFile(File(pf.path)));
        }
      }
    } catch (e) {
      _showToast('Capture failed: ${_friendlyError(e)}', error: true);
    } finally {
      if (mounted) setState(() => _shutterBusy = false);
    }
  }

  Future<void> _pickFromGallery() async {
    if (_shutterBusy) return;
    final pf = await ImagePicker().pickImage(
      source: ImageSource.gallery,
      imageQuality: 92,
      maxWidth: 4032,
    );
    if (pf == null) return;
    if (kIsWeb) {
      final bytes = await pf.readAsBytes();
      unawaited(_uploadBytes(bytes, pf.name));
    } else {
      unawaited(_uploadFile(File(pf.path)));
    }
  }

  Future<void> _pickAndUploadVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
      withData: kIsWeb,
    );
    if (result == null) return;
    final picked = result.files.single;
    if (kIsWeb) {
      _showToast('Video upload only on mobile.', error: true);
      return;
    }
    if (picked.path == null) return;
    if (mounted) setState(() => _uploadInFlight++);
    try {
      final api = ApiClient(widget.settings);
      await api.uploadTrainingVideo(File(picked.path!), _family, _gender);
      if (!mounted) return;
      _showToast('✓ Video sent — server is extracting crops');
    } catch (e) {
      _showToast('Video upload failed: ${_friendlyError(e)}', error: true);
    } finally {
      if (mounted) setState(() => _uploadInFlight--);
    }
  }

  // Uploads happen async; we capture the class + holdout intent at the
  // moment of capture so the user can flip chips for the next shot
  // without retroactively re-routing pending uploads.
  Future<void> _uploadFile(File f) async {
    final cls = _classLabel;
    final isHoldout = _holdout;
    if (mounted) setState(() => _uploadInFlight++);
    try {
      final api = ApiClient(widget.settings);
      if (isHoldout) {
        await api.uploadTestHoldoutPhoto(f, cls);
      } else {
        await api.uploadTrainingPhoto(f, cls);
      }
      if (!mounted) return;
      setState(() => _uploadedCount++);
      _showToast('✓ #$_uploadedCount $cls${isHoldout ? " · holdout" : ""}');
      HapticFeedback.selectionClick();
    } catch (e) {
      _showToast('Upload failed: ${_friendlyError(e)}', error: true);
    } finally {
      if (mounted) setState(() => _uploadInFlight--);
    }
  }

  Future<void> _uploadBytes(Uint8List bytes, String filename) async {
    final cls = _classLabel;
    final isHoldout = _holdout;
    if (mounted) setState(() => _uploadInFlight++);
    try {
      final api = ApiClient(widget.settings);
      if (isHoldout) {
        await api.uploadTestHoldoutPhotoBytes(bytes, cls, filename: filename);
      } else {
        await api.uploadTrainingPhotoBytes(bytes, cls, filename: filename);
      }
      if (!mounted) return;
      setState(() => _uploadedCount++);
      _showToast('✓ #$_uploadedCount $cls${isHoldout ? " · holdout" : ""}');
    } catch (e) {
      _showToast('Upload failed: ${_friendlyError(e)}', error: true);
    } finally {
      if (mounted) setState(() => _uploadInFlight--);
    }
  }

  void _showToast(String msg, {bool error = false}) {
    if (!mounted) return;
    _toastTimer?.cancel();
    setState(() {
      _toast = msg;
      _toastIsError = error;
    });
    _toastTimer = Timer(Duration(seconds: error ? 4 : 2), () {
      if (!mounted) return;
      setState(() => _toast = null);
    });
  }

  String _friendlyError(Object e) {
    final s = e.toString();
    if (s.contains('SocketException') || s.contains('Failed host lookup')) {
      return 'No connection — check Wi-Fi.';
    }
    if (s.contains('TimeoutException') || s.contains('timed out')) {
      return 'Server slow — try again.';
    }
    if (s.contains('ApiError 401') || s.contains('ApiError 403')) {
      return 'Auth failed — check token in Settings.';
    }
    if (s.contains('ApiError 413')) {
      return 'Image too large.';
    }
    return s.length > 100 ? '${s.substring(0, 100)}…' : s;
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).padding.bottom;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildPreview(),
          Positioned(
            top: MediaQuery.of(context).padding.top + 12,
            left: 12,
            right: 12,
            child: _buildTopBar(),
          ),
          // Toast appears between the chips and the camera so it doesn't
          // get clipped by the bottom nav bar.
          if (_toast != null)
            Positioned(
              left: 32,
              right: 32,
              // Sits ~10px above the chip strip block.
              bottom: bottomInset + 296,
              child: _StatusPill(text: _toast!, error: _toastIsError),
            ),
          Positioned(
            left: 0,
            right: 0,
            // Clear the bottom NavigationBar (~80px on Material 3).
            bottom: bottomInset + 90,
            child: _buildControls(),
          ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    final cam = _cam;
    if (cam != null && cam.value.isInitialized) {
      // Sensor returns landscape dimensions; swap for portrait UI then
      // FittedBox.cover lets the preview fill without squishing. Same
      // pattern Identify uses.
      final preview = cam.value.previewSize;
      final pw = preview?.height ?? 1;
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
                      ? 'Live preview not available in browser.\nTap shutter to pick a photo.'
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

  Widget _buildTopBar() {
    return Row(
      children: [
        _CounterPill(
          count: _uploadedCount,
          inFlight: _uploadInFlight,
        ),
        const Spacer(),
        _HoldoutToggle(
          on: _holdout,
          onTap: () => setState(() => _holdout = !_holdout),
        ),
      ],
    );
  }

  Widget _buildControls() {
    Widget chip(String label, bool selected, VoidCallback onTap) {
      return GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 100),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: selected ? Colors.white : Colors.black.withOpacity(0.5),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: selected ? Colors.white : Colors.white24,
            ),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: selected ? Colors.black : Colors.white,
              fontSize: 13,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
            ),
          ),
        ),
      );
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Wrap(
          spacing: 6,
          alignment: WrapAlignment.center,
          children: _kFamilies
              .map((f) => chip(f, _family == f,
                  () => setState(() => _family = f)))
              .toList(),
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 6,
          alignment: WrapAlignment.center,
          children: _kGenders
              .map((g) => chip(g, _gender == g,
                  () => setState(() => _gender = g)))
              .toList(),
        ),
        const SizedBox(height: 12),
        // Class-preview pill — confirms what the next shot will be saved as.
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.6),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: _holdout
                  ? const Color(0xFFFFB347)
                  : Colors.white24,
            ),
          ),
          child: Text(
            _holdout
                ? 'next: $_classLabel  →  HOLDOUT'
                : 'next: $_classLabel',
            style: TextStyle(
              color: _holdout ? const Color(0xFFFFB347) : Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: 12),
        _ShutterButton(busy: _shutterBusy, onTap: _onShutter),
        const SizedBox(height: 10),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _SmallAction(
              icon: Icons.photo_library,
              label: 'Gallery',
              onTap: _pickFromGallery,
            ),
            const SizedBox(width: 24),
            _SmallAction(
              icon: Icons.videocam,
              label: 'Video',
              onTap: _pickAndUploadVideo,
            ),
          ],
        ),
      ],
    );
  }
}

class _CounterPill extends StatelessWidget {
  const _CounterPill({required this.count, required this.inFlight});
  final int count;
  final int inFlight;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (inFlight > 0)
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white,
              ),
            )
          else
            const Icon(Icons.cloud_upload_outlined,
                size: 14, color: Colors.white70),
          const SizedBox(width: 6),
          Text(
            inFlight > 0
                ? '$count uploaded · $inFlight…'
                : '$count uploaded',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _HoldoutToggle extends StatelessWidget {
  const _HoldoutToggle({required this.on, required this.onTap});
  final bool on;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: on
              ? const Color(0xFFFFB347)
              : Colors.black.withOpacity(0.6),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: on ? const Color(0xFFFFB347) : Colors.white24,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              on ? Icons.science : Icons.school_outlined,
              size: 14,
              color: on ? Colors.black : Colors.white70,
            ),
            const SizedBox(width: 6),
            Text(
              on ? 'HOLDOUT' : 'training',
              style: TextStyle(
                color: on ? Colors.black : Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ShutterButton extends StatelessWidget {
  const _ShutterButton({required this.busy, required this.onTap});
  final bool busy;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: busy ? null : onTap,
      child: Container(
        width: 78,
        height: 78,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.9), width: 4),
        ),
        child: Center(
          child: busy
              ? const SizedBox(
                  width: 28,
                  height: 28,
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: Colors.white,
                  ),
                )
              : Container(
                  width: 60,
                  height: 60,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                ),
        ),
      ),
    );
  }
}

class _SmallAction extends StatelessWidget {
  const _SmallAction({
    required this.icon,
    required this.label,
    required this.onTap,
  });
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.55),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white24),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: Colors.white70),
            const SizedBox(width: 6),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.text, required this.error});
  final String text;
  final bool error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: error
              ? Colors.redAccent.withOpacity(0.95)
              : const Color(0xFF4ADE80).withOpacity(0.95),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          text,
          textAlign: TextAlign.center,
          style: TextStyle(
            color: error ? Colors.white : Colors.black,
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }
}
