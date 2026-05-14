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
import '../ondevice/classifier.dart';
import '../settings.dart';

const _kFamilies = ['SMA', '3.5mm', '2.92mm', '2.4mm', '1.85mm'];
const _kGenders = ['M', 'F'];

class _SessionRecord {
  _SessionRecord({required this.record, required this.holdout});
  final UploadRecord record;
  final bool holdout;
}

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
  final Map<String, int> _sessionCounts = {};   // training uploads per class
  final Map<String, int> _sessionHoldout = {};  // holdout uploads per class
  // Tail-ordered stack of server-acked uploads in this session.
  // Tapping Undo pops + DELETEs the tail. Capped at 500 to bound memory
  // — well above any plausible per-session capture volume.
  final List<_SessionRecord> _undoStack = [];
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
    // Fire-and-forget warm-up — singleton means Identify benefits too.
    unawaited(OnDeviceClassifier.instance.init().catchError((_) {}));
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
        unawaited(_runOnDeviceCheck(File(shot.path), _classLabel));
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
      final UploadResult result = isHoldout
          ? await api.uploadTestHoldoutPhoto(f, cls)
          : await api.uploadTrainingPhoto(f, cls);
      if (!mounted) return;
      setState(() {
        for (final rec in result.saved) {
          _undoStack.add(_SessionRecord(record: rec, holdout: isHoldout));
          if (_undoStack.length > 500) _undoStack.removeAt(0);
          _uploadedCount++;
          final m = isHoldout ? _sessionHoldout : _sessionCounts;
          m[rec.cls] = (m[rec.cls] ?? 0) + 1;
        }
      });
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
      final UploadResult result = isHoldout
          ? await api.uploadTestHoldoutPhotoBytes(bytes, cls, filename: filename)
          : await api.uploadTrainingPhotoBytes(bytes, cls, filename: filename);
      if (!mounted) return;
      setState(() {
        for (final rec in result.saved) {
          _undoStack.add(_SessionRecord(record: rec, holdout: isHoldout));
          if (_undoStack.length > 500) _undoStack.removeAt(0);
          _uploadedCount++;
          final m = isHoldout ? _sessionHoldout : _sessionCounts;
          m[rec.cls] = (m[rec.cls] ?? 0) + 1;
        }
      });
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

  Future<void> _undoLast() async {
    if (_undoStack.isEmpty) return;
    final last = _undoStack.removeLast();
    if (mounted) setState(() {});
    try {
      await ApiClient(widget.settings).deleteLabelerFile(last.record.path);
      if (!mounted) return;
      setState(() {
        _uploadedCount = (_uploadedCount - 1).clamp(0, 1 << 30);
        final m = last.holdout ? _sessionHoldout : _sessionCounts;
        if ((m[last.record.cls] ?? 0) > 0) {
          m[last.record.cls] = m[last.record.cls]! - 1;
        }
      });
      _showToast('↩ Undone ${last.record.cls}');
    } catch (e) {
      if (mounted) setState(() => _undoStack.add(last));
      _showToast('Undo failed: ${_friendlyError(e)}', error: true);
    }
  }

  Future<void> _runOnDeviceCheck(File f, String cls) async {
    try {
      final bytes = await f.readAsBytes();
      final pred = await OnDeviceClassifier.instance.predict(bytes);
      if (!mounted) return;
      if (pred.className == cls) {
        // Model agrees with the chip — the upload's success toast is
        // sufficient. Don't fire a redundant toast that would race the
        // upload's setState and confuse the displayed counter.
        return;
      }
      final selFamily = cls.contains('-')
          ? cls.substring(0, cls.lastIndexOf('-'))
          : cls;
      final familyMatch = pred.family == selFamily;
      final lowConf = pred.confidence < 0.4;
      if (familyMatch && !lowConf) {
        _showToast(
          '⚠ picked ${cls.substring(cls.lastIndexOf("-") + 1)}, '
          'model says ${pred.gender} (${pred.confidence.toStringAsFixed(2)})',
          error: true,
        );
      } else {
        _showToast(
          '⚠ model says ${pred.className} '
          '(${pred.confidence.toStringAsFixed(2)})',
          error: true,
        );
      }
    } catch (_) {
      // Model not loaded / inference failed — silently fall back.
    }
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

  Future<void> _showStatsSheet() async {
    HapticFeedback.selectionClick();
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF1B1B1B),
      isScrollControlled: true,
      builder: (ctx) => _StatsSheet(
        settings: widget.settings,
        sessionTrain: Map.unmodifiable(_sessionCounts),
        sessionHoldout: Map.unmodifiable(_sessionHoldout),
      ),
    );
  }

  Widget _buildTopBar() {
    return Row(
      children: [
        GestureDetector(
          onTap: _showStatsSheet,
          child: _CounterPill(
            count: _uploadedCount,
            inFlight: _uploadInFlight,
          ),
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
            const SizedBox(width: 24),
            _SmallAction(
              icon: Icons.undo,
              label: _undoStack.isEmpty ? 'Undo' : 'Undo (${_undoStack.length})',
              onTap: _undoStack.isEmpty ? null : _undoLast,
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
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(disabled ? 0.3 : 0.55),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color: disabled ? Colors.white12 : Colors.white24),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14,
                color: disabled ? Colors.white24 : Colors.white70),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: disabled ? Colors.white38 : Colors.white,
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

class _StatsSheet extends StatefulWidget {
  const _StatsSheet({
    required this.settings,
    required this.sessionTrain,
    required this.sessionHoldout,
  });
  final Settings settings;
  final Map<String, int> sessionTrain;
  final Map<String, int> sessionHoldout;

  @override
  State<_StatsSheet> createState() => _StatsSheetState();
}

class _StatsSheetState extends State<_StatsSheet> {
  LabelerStats? _stats;
  String? _err;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final s = await ApiClient(widget.settings).fetchLabelerStats();
      if (!mounted) return;
      setState(() {
        _stats = s;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  static const _kClasses = [
    'SMA-M', 'SMA-F',
    '1.85mm-M', '1.85mm-F',
    '2.4mm-M', '2.4mm-F',
    '2.92mm-M', '2.92mm-F',
    '3.5mm-M', '3.5mm-F',
  ];

  @override
  Widget build(BuildContext context) {
    final stats = _stats;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'Per-class progress',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.refresh, color: Colors.white70),
                  onPressed: _loading ? null : _refresh,
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (_loading)
              const Padding(
                padding: EdgeInsets.all(24),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_err != null)
              Text(
                'Failed to load stats: $_err',
                style: const TextStyle(color: Colors.redAccent),
              )
            else
              ..._kClasses.map((cls) {
                final sessTrain = widget.sessionTrain[cls] ?? 0;
                final sessHoldout = widget.sessionHoldout[cls] ?? 0;
                final serverTrain = stats?.train[cls] ?? 0;
                final serverHoldout = stats?.holdout[cls] ?? 0;
                final starved = serverTrain < 5;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      if (starved)
                        const Padding(
                          padding: EdgeInsets.only(right: 6),
                          child: Icon(Icons.circle,
                              color: Colors.redAccent, size: 8),
                        )
                      else
                        const SizedBox(width: 14),
                      Expanded(
                        child: Text(
                          cls,
                          style: TextStyle(
                            color: starved
                                ? Colors.redAccent
                                : Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      _statCell(
                          label: 'session', train: sessTrain, holdout: sessHoldout),
                      const SizedBox(width: 16),
                      _statCell(
                          label: 'server', train: serverTrain, holdout: serverHoldout),
                    ],
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }

  Widget _statCell({
    required String label,
    required int train,
    required int holdout,
  }) {
    return SizedBox(
      width: 90,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            label,
            style: const TextStyle(color: Colors.white38, fontSize: 10),
          ),
          Text(
            holdout > 0 ? '$train  +$holdout' : '$train',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
