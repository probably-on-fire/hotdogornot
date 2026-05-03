import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
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
  String? _camInitError;

  _Mode _mode = _Mode.photo;
  bool _busy = false;       // capture / classify in flight
  bool _recording = false;  // video record in progress
  String? _error;
  PredictResponse? _result;
  File? _capturedFile;      // photo or video that produced _result
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
    if (cam == null || !cam.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      cam.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    if (kIsWeb) {
      // Browser camera access through this plugin is fiddly; fall back
      // to image_picker on web (tap shutter → OS file picker).
      setState(() => _camInitFailed = true);
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _camInitFailed = true;
          _camInitError = 'No cameras found on this device.';
        });
        return;
      }
      final rear = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        rear,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _cam = controller;
        _camInitFailed = false;
      });
    } catch (e) {
      setState(() {
        _camInitFailed = true;
        _camInitError = '$e';
      });
    }
  }

  Future<void> _onShutter() async {
    if (_busy) return;
    if (_mode == _Mode.video) {
      await _toggleRecording();
    } else {
      await _capturePhoto();
    }
  }

  Future<void> _capturePhoto() async {
    final cam = _cam;
    File? file;
    if (cam != null && cam.value.isInitialized) {
      try {
        final shot = await cam.takePicture();
        file = File(shot.path);
      } catch (e) {
        setState(() => _error = 'Capture failed: $e');
        return;
      }
    } else {
      // Web / no-camera fallback: use image_picker.
      final pf = await ImagePicker().pickImage(
        source: ImageSource.camera,
        imageQuality: 92,
        maxWidth: 4032,
      );
      if (pf == null) return;
      file = File(pf.path);
    }
    await _classifyPhoto(file);
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

  Future<void> _classifyPhoto(File f) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _capturedFile = f;
      _contributionStatus = null;
    });
    try {
      final r = await ApiClient(widget.settings).predict(f);
      setState(() => _result = r);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _classifyVideo(File f) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _capturedFile = f;
      _contributionStatus = null;
    });
    try {
      final r = await ApiClient(widget.settings).predictVideo(f);
      setState(() => _result = r);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() => _busy = false);
    }
  }

  void _resetToLive() {
    setState(() {
      _result = null;
      _error = null;
      _capturedFile = null;
      _contributionStatus = null;
    });
  }

  Future<void> _contribute(String cls) async {
    final f = _capturedFile;
    if (f == null) return;
    // Video contribution would need a different endpoint; only photos
    // route to /labeler/upload-train today.
    if (_mode == _Mode.video) {
      setState(() => _contributionStatus =
          'Video contribution: open Contribute tab and use the Video uploader.');
      return;
    }
    setState(() {
      _contributing = true;
      _contributionStatus = null;
    });
    try {
      await ApiClient(widget.settings).uploadTrainingPhoto(f, cls);
      setState(() => _contributionStatus = '✓ Added to training as $cls');
    } catch (e) {
      setState(() => _contributionStatus = 'Upload failed: $e');
    } finally {
      setState(() => _contributing = false);
    }
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
    if (_capturedFile != null && _result != null && _mode == _Mode.photo) {
      return Image.file(
        _capturedFile!,
        fit: BoxFit.cover,
        alignment: Alignment.center,
      );
    }
    final cam = _cam;
    if (cam != null && cam.value.isInitialized) {
      return Center(
        child: AspectRatio(
          aspectRatio: cam.value.aspectRatio,
          child: CameraPreview(cam),
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
    if (r.predictions.isEmpty) {
      return _ResultCard(
        child: const Padding(
          padding: EdgeInsets.symmetric(vertical: 12),
          child: Center(
            child: Text(
              'No connector detected.\nTry a clearer mating-face shot.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ),
        ),
      );
    }
    final p = r.predictions.first;
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
