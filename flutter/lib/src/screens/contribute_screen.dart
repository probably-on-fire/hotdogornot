import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../settings.dart';

const _kCanonicalClasses = [
  'SMA-M', 'SMA-F',
  '3.5mm-M', '3.5mm-F',
  '2.92mm-M', '2.92mm-F',
  '2.4mm-M', '2.4mm-F',
];
const _kFamilies = ['SMA', '3.5mm', '2.92mm', '2.4mm'];

class ContributeScreen extends StatefulWidget {
  const ContributeScreen({super.key, required this.settings});
  final Settings settings;

  @override
  State<ContributeScreen> createState() => _ContributeScreenState();
}

class _ContributeScreenState extends State<ContributeScreen> {
  String _photoClass = '2.4mm-M';
  String _videoFamily = '2.4mm';
  bool _busy = false;
  String? _status;

  Future<void> _uploadPhoto() async {
    final picker = ImagePicker();
    final pf = await picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 92,
      maxWidth: 4032,
    );
    if (pf == null) return;
    await _doUploadPhoto(File(pf.path));
  }

  Future<void> _pickPhoto() async {
    final picker = ImagePicker();
    final pf = await picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 92,
      maxWidth: 4032,
    );
    if (pf == null) return;
    await _doUploadPhoto(File(pf.path));
  }

  Future<void> _doUploadPhoto(File f) async {
    setState(() {
      _busy = true;
      _status = null;
    });
    try {
      final api = ApiClient(widget.settings);
      await api.uploadTrainingPhoto(f, _photoClass);
      setState(() => _status = '✓ Uploaded as $_photoClass');
    } catch (e) {
      setState(() => _status = 'Upload failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _pickVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );
    if (result == null || result.files.single.path == null) return;
    final f = File(result.files.single.path!);
    setState(() {
      _busy = true;
      _status = 'Uploading video — server will extract crops…';
    });
    try {
      final api = ApiClient(widget.settings);
      final body = await api.uploadTrainingVideo(f, _videoFamily);
      setState(() => _status = '✓ ${body.replaceAll(RegExp(r'<[^>]+>'), '').trim()}');
    } catch (e) {
      setState(() => _status = 'Upload failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Contribute training data')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _PhotoSection(
                selectedClass: _photoClass,
                onClassChanged: (c) => setState(() => _photoClass = c),
                onTakePhoto: _busy ? null : _uploadPhoto,
                onPickPhoto: _busy ? null : _pickPhoto,
              ),
              const SizedBox(height: 16),
              _VideoSection(
                selectedFamily: _videoFamily,
                onFamilyChanged: (f) => setState(() => _videoFamily = f),
                onPickVideo: _busy ? null : _pickVideo,
              ),
              const SizedBox(height: 16),
              if (_busy) const _Busy(),
              if (_status != null) _StatusCard(status: _status!),
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFF2A313E)),
                ),
                child: Text(
                  'Photos give the best resolution for the central pin/socket cue. '
                  'Videos let the server auto-extract many candidate crops via '
                  'Hough detection — you then clean them up in the labeler.',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                    fontSize: 12,
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PhotoSection extends StatelessWidget {
  const _PhotoSection({
    required this.selectedClass,
    required this.onClassChanged,
    required this.onTakePhoto,
    required this.onPickPhoto,
  });
  final String selectedClass;
  final ValueChanged<String> onClassChanged;
  final VoidCallback? onTakePhoto;
  final VoidCallback? onPickPhoto;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Photo (recommended)',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: selectedClass,
              decoration: const InputDecoration(labelText: 'Class'),
              items: _kCanonicalClasses
                  .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                  .toList(),
              onChanged: (v) { if (v != null) onClassChanged(v); },
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: onTakePhoto,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Take'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: onPickPhoto,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Pick'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _VideoSection extends StatelessWidget {
  const _VideoSection({
    required this.selectedFamily,
    required this.onFamilyChanged,
    required this.onPickVideo,
  });
  final String selectedFamily;
  final ValueChanged<String> onFamilyChanged;
  final VoidCallback? onPickVideo;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Video — auto-extract crops',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: selectedFamily,
              decoration: const InputDecoration(labelText: 'Family'),
              items: _kFamilies
                  .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                  .toList(),
              onChanged: (v) { if (v != null) onFamilyChanged(v); },
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: onPickVideo,
              icon: const Icon(Icons.videocam),
              label: const Text('Pick video'),
            ),
            const SizedBox(height: 8),
            Text(
              'Server runs Hough detection at fps=5 and dumps every detected '
              'crop into <family>-M for you to flip M↔F in the labeler.',
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.5),
                fontSize: 11,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Busy extends StatelessWidget {
  const _Busy();
  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 16),
      child: Center(child: CircularProgressIndicator()),
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.status});
  final String status;
  @override
  Widget build(BuildContext context) {
    final ok = status.startsWith('✓');
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(ok ? Icons.check_circle : Icons.error_outline,
                color: ok ? const Color(0xFF4ADE80) : Colors.redAccent),
            const SizedBox(width: 10),
            Expanded(
              child: Text(status,
                  style: const TextStyle(fontSize: 13, height: 1.3)),
            ),
          ],
        ),
      ),
    );
  }
}
