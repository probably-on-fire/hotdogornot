import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../settings.dart';
import '../theme.dart';

class IdentifyScreen extends StatefulWidget {
  const IdentifyScreen({super.key, required this.settings});
  final Settings settings;

  @override
  State<IdentifyScreen> createState() => _IdentifyScreenState();
}

const _kCanonicalClasses = [
  'SMA-M', 'SMA-F',
  '3.5mm-M', '3.5mm-F',
  '2.92mm-M', '2.92mm-F',
  '2.4mm-M', '2.4mm-F',
];

class _IdentifyScreenState extends State<IdentifyScreen> {
  File? _image;
  bool _loading = false;
  bool _contributing = false;
  String? _error;
  PredictResponse? _response;
  String? _contributionStatus;

  Future<void> _pickFromCamera() => _pickImage(ImageSource.camera);
  Future<void> _pickFromGallery() => _pickImage(ImageSource.gallery);

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    try {
      final pf = await picker.pickImage(
        source: source,
        imageQuality: 92,
        maxWidth: 4032,
      );
      if (pf == null) return;
      setState(() {
        _image = File(pf.path);
        _response = null;
        _error = null;
        _contributionStatus = null;
      });
      await _classify();
    } catch (e) {
      setState(() => _error = 'Pick failed: $e');
    }
  }

  Future<void> _classify() async {
    final image = _image;
    if (image == null) return;
    setState(() {
      _loading = true;
      _error = null;
      _response = null;
      _contributionStatus = null;
    });
    try {
      final api = ApiClient(widget.settings);
      final r = await api.predict(image);
      setState(() => _response = r);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _contribute(String cls) async {
    final image = _image;
    if (image == null) return;
    setState(() {
      _contributing = true;
      _contributionStatus = null;
    });
    try {
      final api = ApiClient(widget.settings);
      await api.uploadTrainingPhoto(image, cls);
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
                        fontWeight: c == suggested ? FontWeight.bold : FontWeight.normal,
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
      appBar: AppBar(title: const Text('Identify')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: _buildBody(),
              ),
            ),
            _buildBottomBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildBody() {
    if (_image == null) {
      return Container(
        height: 360,
        margin: const EdgeInsets.symmetric(vertical: 40),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          border: Border.all(color: const Color(0xFF2A313E)),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.add_a_photo, size: 56,
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.4)),
            const SizedBox(height: 16),
            Text('Tap below to take a photo',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                  fontSize: 15,
                )),
          ],
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.file(_image!, fit: BoxFit.cover),
        ),
        const SizedBox(height: 16),
        if (_loading) const _LoadingCard(),
        if (_error != null) _ErrorCard(message: _error!),
        if (_response != null) _ResultCard(response: _response!),
        if (_response != null && _response!.predictions.isNotEmpty)
          _ContributeCard(
            suggestedClass: _response!.predictions.first.className,
            busy: _contributing,
            status: _contributionStatus,
            onConfirm: () => _contribute(_response!.predictions.first.className),
            onCorrect: () => _pickCorrectClass(_response!.predictions.first.className),
          ),
      ],
    );
  }

  Widget _buildBottomBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: const Border(top: BorderSide(color: Color(0xFF2A313E))),
      ),
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: _loading ? null : _pickFromCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Camera'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: _loading ? null : _pickFromGallery,
              icon: const Icon(Icons.photo_library),
              label: const Text('Gallery'),
            ),
          ),
        ],
      ),
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard();
  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(24),
        child: Row(
          children: [
            SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2.5)),
            SizedBox(width: 16),
            Text('Identifying…', style: TextStyle(fontSize: 16)),
          ],
        ),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});
  final String message;
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
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
                      style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text(message,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
                        fontSize: 13,
                      )),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.response});
  final PredictResponse response;
  @override
  Widget build(BuildContext context) {
    if (response.predictions.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Center(
            child: Text('No connector detected.\nTry a clearer mating-face photo.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16)),
          ),
        ),
      );
    }
    final p = response.predictions.first;
    final hotDog = p.isMale;
    final color = hotDog ? kHotDogColor : kNotHotDogColor;
    final label = hotDog ? 'HOT DOG' : 'NOT HOT DOG';
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 24),
        child: Column(
          children: [
            // BIG hot-dog / not-hot-dog verdict.
            Text(
              label,
              style: TextStyle(
                color: color,
                fontSize: 44,
                fontWeight: FontWeight.w900,
                letterSpacing: 1.0,
              ),
            ),
            const SizedBox(height: 12),
            // Family below.
            Text(
              p.family,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurface,
                fontSize: 28,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                '${(p.confidence * 100).toStringAsFixed(0)}% confidence',
                style: TextStyle(
                  color: color,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              p.className,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.5),
                fontSize: 12,
                fontFamily: 'monospace',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ContributeCard extends StatelessWidget {
  const _ContributeCard({
    required this.suggestedClass,
    required this.busy,
    required this.status,
    required this.onConfirm,
    required this.onCorrect,
  });
  final String suggestedClass;
  final bool busy;
  final String? status;
  final VoidCallback onConfirm;
  final VoidCallback onCorrect;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Help train the model',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Add this photo to the training set so the model gets better at recognizing this connector.',
                style: TextStyle(
                  fontSize: 12,
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.5),
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: busy ? null : onConfirm,
                      icon: const Icon(Icons.check, size: 18),
                      label: Text('Confirm $suggestedClass'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF4ADE80),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: busy ? null : onCorrect,
                      icon: const Icon(Icons.edit, size: 18),
                      label: const Text('Correct…'),
                    ),
                  ),
                ],
              ),
              if (busy)
                const Padding(
                  padding: EdgeInsets.only(top: 12),
                  child: Center(child: CircularProgressIndicator()),
                ),
              if (status != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(
                    status!,
                    style: TextStyle(
                      fontSize: 12,
                      color: status!.startsWith('✓')
                          ? const Color(0xFF4ADE80)
                          : Colors.redAccent,
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
