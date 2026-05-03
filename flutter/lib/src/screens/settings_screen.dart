import 'package:flutter/material.dart';

import '../settings.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.settings});
  final Settings settings;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late TextEditingController _relayCtl;
  late TextEditingController _tokenCtl;
  late TextEditingController _userCtl;
  late TextEditingController _passCtl;

  @override
  void initState() {
    super.initState();
    _relayCtl = TextEditingController(text: widget.settings.relayBaseUrl);
    _tokenCtl = TextEditingController(text: widget.settings.deviceToken);
    _userCtl = TextEditingController(text: widget.settings.labelerUser);
    _passCtl = TextEditingController(text: widget.settings.labelerPass);
  }

  @override
  void dispose() {
    _relayCtl.dispose();
    _tokenCtl.dispose();
    _userCtl.dispose();
    _passCtl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    widget.settings
      ..relayBaseUrl = _relayCtl.text.trim()
      ..deviceToken = _tokenCtl.text.trim()
      ..labelerUser = _userCtl.text.trim()
      ..labelerPass = _passCtl.text.trim();
    await widget.settings.save();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Saved')),
      );
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        actions: [
          IconButton(icon: const Icon(Icons.save), onPressed: _save),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _relayCtl,
                decoration: const InputDecoration(
                  labelText: 'Relay base URL',
                  helperText: 'e.g. https://aired.com/rfcai',
                ),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _tokenCtl,
                decoration: const InputDecoration(
                  labelText: 'Device token (X-Device-Token)',
                  helperText: 'For /predict',
                ),
                obscureText: false,
              ),
              const SizedBox(height: 24),
              const Text('Labeler upload (HTTP Basic)',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              TextField(
                controller: _userCtl,
                decoration: const InputDecoration(labelText: 'Username'),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _passCtl,
                decoration: const InputDecoration(labelText: 'Password'),
                obscureText: true,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _save,
                child: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 4),
                  child: Text('Save'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
