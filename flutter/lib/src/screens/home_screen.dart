import 'package:flutter/material.dart';

import '../settings.dart';
import 'identify_screen.dart';
import 'contribute_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Settings? _settings;

  @override
  void initState() {
    super.initState();
    Settings.load().then((s) => setState(() => _settings = s));
  }

  @override
  Widget build(BuildContext context) {
    final settings = _settings;
    if (settings == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Connector ID'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: () async {
              await Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => SettingsScreen(settings: settings),
                ),
              );
              setState(() {});
            },
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 40),
              Text(
                'RF Connector\nIdentification',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      height: 1.1,
                    ),
              ),
              const SizedBox(height: 12),
              Text(
                'Take a photo. Get an answer.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                  fontSize: 14,
                ),
              ),
              const Spacer(),
              ElevatedButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => IdentifyScreen(settings: settings),
                  ),
                ),
                icon: const Icon(Icons.camera_alt, size: 22),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 6),
                  child: Text('Identify a connector', style: TextStyle(fontSize: 17)),
                ),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => ContributeScreen(settings: settings),
                  ),
                ),
                icon: const Icon(Icons.upload_file, size: 22),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 6),
                  child: Text('Contribute training data', style: TextStyle(fontSize: 17)),
                ),
              ),
              const Spacer(),
              Text(
                settings.relayBaseUrl.replaceFirst(RegExp(r'^https?://'), ''),
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.4),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
