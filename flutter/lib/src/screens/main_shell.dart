import 'package:flutter/material.dart';

import '../settings.dart';
import 'contribute_screen.dart';
import 'identify_screen.dart';
import 'settings_screen.dart';

/// Bottom-nav shell. Tabs: Identify (camera-first) · Contribute · Settings.
class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  Settings? _settings;
  int _index = 0;

  @override
  void initState() {
    super.initState();
    Settings.load().then((s) {
      if (!mounted) return;
      setState(() => _settings = s);
    });
  }

  @override
  Widget build(BuildContext context) {
    final settings = _settings;
    if (settings == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    // Pass isActive so each screen's camera (Identify, Contribute) can
    // tear itself down when the user is on another tab — Android allows
    // only one CameraController on the hardware at a time.
    final pages = [
      IdentifyScreen(settings: settings, isActive: _index == 0),
      ContributeScreen(settings: settings, isActive: _index == 1),
      SettingsScreen(settings: settings),
    ];
    return Scaffold(
      extendBody: true,
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.camera_alt_outlined),
            selectedIcon: Icon(Icons.camera_alt),
            label: 'Identify',
          ),
          NavigationDestination(
            icon: Icon(Icons.upload_outlined),
            selectedIcon: Icon(Icons.upload),
            label: 'Contribute',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
