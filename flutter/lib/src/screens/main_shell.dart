import 'package:flutter/material.dart';

import '../settings.dart';
import 'about_screen.dart';
import 'contribute_screen.dart';
import 'identify_screen.dart';

/// Bottom-nav shell. End users see Identify + About. When dev mode is
/// flipped on (7-tap version in About) the Contribute tab also appears
/// and the About screen reveals an Advanced (relay/token/labeler) panel.
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

  void _onDevModeChanged(bool next) {
    final s = _settings;
    if (s == null) return;
    setState(() {
      s.devMode = next;
      // Contribute slot disappears when dev mode flips off — bounce
      // back to Identify so we don't end up pointing at a now-removed
      // tab index.
      _index = 0;
    });
    s.save();
  }

  @override
  Widget build(BuildContext context) {
    final settings = _settings;
    if (settings == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // Dev mode adds the Contribute tab between Identify and About.
    // Identify is always slot 0, About is the last slot, Contribute (if
    // present) is slot 1. Each camera-bearing screen gets isActive so
    // it can dispose its CameraController when not the selected tab —
    // Android allows only one CameraController on the hardware at a time.
    final pages = <Widget>[
      IdentifyScreen(settings: settings, isActive: _index == 0),
      if (settings.devMode)
        ContributeScreen(settings: settings, isActive: _index == 1),
      AboutScreen(
        settings: settings,
        onDevModeChanged: _onDevModeChanged,
      ),
    ];

    final destinations = <NavigationDestination>[
      const NavigationDestination(
        icon: Icon(Icons.camera_alt_outlined),
        selectedIcon: Icon(Icons.camera_alt),
        label: 'Identify',
      ),
      if (settings.devMode)
        const NavigationDestination(
          icon: Icon(Icons.upload_outlined),
          selectedIcon: Icon(Icons.upload),
          label: 'Contribute',
        ),
      const NavigationDestination(
        icon: Icon(Icons.info_outline),
        selectedIcon: Icon(Icons.info),
        label: 'About',
      ),
    ];

    return Scaffold(
      extendBody: true,
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: destinations,
      ),
    );
  }
}
