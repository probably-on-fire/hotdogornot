import 'package:flutter/material.dart';

import '../auth.dart';
import '../settings.dart';
import 'about_screen.dart';
import 'contribute_screen.dart';
import 'identify_screen.dart';

/// Bottom-nav shell with three tabs: Identify, Contribute, About.
/// The About screen's Advanced (relay/token) panel is still gated
/// behind a 7-tap dev-mode unlock so end users don't accidentally
/// edit settings.
class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  Settings? _settings;
  AuthService? _auth;
  int _index = 0;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final s = await Settings.load();
    final a = AuthService(s);
    await a.load();
    if (!mounted) return;
    setState(() {
      _settings = s;
      _auth = a;
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
    final auth = _auth;
    if (settings == null || auth == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // Three tabs: Identify (slot 0), Contribute (slot 1), About (slot 2).
    // Each camera-bearing screen gets isActive so it can dispose its
    // CameraController when not the selected tab — Android allows only
    // one CameraController on the hardware at a time.
    //
    // Contribute is wrapped in AnimatedBuilder so sign-in state changes
    // reactively swap the camera UI vs the sign-in card without
    // requiring the whole shell to rebuild.
    final pages = <Widget>[
      IdentifyScreen(settings: settings, isActive: _index == 0),
      AnimatedBuilder(
        animation: auth,
        builder: (context, _) => ContributeScreen(
          settings: settings,
          auth: auth,
          isActive: _index == 1,
        ),
      ),
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
