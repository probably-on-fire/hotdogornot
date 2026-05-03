import 'package:flutter/material.dart';

import 'screens/main_shell.dart';
import 'theme.dart';

class ConnectorIdApp extends StatelessWidget {
  const ConnectorIdApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Connector ID',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const MainShell(),
    );
  }
}
