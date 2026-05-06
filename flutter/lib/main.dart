// Connector ID — Flutter app. Camera-first identification of RF
// coaxial connectors; talks to the FastAPI predict + labeler service
// at aired.com over HTTPS.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'src/app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Camera-first UX is a portrait experience; locking avoids preview
  // re-rotation glitches and keeps the layout predictable.
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(const ConnectorIdApp());
}
