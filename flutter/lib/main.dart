// RF Connector ID — Flutter app.
//
// Two screens:
//   1. Identify — take/pick a photo, POST to /rfcai/predict, show
//      "HOT DOG" or "NOT HOT DOG" big, with the connector family below.
//   2. Contribute — upload a training photo or video to grow the
//      labeled dataset behind the same model.
//
// Talks to the existing FastAPI predict + labeler service over HTTPS.
// Auth: a device token (X-Device-Token header) for /predict, HTTP
// Basic for /labeler/* uploads. Both stored in shared_preferences and
// editable via a Settings screen.

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
