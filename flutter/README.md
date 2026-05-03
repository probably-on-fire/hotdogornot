# Connector ID — Flutter App

Cross-platform (iOS + Android) take-a-photo identifier for RF connectors.
Replaces the Unity AR app (which is sidelined for now). Talks to the
existing `aired.com/rfcai/predict` and `/rfcai/labeler/upload-*` services.

## Two screens

1. **Identify** — Take or pick a photo of an RF connector. App POSTs to
   `/rfcai/predict` and shows:
   - **HOT DOG** (orange, big) for male connectors (pin)
   - **NOT HOT DOG** (teal, big) for female connectors (socket)
   - Connector family below (`SMA` / `3.5mm` / `2.92mm` / `2.4mm`)
   - Confidence percentage and full class name

2. **Contribute** — Add training data:
   - Photo: drop a phone photo straight into a class folder. *Recommended* —
     12 MP gives enough resolution for the central pin/socket cue.
   - Video: upload a video, server runs Hough detection at fps=5 and dumps
     candidate crops into `<family>-M` for cleanup in the labeler.

Plus a **Settings** screen (gear icon, top right) to edit:
- Relay base URL (default: `https://aired.com/rfcai`)
- Device token (X-Device-Token header for `/predict`)
- Labeler HTTP Basic credentials (for upload endpoints)

Defaults are baked in for the project's deployed services so the app
works out of the box for the project owner.

## Running

```
cd E:\anduril\flutter
"C:\flutter\bin\flutter.bat" pub get
"C:\flutter\bin\flutter.bat" run            # picks up an attached device or emulator
```

To build a release APK for Android:

```
"C:\flutter\bin\flutter.bat" build apk --release
# output: build\app\outputs\flutter-apk\app-release.apk
```

For iOS you need Xcode on a Mac:

```
flutter build ipa --release
```

## Permissions

- iOS: camera + photo library + microphone (the last is required by the
  video picker plugin even though the app doesn't record audio).
- Android: INTERNET + CAMERA. Photo / video gallery access is handled
  by the system pickers without an explicit permission entry.

## Code layout

```
lib/
  main.dart                 — entry point
  src/
    app.dart                — MaterialApp + theme wiring
    theme.dart              — dark theme + hot-dog/not-hot-dog colors
    settings.dart           — persisted Settings (shared_preferences)
    api.dart                — POST to /predict, /labeler/upload-train, /upload-video
    screens/
      home_screen.dart      — landing page
      identify_screen.dart  — camera + result display
      contribute_screen.dart — upload photo / video
      settings_screen.dart  — edit relay URL / token / labeler creds
```

## Backend coupling

This app is a thin client over the FastAPI service in
`E:\anduril\training\rfconnectorai\server\`. Specifically:

- **POST** `/rfcai/predict` (multipart `image=<jpeg>`, header
  `X-Device-Token`) — returns `{predictions: [{class_name, confidence,
  probabilities, bbox}, ...], image_width, image_height}`.
- **POST** `/rfcai/labeler/upload-train` (multipart `cls`, `images=[...]`,
  HTTP Basic) — saves to `data/labeled/embedder/<class>/photo_*`.
- **POST** `/rfcai/labeler/upload-video` (multipart `family`, `fps`,
  `sensitivity`, `max_crops`, `file=<video>`, HTTP Basic) — server
  extracts crops via Hough and dumps them to `<family>-M`.

If the predict service moves or the auth changes, edit Settings inside
the app rather than touching code.
