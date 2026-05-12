# Connector ID — Flutter App

Cross-platform (iOS + Android) take-a-photo identifier for RF coaxial
connectors (SMA, 3.5mm, 2.92mm, 2.4mm, 1.85mm — male and female).
Talks to the FastAPI predict + labeler service at `aired.com/rfcai/*`.
Replaces the sidelined Unity AR app under `unity/`.

The app is being extended under the root roadmap in
`../IMPLEMENTATION_PLAN.md` and `../TASKS.md`. Current camera flow and
current `/predict` parsing must remain compatible while the backend grows
from a flat ResNet class label into structured connector attributes:

- family/type,
- standard vs reverse polarity,
- gender/contact configuration,
- mount style,
- orientation,
- confidence state and warnings,
- top alternatives,
- spec summary from `training/rfconnectorai/specs/connectors.yaml`.

The detailed system diagram source is `../docs/SOFTWARE_ARCHITECTURE.dot`.
The model transition plan is `../docs/MULTI_ARCHITECTURE_TRANSITION.md`;
the Flutter contract is still to preserve existing `/predict` parsing while
new structured fields are added beside the old ones.

## Tabs the user sees

End users see a deliberately small two-tab shell. Most config lives
behind a hidden dev-mode gesture so customers don't have to wrangle
URLs and tokens.

### 1. Identify (default tab)

Camera fills the screen. Tap shutter → photo POSTs to
`/rfcai/predict` → result panel slides up over the frozen frame:

- Big colored label (the demo joke — orange "HOT DOG" for male, teal
  "NOT HOT DOG" for female; will be relabeled before public release)
- Connector family below (`SMA` / `3.5mm` / `2.92mm` / `2.4mm`)
- Confidence + full class name in a small pill
- **Chip-correction strip**: family chips
  (`SMA · 3.5mm · 2.92mm · 2.4mm · 1.85mm`) and gender chips (`M · F`)
  pre-selected from the prediction. Tap a different chip to flip just
  that axis. Single button below switches between green "Confirm X-Y"
  (matches prediction) and amber "Save as X-Y" (you've changed
  something). Save → photo lands in the training folder, brief
  success toast, auto-return to live preview.

There's also a Photo / Video toggle pill at the top — Video mode
records a short clip and uploads to `/rfcai/predict-video`, which
samples ~30 frames and returns the highest-confidence prediction.

### 2. About (default tab)

Compact hero (small aired.com brain mark + "Connector ID" + version),
one-line description, then the **main feature: a "Request a new
connector type" form** — name + optional notes + Send button. Submission
opens the user's email app with subject `Connector request: <name>`
and a body containing their notes plus app version + platform; the
recipient is `chris@aired.com`. The app doesn't intercept or relay.

Below the form: a collapsible **Privacy** section (what happens to
photos taken in Identify, what happens to connector requests, what's
stored locally, removal contact), then a small "Powered by aired.com"
footer that opens the website.

The app version line on the hero is the **dev-mode unlock**: tap it
seven times in quick succession (Android-style "developer options"
gesture) to flip dev mode on. Snackbars hint at the remaining tap
count after tap 4. Dev-mode state persists in SharedPreferences.

### 3. Contribute (dev mode only — admin tab)

Reappears between Identify and About when dev mode is on. Camera-first
capture screen for owners building the training set:

- Live camera fills the screen.
- Top-left counter pill ("12 uploaded" + spinner during in-flight
  uploads).
- Top-right `training` / `HOLDOUT` toggle (amber when on; HOLDOUT
  routes the next captures to `data/test_holdout/<class>/` instead
  of `data/labeled/embedder/<class>/`).
- Bottom: family chips, gender chips, "next: 2.4mm-M" preview pill,
  shutter, small Gallery / Video buttons.
- Tap shutter → camera takes photo → fire-and-forget upload while
  the camera stays live for the next shot. Class chips persist; one
  tap per shot.
- Holdout intent is captured at shutter time, so flipping the toggle
  for the next shot doesn't retroactively re-route pending uploads.

When dev mode is on, the About screen also reveals an **Advanced**
card with the relay URL, device token, and labeler creds — the
content of the old Settings screen, inlined.

## Camera lifecycle

Both Identify and Contribute hold a `CameraController`, but Android
allows only one on the hardware at a time. `main_shell.dart` passes
each screen an `isActive` prop based on the selected tab; the
inactive tab disposes its controller in `didUpdateWidget` so the
active tab can claim it.

## On-device inference (Tier 1 spike)

The app bundles the v18 ResNet-18 ONNX in `assets/models/` and can
run the classifier entirely on-device via the `onnxruntime` package.
Toggle in **About → Advanced → "On-device inference"** (dev-mode-gated).
When on, the Identify screen's predict path bypasses `/predict` and
runs locally — no network round-trip, works offline, ~50–100 ms per
frame. Result is wrapped in the same `PredictResponse` shape so the
rest of the UI (chip correction, spec card) is unchanged.

The on-device path intentionally skips rembg + Hough + TTA in this
tier, so accuracy may differ from the server path. See
`../training/docs/yolo_hybrid_evaluation_2026-05-11.md` for the
field-test plan and Tier-2/Tier-3 follow-ups.

## Code layout

```
lib/
  main.dart                       — entry point, portrait lock
  src/
    app.dart                      — MaterialApp + theme wiring
    theme.dart                    — dark theme + HOT DOG / NOT HOT DOG colors
    settings.dart                 — persisted Settings (relay, token,
                                    labeler creds, devMode, onDeviceMode)
    api.dart                      — multipart POST to /predict and
                                    /labeler/upload-{train,test,video}.
                                    Parses optional structured fields
                                    (family, gender, *_confidence, spec).
    ondevice/
      classifier.dart             — singleton ResNet-18 ONNX wrapper.
                                    Loads weights.synth_20ep.onnx from
                                    the asset bundle on first use,
                                    reused across requests.
    screens/
      main_shell.dart             — bottom-nav shell, conditional tabs
      identify_screen.dart        — camera + predict + chip-correction.
                                    Routes through ondevice classifier
                                    when onDeviceMode is on.
      contribute_screen.dart      — camera-first capture (dev mode)
      about_screen.dart           — hero + request form + privacy +
                                    Advanced (dev mode, has on-device
                                    toggle)
tool/
  generate_icon.py                — PIL script that crops the aired.com
                                    brain mark out of the full logo +
                                    emits icon.png and icon_foreground.png
                                    consumed by flutter_launcher_icons
assets/
  icon/
    icon.png                      — full app icon (white bg + brain)
    icon_foreground.png           — Android adaptive foreground
    source/aired_logo_full.png    — committed source; re-crop with
                                    tool/generate_icon.py
  models/
    connector_classifier.onnx     — bundled v18 weights for on-device
                                    inference (~44 MB)
    labels.json                   — class_names + ImageNet preproc
```

## Backend coupling

This app is a thin client over the FastAPI service in
`E:\anduril\training\rfconnectorai\server\`:

- **POST** `/rfcai/predict` (multipart `image=<jpeg>`, header
  `X-Device-Token`) — returns `{predictions: [{class_name, confidence,
  probabilities, bbox}, ...], image_width, image_height}`. Server
  pre-filters with rembg, re-composites the silhouette on white,
  Hough-detects crops, runs ResNet-18 with 5× test-time augmentation.
- **POST** `/rfcai/predict-video` (multipart `video=<mp4|mov|...>`,
  header `X-Device-Token`) — server samples the clip at 1 fps (capped
  at 30 frames), runs detect+classify on every frame, returns the
  single highest-confidence prediction.
- **POST** `/rfcai/labeler/upload-train` (multipart `cls`,
  `images=[...]`, HTTP Basic) — saves to
  `data/labeled/embedder/<class>/`. Used by Contribute when HOLDOUT
  is off.
- **POST** `/rfcai/labeler/upload-test` (multipart `cls`,
  `images=[...]`, HTTP Basic) — saves to `data/test_holdout/<class>/`.
  Used by Contribute when HOLDOUT is on.
- **POST** `/rfcai/labeler/upload-video` (multipart `family`, `fps`,
  `sensitivity`, `max_crops`, `file=<video>`, HTTP Basic) — server
  extracts crops via Hough and dumps them to `<family>-M` for cleanup.
- **GET** `/rfcai/healthz` (no auth) — server health snapshot
  including which classifier is loaded and whether the rembg fg
  filter is available.

Roadmap note: future `/predict` responses should add richer structured
fields beside the existing `predictions` list, not instead of it. That
lets this current Flutter parser keep working while new UI elements are
added for connector attributes, warnings, and spec lookup.

If the predict service moves or auth rotates, edit the Advanced
section (dev mode → About) inside the app rather than touching code.

See `training/docs/architecture.md` for the end-to-end inference and
training flow. See `training/docs/runbook.md` for deploy/retrain
operational details.

## Running

```bash
cd E:\anduril\flutter
"C:\flutter\bin\flutter.bat" pub get
"C:\flutter\bin\flutter.bat" run         # picks up an attached device or emulator
```

To build a release APK for Android:

```bash
"C:\flutter\bin\flutter.bat" build apk --release
# output: build\app\outputs\flutter-apk\app-release.apk
```

For iOS you need Xcode on a Mac:

```bash
flutter build ipa --release
```

## Permissions

- iOS: camera + photo library + microphone (last is required by the
  video picker plugin even though the app doesn't record audio).
- Android: `INTERNET` + `CAMERA`. Photo / video gallery access is
  handled by the system pickers without an explicit permission entry.

## Regenerating the app icon

After replacing `assets/icon/source/aired_logo_full.png`:

```bash
python tool/generate_icon.py
"C:\flutter\bin\dart.bat" run flutter_launcher_icons
```

That regenerates `icon.png` + `icon_foreground.png` from the source,
then writes the iOS `AppIcon.appiconset` + Android `mipmap-*` /
`drawable-*` files via `flutter_launcher_icons`. Commit all of them
together.
