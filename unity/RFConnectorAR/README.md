# RF Connector AR — Unity App

Cross-platform Unity AR application that identifies RF connectors in real time via the camera.

Spec: `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`
Plan: `docs/superpowers/plans/2026-04-23-unity-scanner-mvp.md`

## Requirements

- Unity 6.0 LTS (6000.0.x)
- iOS Build Support (for deploy to iOS)
- Android Build Support with Android SDK + NDK (for deploy to Android)

## Packages

- `com.unity.xr.arfoundation`
- `com.unity.xr.arkit`
- `com.unity.xr.arcore`
- `com.unity.sentis`
- `com.unity.test-framework`

## Running in the editor

Open `Assets/Scenes/Scanner.unity` and press Play. In the editor, AR Foundation uses a simulated AR mode — a fake camera feed and a scriptable environment prefab — so the pipeline is runnable on desktop without a device.

## Running on device

1. Connect an iPhone Pro (iOS 13+) or a Pixel 8 Pro / Galaxy S24 Ultra or equivalent (Android 7+).
2. **File → Build Settings → Switch Platform** to iOS or Android.
3. **Build And Run**.

## Tests

- **EditMode tests** run headlessly on the command line:
  ```
  Unity -batchmode -projectPath unity/RFConnectorAR -runTests -testPlatform EditMode -testResults test-results.xml -quit
  ```
  They cover the pure-C# perception logic (ReferenceDatabase, ConfidenceFuser, PerceptionPipeline with stubs, ConfirmationLog).
- **PlayMode tests** are reserved for future work that needs a runtime.

## Status

- Plan 2 — scanner MVP with stub perception (in progress)
- Plan 2b — rich UI (specs card, mating warning, confirmation prompt): deferred
- Plan 3 — real ONNX model integration: deferred
