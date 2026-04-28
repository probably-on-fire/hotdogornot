# Build the AR app

## What's in here

- Unity 6000.0.73f1 LTS project
- AR Foundation 6.0.3 (ARKit + ARCore)
- Sentis 2.1.2 (on-device ONNX inference)
- Pre-bundled classifier ONNX in `Assets/StreamingAssets/model/`

The app's core flow:

1. `CameraFrameSource` pulls AR camera frames
2. `SentisClassifier` runs the bundled ResNet-18 on each frame (~5 Hz)
3. `ClassifierLoop` feeds results into `InlineCorrectionPanel`
4. When confidence is low (< 0.6), a banner says "Not sure — tap to help train"
5. User taps → class picker → `RelayClient.UploadFrame` POSTs to `https://aired.com/rfcai/uploads`
6. `ModelUpdater` polls the relay every 10 minutes for newer model versions and OTA-swaps

---

## One-time setup (~10 min)

1. **Open the project** in Unity Hub at `unity/RFConnectorAR`.
   - First open re-imports everything; takes 5-10 minutes.
2. **Install Android build support** (Window → Package Manager → Unity Registry → Android Build Support module). Skip if it's already there.
3. **Build the scenes programmatically:**
   - Menu: `RFConnectorAR → Build All Scenes` (constructs Scanner / Enroll / Curate)
   - Menu: `RFConnectorAR → Augment Scanner with ML Pipeline` (adds the classifier + relay client + correction UI to Scanner)
4. **Set the device token** (REQUIRED, otherwise uploads will 401):
   - Open `Assets/Scenes/Scanner.unity`
   - In the hierarchy, select the `MLPipeline` GameObject
   - In the Inspector, set `Device Token` on both `Model Updater` and (descendant) `Inline Correction Panel`
   - Use the same token that's in `/etc/default/rfcai-relay` on aired.com
5. **Save the scene** (Ctrl+S).

---

## Build the APK

### From the Editor

`File → Build Settings → Android → Build` → choose an output dir.

### From the command line (no GUI, headless)

```bash
"C:\Program Files\Unity\Hub\Editor\6000.0.73f1\Editor\Unity.exe" \
    -batchmode -nographics -projectPath "E:\anduril\unity\RFConnectorAR" \
    -buildTarget Android \
    -executeMethod RFConnectorAR.EditorTools.BuildScript.BuildAndroid \
    -logFile build-android.log \
    -quit
```

Output: `unity/RFConnectorAR/Builds/Android/RFConnectorAR.apk`

---

## Install on a phone

### Android (USB debugging enabled)

```bash
adb install -r unity/RFConnectorAR/Builds/Android/RFConnectorAR.apk
```

If `adb` isn't on PATH, it's at `%USERPROFILE%\AppData\Local\Android\Sdk\platform-tools\adb.exe`.

### iOS

`BuildScript.BuildIOS` produces an Xcode project at `unity/RFConnectorAR/Builds/iOS/`. Open `Unity-iPhone.xcodeproj` in Xcode on a Mac, set the team / signing identity, and Run on a connected device or build for TestFlight. (No way to build iOS bundles from Windows.)

---

## Smoke-test on the phone

1. Launch the app
2. Camera permission prompt → allow
3. Point at any object — you should see the correction banner appear at the top within ~2 seconds (because nothing matches an RF connector at high confidence)
4. Tap the banner → class picker appears
5. Pick any class → status text changes to "sent ([class]) — thanks"
6. Verify the upload landed on aired.com:
   ```bash
   ssh chris@aired.com 'sudo ls -la /srv/rfcai/incoming/'
   ```
   You should see a new `<timestamp>_<id>/` directory with the frame inside.
7. Within ~2 minutes the training-box ingestion daemon will process it and write a `_processed.json` sidecar.

---

## Update the bundled model

When the relay has a new version, the OTA path takes care of it on next launch. But the *bundled* model in `Assets/StreamingAssets/model/connector_classifier.onnx` is what ships in the APK and is what runs on first launch before OTA fires. Refresh it whenever you cut a release:

```bash
# from the training repo, after a successful retrain:
cp training/models/connector_classifier/weights.onnx \
   unity/RFConnectorAR/Assets/StreamingAssets/model/connector_classifier.onnx
cp training/models/connector_classifier/labels.json \
   unity/RFConnectorAR/Assets/StreamingAssets/model/labels.json
```

Then rebuild the APK.

---

## Troubleshooting

**"401 invalid device token" in adb logcat**
The device token isn't set or doesn't match the relay. Check the Inspector values on `MLPipeline → ModelUpdater` and `InlineCorrectionPanel`.

**Banner never appears**
Either the classifier isn't loaded (check logcat for `[ModelUpdater] loaded classifier`) or the classifier confidence is staying above 0.6 on whatever you're pointing at. Lower `confidenceThreshold` on the InlineCorrectionPanel for testing.

**App crashes on launch with "Sentis: invalid model"**
The bundled ONNX is stale or wasn't packaged. Verify `Assets/StreamingAssets/model/connector_classifier.onnx` exists and is ≈40 MB. Rebuild.

**Uploads fail with TLS error**
Android 9+ blocks cleartext HTTP by default. We're using HTTPS to aired.com so this shouldn't bite you, but if you point the relay URL at an http:// host you'll need to add a network security config (`Assets/Plugins/Android/AndroidManifest.xml` → `android:usesCleartextTraffic="true"`).
