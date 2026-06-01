# Connector ID — Screenshot Capture Guide

App Store needs **1320 × 2868** (iPhone 16 Pro Max, 6.9") portrait screenshots. Up to 10; aim for 3–5.

**Important:** the Identify screen uses the live camera, and **the iOS Simulator has no camera** — the preview will be black. So the camera screens must be captured on a **real iPhone**. The About / request-form screen can be captured in the Simulator.

---

## Option A — Real device (best; required for camera screens)

1. Build & run on your iPhone from the `flutter/` folder:
   ```bash
   cd ~/Desktop/hotdogornot-master/flutter
   flutter devices                 # confirm your iPhone is listed
   flutter run --release -d <your-iphone-id>
   ```
2. In the app, navigate to the screen you want, then take a screenshot on the phone:
   - Face ID iPhone: **Side button + Volume Up**, release quickly.
3. Get the images onto your Mac: AirDrop them, or in Photos open each and **File → Export**, or plug in and use **Image Capture / Photos import**.
4. A 16 Pro Max screenshot is already 1320 × 2868 — upload as-is. For other models, resize to 1320 × 2868 (see "Resize" below).

## Option B — Simulator (non-camera screens, e.g. About)

```bash
cd ~/Desktop/hotdogornot-master/flutter
open -a Simulator
xcrun simctl list devices available | grep "Pro Max"      # find the 6.9" device
flutter run -d "iPhone 16 Pro Max"
# navigate to the screen in the simulator, then capture:
mkdir -p ~/Desktop/connectorid_screenshots
xcrun simctl io booted screenshot ~/Desktop/connectorid_screenshots/about.png
```
A booted iPhone 16 Pro Max simulator captures at exactly **1320 × 2868**.

## Resize / verify any image to spec

```bash
# check size
sips -g pixelWidth -g pixelHeight image.png
# force to 1320 x 2868 (only if the source is the same 9:19.5 aspect ratio)
sips -z 2868 1320 image.png --out image_1320x2868.png
```

---

## Shot list (suggested 4)

1. **Identify – aiming**: connector centered in the reticle, live camera. *(device)*
2. **Result panel**: family + gender + confidence pill after a successful ID. *(device)*
3. **Chip-correction strip**: the family/gender chips shown over a result. *(device)*
4. **About / Request a connector**: the request form + privacy section. *(simulator OK)*

Put the final files in `~/Desktop/connectorid_screenshots/`. When they're ready I'll upload them to App Store Connect along with the metadata.
