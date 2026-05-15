# iOS build — RF Connector ID

Quick reference for building/installing on iPhone from a Mac.

## Prerequisites (one-time)

- Xcode (15.x+ recommended)
- CocoaPods: `sudo gem install cocoapods` or `brew install cocoapods`
- An Apple ID — free works for personal device sideload; paid Apple Developer Program ($99/yr) needed for TestFlight / 7+ day install windows
- Your iPhone plugged in, **trusted** (Settings → General → VPN & Device Management → trust this Mac)

## First-time setup on the Mac

```bash
git clone <this-repo>
cd flutter
flutter pub get
cd ios
pod install        # creates Podfile.lock and the .xcworkspace
cd ..
```

Then open Xcode:

```bash
open ios/Runner.xcworkspace   # NOT Runner.xcodeproj — use the workspace
```

In Xcode:

1. Click the **Runner** target → **Signing & Capabilities** tab
2. Set **Team** to your Apple ID (Personal Team is fine for sideloading)
3. **Bundle Identifier** is already `com.aired.connectorId`. If Xcode complains
   it's not available with your free Apple ID, change to something unique like
   `com.YOURNAME.connectorId.dev`.
4. Plug in iPhone, select it as the run destination
5. Click ▶ (Run) — Xcode builds + installs + launches

On first launch on the device, iOS will block the unsigned app. Go to
**Settings → General → VPN & Device Management → Developer App** and trust
your developer certificate, then re-open the app.

## Build a release IPA (TestFlight or ad-hoc distribution)

```bash
flutter build ipa --release
open build/ios/archive/Runner.xcarchive
```

In Xcode Organizer: **Distribute App** → **TestFlight & App Store** (or
**Ad Hoc** for direct device installs).

## Key requirements already in place

- `Info.plist` has `NSCameraUsageDescription`, `NSPhotoLibraryUsageDescription`,
  `NSPhotoLibraryAddUsageDescription`, `NSMicrophoneUsageDescription`
- Deployment target: **iOS 13.0** (required by `flutter_secure_storage 9.x`
  and `onnxruntime 1.x`)
- Plugins auto-registered in `AppDelegate.swift` via `GeneratedPluginRegistrant`
- `flutter_secure_storage` uses iOS Keychain — no additional capability needed
  for single-app keychain access (which is what we use)

## Likely first-build gotchas

1. **`pod install` fails on Apple Silicon** — try `arch -x86_64 pod install`
   or update CocoaPods (`sudo gem install cocoapods`).
2. **Code signing errors** — make sure your Team is set in Xcode and the
   Bundle Identifier is unique to your Apple ID.
3. **"Untrusted Developer" on first launch** — expected; trust via Settings.
4. **Camera doesn't work in iOS Simulator** — Simulator has no camera. Use
   a real device.

## App behaviour on iOS (parity with Android)

- **Identify tab**: anonymous, runs the live camera and uploads frames to
  `/predict`. Same flow as Android.
- **Contribute tab**: shows a Sign In card on first launch. After signing in
  with `chris / Elad9651!` (or `jdcrunchman` / `zapperman`), the token is
  stored in iOS Keychain via `flutter_secure_storage`.
- **About tab**: red AI Red brain logo, app version, privacy disclosure,
  on-device-inference toggle, "Powered by aired.com" footer.

Token survives app restart. Sign out from the avatar pill in the Contribute
top-bar to clear it.
