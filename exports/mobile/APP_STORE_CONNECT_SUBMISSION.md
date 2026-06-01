# Connector ID — App Store Connect Submission Pack

App: **Connector ID** · Bundle ID `com.aired.connectorId` · Version **1.0.0** (build 1)
Publisher: aired.com · Contact: chris@aired.com · Prepared 2026-05-28

Character limits are shown in [brackets]. Copy is written to fit. Everything below is paste-ready.

---

## 1. App Information (set once, language: English U.S.)

**Name** [30]: `Connector ID`

**Subtitle** [30]: `Identify RF coax connectors`

**Category** — Primary: `Utilities` · Secondary: `Productivity`

**Content Rights**: Check "I confirm... does not contain, show, or access third-party content" (the app uses your own model + your own backend). If unsure, leave unchecked and explain in review notes.

**Age Rating** — answer every questionnaire item **None / No**. Result: **4+**.

---

## 2. Pricing & Availability

- Price: **Free** (Tier 0)
- Availability: **All territories** (or restrict as you prefer)

---

## 3. Version Information (1.0.0, English U.S.)

**Promotional Text** [170] — editable anytime without review:
```
Point your camera at any RF coaxial connector and get an instant ID — type and gender, with the spec that matters. Built for RF engineers, technicians, and lab benches.
```

**Description** [4000]:
```
Connector ID identifies RF coaxial connectors from a single photo. Point your phone at a connector, tap the shutter, and get the connector family and gender back in about a second — no calipers, no guessing, no digging through datasheets.

Built for RF and microwave engineers, lab technicians, field installers, and anyone who has ever stared at two nearly identical connectors and wondered which is which.

IDENTIFIES TEN CLASSES
SMA, 3.5 mm, 2.92 mm / K, 2.4 mm, and 1.85 mm — each in male and female. These connectors are visually subtle and physically tiny; several differ by barely more than a millimeter in diameter. Connector ID is trained specifically to tell them apart.

HOW IT WORKS
1. Center the connector in the on-screen reticle.
2. Tap the shutter.
3. Read the result: connector family, gender, and confidence — plus the key spec (max frequency, impedance, coupling).

If the model isn't confident, it tells you to try again instead of guessing. You get a correct answer or no answer — never a confident wrong one.

ACCURATE WHERE IT COUNTS
Trained and validated on real-world phone photos taken at normal holding distance, not just pristine bench close-ups — so it performs the way you actually use it.

REQUEST NEW CONNECTOR TYPES
Need a family we don't cover yet? Send a request right from the app and help shape what comes next.

Powered by aired.com.
```

**Keywords** [100, comma-separated, no spaces]:
```
RF,connector,SMA,coaxial,coax,identify,3.5mm,2.92mm,2.4mm,microwave,antenna,adapter,Kconnector,1.85mm
```

**Support URL** (required): `https://aired.com`  ← confirm this page exists / add a support or contact page
**Marketing URL** (optional): `https://aired.com`
**Version**: `1.0.0`
**Copyright**: `2026 aired.com`  ← replace with the legal entity name if different (e.g. "2026 Aired, Inc.")

**What's New**: leave blank — this is the first version.

---

## 4. App Privacy (data collection questionnaire)

Based on the app's behavior (uploads photos to aired.com for identification; connector requests open the user's email app):

- **Do you collect data?** Yes.
- **Photos** → collected. Linked to user? **No** (no account for the core Identify flow). Used for tracking? **No**. Purpose: **App Functionality** (identification). State whether photos are stored after processing — confirm with your backend; the in-app Privacy section should match this answer exactly.
- **Contact Info (email)** — only if you consider the "Request a connector" email a data collection. The app does not send it; it opens the user's own mail client. Generally **not** declared, but note it in the privacy policy.
- **Identifiers / Device ID**: the app sends an `X-Device-Token` header. If that token is a per-install identifier, declare **Device ID → App Functionality, not linked, no tracking**.

You must also provide a **Privacy Policy URL** (required for all apps). The app has an in-app privacy section but App Store requires a hosted URL — e.g. `https://aired.com/privacy`. **This page must exist before submission.** I can draft the policy text for you to host.

---

## 5. App Review Information

- **Sign-in required?** No — the default Identify and About tabs need no login.
- **Contact**: First/Last name, phone, email (chris@aired.com).
- **Notes for the reviewer** (paste):
```
Connector ID identifies RF coaxial connectors from a photo.

To test: open the app, allow camera access, point the camera at any small cylindrical object (or use the photo-library button to pick an existing image), and tap the shutter. A result panel showing connector family, gender, and confidence will appear.

Note: the live camera preview is unavailable in the iOS Simulator. Please test on a physical device, or use the photo-library import option.

Identification runs against our server at aired.com. No account or sign-in is required for normal use. The "Request a connector type" form on the About screen simply opens the device mail composer.
```
- **Attachment**: optional. A short screen-recording of a successful identification helps reviewers.

---

## 6. Export Compliance

The app uses only standard HTTPS / Keychain encryption. Add this to `flutter/ios/Runner/Info.plist` so you aren't asked on every upload:
```xml
<key>ITSAppUsesNonExemptEncryption</key>
<false/>
```
Then in ASC answer: uses encryption = **Yes**, but **exempt** (standard encryption only).

---

## 7. Screenshots — required sizes

App Store Connect requires at least the largest iPhone size; it will scale it to smaller devices.

- **6.9" iPhone** (iPhone 16 Pro Max): **1320 × 2868 px** portrait — the recommended target.
- (6.7" 1290 × 2796 is also accepted as the baseline.)
- iPad screenshots are required **only if** the app is offered on iPad. Connector ID is camera-first; recommend setting the target to **iPhone only** in Xcode (TARGETS → Runner → General → Supported Destinations) to avoid needing iPad screenshots.

Minimum 1 screenshot per size; up to 10. Aim for 3–5. See the capture guide in this folder.

Suggested shots: (1) Identify screen with a connector centered in the reticle, (2) result panel with family + gender + confidence, (3) the chip-correction strip, (4) About screen / "Request a connector" form.
```
