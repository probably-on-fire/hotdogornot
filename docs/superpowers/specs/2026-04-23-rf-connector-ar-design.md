# RF Connector AR Identification App — Design Spec

**Date:** 2026-04-23
**Status:** Draft — pending approval
**Target:** Anduril lab technician tool (pitch → engagement)

## 1. Problem

Anduril lab technicians need to identify RF connectors quickly. The initial scope is eight connector types: male and female variants of SMA, 3.5mm, 2.92mm, and 2.4mm. The three precision sizes (3.5 / 2.92 / 2.4 mm) share the same thread, body, and hex dimensions; the only physical differences are sub-millimeter inner-conductor diameters and subtle dielectric details. Manual identification is error-prone, and mismating incompatible precision connectors damages expensive equipment.

## 2. Goals

- Real-time, live-camera AR identification of RF connectors on iOS and Android.
- Tech holds a single connector up to the camera; app draws a world-anchored ring and label on the part.
- High-confidence identification of connector class and physical size with visible uncertainty when the model is not sure.
- Extensible beyond the initial eight: architecture supports adding new connector types without retraining the model.
- Self-improving: every user interaction either improves the model or grows the reference database.
- Polished enough to demo to Anduril as a pitch for a larger engagement.

## 3. Non-Goals (V1)

- Support for devices older than two years (tier-B devices are V2).
- Damage detection (bent pins, worn threads) — separate product.
- Inventory / shelf-scanning workflows — different UX.
- Multi-language support — English only for V1.
- Offline-only / classified environments — considered but deferred.
- Integration with Anduril's internal parts database — deferred until engagement.

## 4. Constraints

- **Platforms:** iOS and Android, latest flagship tier (iPhone 15 Pro and up; Pixel 8 Pro, Galaxy S24 Ultra, etc.).
- **Device features relied on:** LiDAR (iPhone Pro) or ToF (Android flagship) for depth, ARKit or ARCore for world tracking.
- **Developer:** solo, one codebase preferred.
- **Data:** no physical connectors yet at project start. Proxy data strategy required to build the full pipeline before real parts arrive.
- **Timeline:** optimize for demo-ready artifact within 8–10 weeks.

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNITY APP (iOS + Android)                   │
│                                                                 │
│  ┌───────────────────┐     ┌──────────────────────────────┐     │
│  │  AR Foundation    │────▶│    Perception Pipeline       │     │
│  │  (ARKit / ARCore) │     │                              │     │
│  │                   │     │  1. YOLO detect (connector)  │     │
│  │  • Camera texture │     │  2. Crop + embed (RGBD in)   │     │
│  │  • Depth (LiDAR/  │     │  3. Nearest reference match  │     │
│  │    ToF)           │     │  4. Measure Ø (precision)    │     │
│  │  • World anchors  │     │  5. Confidence fuse          │     │
│  └───────────────────┘     └──────────────┬───────────────┘     │
│                                           ▼                     │
│                            ┌──────────────────────────┐         │
│                            │   AR Overlay Renderer    │         │
│                            │  • World-anchored ring   │         │
│                            │  • Billboard label       │         │
│                            │  • Confidence bar        │         │
│                            │  • Specs card (on tap)   │         │
│                            └──────────────────────────┘         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Sentis Runtime (GPU inference)              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │ (OTA model + reference DB updates)
                               │
┌─────────────────────────────────────────────────────────────────┐
│                  OFFLINE TRAINING PIPELINE (Python)             │
│                                                                 │
│   Data ── Roboflow ── PyTorch + timm ── ONNX ── Signed bundle   │
│    │                                              │             │
│    │                                              ▼             │
│    │                                      App OTA download     │
│    ▼                                                            │
│  Phase 0: proxy data (catalog images + Blender synthetic)       │
│  Phase 1: real photos of connectors                             │
│  Phase 2+: field-collected confirmations feed retraining        │
└─────────────────────────────────────────────────────────────────┘
```

### Key decisions

- **Stack: Unity + AR Foundation + Sentis.** Unity keeps camera frames on GPU end-to-end, gives true world-locked AR overlays, and ships one codebase for both platforms. React Native and Flutter were evaluated and rejected primarily on frame-pipeline overhead and weaker AR tracking.
- **Two-stage perception, not one-stage.** Detection (1-class YOLO) is separate from identification (embedding model). Detection finds the connector fast; identification gets a high-resolution crop and does the hard fine-grained work.
- **Metric-learning embeddings, not fixed N-way classifier.** Model outputs a 128-d vector per connector. Identification is nearest-neighbor lookup against a reference database. Adding new connector types requires no retraining — just new reference vectors.
- **Depth as a 4th input channel to the embedder.** The precision-size question is fundamentally metrological. Feeding depth alongside RGB lets the model reason about physical dimensions natively rather than learn visual shortcuts.
- **Measurement stage as cross-check.** For precision connectors, a classical-CV inner-pin diameter measurement, combined with LiDAR depth, serves as an independent physical cross-check. The model and the measurement must agree before a precision-size verdict is committed.

## 6. Components

### 6.1 Unity project layout

```
RFConnectorAR/
├── Assets/
│   ├── Scenes/ScannerScene.unity
│   ├── Scripts/
│   │   ├── Perception/
│   │   │   ├── CameraFrameSource.cs      # AR Foundation camera + depth
│   │   │   ├── DetectorStage.cs          # YOLO, 1-class
│   │   │   ├── EmbeddingStage.cs         # MobileViT-v2 RGBD → 128-d
│   │   │   └── MeasurementStage.cs       # pixel-Ø + depth → mm
│   │   ├── Reference/
│   │   │   ├── ReferenceDatabase.cs      # loads reference embeddings
│   │   │   └── NearestMatcher.cs         # cosine-similarity lookup
│   │   ├── Fusion/
│   │   │   └── ConfidenceFuser.cs        # ML + measurement → verdict
│   │   ├── Knowledge/
│   │   │   ├── ConnectorSpecs.cs         # loads specs JSON
│   │   │   └── MatingRules.cs            # compatibility matrix + state
│   │   ├── AR/
│   │   │   ├── AnchorManager.cs          # world anchors + debounce
│   │   │   └── OverlayController.cs      # ring + label prefabs
│   │   ├── UI/
│   │   │   ├── ScannerHUD.cs
│   │   │   ├── SpecsCard.cs
│   │   │   ├── MatingWarning.cs
│   │   │   └── ConfirmationPrompt.cs
│   │   ├── Learning/
│   │   │   ├── ConfirmationLog.cs        # encrypted local SQLite
│   │   │   └── TelemetryUploader.cs      # opt-in batched upload
│   │   └── App/
│   │       ├── AppBootstrap.cs
│   │       └── ModelUpdater.cs           # OTA signed-bundle downloader
│   ├── Prefabs/ (ConnectorRing, ConnectorLabel, SpecsCard, etc.)
│   ├── Shaders/ (ConfidenceGlow, BillboardText)
│   └── StreamingAssets/
│       ├── detector.onnx
│       ├── embedder.onnx
│       ├── reference_embeddings.bin
│       ├── connector_specs.json
│       └── mating_matrix.json
└── Packages/
    ├── com.unity.xr.arfoundation
    ├── com.unity.xr.arkit
    ├── com.unity.xr.arcore
    └── com.unity.sentis
```

### 6.2 Python training pipeline

```
training/
├── data/
│   ├── raw/                 # scraped + captured images
│   ├── labeled/             # Roboflow exports
│   └── synthetic/           # Blender-rendered proxy images
├── scripts/
│   ├── scrape_catalogs.py   # grab Mouser/Digi-Key product images
│   ├── render_synthetic.py  # Blender + 3D connector models
│   ├── train_detector.py    # Ultralytics YOLOv11n, 1 class
│   ├── train_embedder.py    # triplet-loss MobileViT-v2, RGBD input
│   ├── build_references.py  # compute + save reference embeddings
│   ├── export_onnx.py       # → Sentis-ready ONNX
│   ├── eval.py              # regression set, golden set, calibration
│   └── sign_bundle.py       # sign + package OTA release
├── configs/
└── requirements.txt
```

### 6.3 Model assets

| Asset | Details | Size (FP16) |
|---|---|---|
| `detector.onnx` | YOLOv11n, 1 class ("connector"), 640×640 | ~6 MB |
| `embedder.onnx` | MobileViT-v2, RGBD input 384×384, 128-d output | ~10 MB |
| `reference_embeddings.bin` | N × 128 float16 + labels | ~1 KB per connector |
| `connector_specs.json` | Specs per connector | ~5 KB total |
| `mating_matrix.json` | Compatibility rules | ~2 KB |

Total install size with assets: ~40–60 MB.

## 7. Data Flow

```
Tech holds connector in front of camera
              │
              ▼
  Frame N arrives (60 Hz from AR Foundation)
    • cameraTexture (GPU)
    • depthTexture  (LiDAR/ToF, GPU)
    • cameraPose    (AR tracking)
              │
              ▼ (every 2nd frame ≈ 30 Hz)
  DetectorStage (YOLOv11n) → bboxes
              │
              ▼ (only if score > 0.6 stable across 3 frames)
  EmbeddingStage: crop to 384×384 + depth patch → 4-channel → 128-d vector
              │
              ▼
  NearestMatcher: cosine similarity against reference DB → top-1 + distance
              │
              ▼ (if precision family)
  MeasurementStage: pixel Ø of inner pin + depth → physical Ø in mm
              │
              ▼
  ConfidenceFuser:
    • distance acceptable?
    • ML class matches measured class (for precision)?
    • HIGH / MEDIUM / LOW / UNKNOWN verdict
              │
              ▼
  AnchorManager: bbox + depth → world point → ARAnchor (debounced)
              │
              ▼
  OverlayController: ring + label + confidence bar at anchor
              │
              ▼
  Side effects:
    • Log signal record (always)
    • Trigger ConfirmationPrompt if 55–75% confidence
    • Trigger MatingWarning if hazardous pair in session
    • Update status bar (model version, etc.)
```

### Timing budget (per frame, flagship device)

| Stage | Time | Frequency |
|---|---|---|
| AR Foundation frame | native | 60 Hz |
| Detector | ~8 ms | 30 Hz |
| Embedder + matcher | ~15 ms | on stable hit |
| Measurement | ~5 ms | precision only |
| Fusion + AR update | <2 ms | per verdict |
| **End-to-end when identifying** | **~28 ms** | **≈35 FPS sustained** |

## 8. Error Handling and Edge Cases

### Failure taxonomy

| Failure | Detection | User sees |
|---|---|---|
| No connector in frame | Detector score < 0.3 × 10 frames | "Aim camera at connector" |
| Partial connector at edge | bbox touches frame border | "Center the connector" |
| Motion blur | Laplacian variance below threshold | "Hold steady" |
| Multiple connectors | Detector returns >1 high-score box | Ring + label on each |
| ML / measurement disagree | Fuser rule mismatch | Dashed amber ring, "Precision (uncertain)" |
| No depth at bbox centre | Depth NaN/0 | ML-only with "(no depth)" label |
| Low embedding match confidence | Cosine distance > threshold | "Unknown — report to add" |
| Unknown connector | Best match still too far | "Unknown" + report affordance |
| Inference throws | Sentis exception | Banner, auto-recover next frame |
| AR tracking lost | ARFoundation state | "Move phone to re-track" |
| Thermal throttle | Device temp threshold | "Cooling" icon, drop to 15 Hz |

### Philosophy

Honesty over false precision. The app surfaces uncertainty clearly. It never hides a confidence value behind a cleaner-looking label. The scariest failure mode — a confident wrong answer on a precision connector — is architecturally prevented by the fusion rule: ML and measurement must agree.

### Smoothing (not hiding)

- Three-frame detection debounce suppresses momentary flicker.
- Exponential moving average on verdicts prevents label jitter between adjacent frames.
- Depth median-filtering over a 5×5 patch handles depth-sensor noise.

## 9. Self-Improvement Loop

### Signal capture (on device)

Every detection writes a signal record to an encrypted local SQLite store:

```
{
  timestamp, device_id, app_version, model_version, reference_db_version,
  crop_image (384×384 jpg),
  depth_patch (16×16 fp16),
  embedding (128d),
  top3_matches: [(class, cosine_dist), ...],
  measurement: {pixel_dia, depth_mm, physical_dia_mm},
  fusion_verdict: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN",
  user_action: "confirmed" | "corrected_to_X" | "ignored" | "timed_out"
}
```

Only crops (never full frames) are retained. EXIF stripped. Opt-in upload batched over HTTPS.

### Four sources of learning signal

| Source | Value |
|---|---|
| Explicit confirmation (Yes) | Strong positive |
| Explicit correction (No + picks class) | Strong positive + negative |
| Unknown flag (distance > threshold) | New-class candidate |
| ML / measurement disagreement | Edge case, hard-negative mining |

### Offline pipeline

1. Ingest uploaded logs.
2. Auto-triage by signal strength.
3. Human review for ambiguous cases. No silent auto-labeling.
4. Retrain on cadence (monthly for embedder, weekly for reference DB).
5. Eval gate enforces regression-free ship.
6. Sign and publish new bundle.
7. App downloads, verifies signature, swaps model live.

### Eval gate (hard rules)

A new model ships only if all pass on the frozen regression set:

- Top-1 accuracy ≥ previous − 0.5%
- No per-class recall drop > 2%
- Golden demo set ≥ 99%
- Expected calibration error (ECE) ≤ previous + 0.02

Plus: improves on at least 50% of hard-correction cases from the field.

Gate is implemented in code; bypass is not possible through the deploy tooling.

### Two cadences

- **Model retraining:** monthly. Requires full eval gate.
- **Reference database update:** weekly. Adds confirmed new reference embeddings without retraining. Lower-risk path for onboarding new connector types.

### Safety rails

1. Server-signed models; app verifies signature before loading.
2. No silent auto-labeling.
3. Eval gate cannot be skipped.
4. Previous model retained for one-tap rollback.
5. Only crops leave device. Opt-in only.
6. Every log record versioned with model + reference DB version.

## 10. UX Principles

### Anchoring principles

1. **Glanceable** — primary answer readable in under a second.
2. **Hold-and-read** — tech never needs two hands on the phone.
3. **Honest** — uncertainty always surfaced.
4. **Visibly improving** — self-improvement is surfaced to the tech.

### First run

No account, no onboarding carousel, no feature tour. App launches into the scanner. First successful ID shows a single inline hint about the specs card. Second ID onward is silent.

### Screen hierarchy

- Primary: class + size, large and high-contrast.
- Secondary: confidence bar, always visible.
- Tertiary: tap-for-specs affordance.
- Bottom status: model version and last update date.

### States

| State | Visual |
|---|---|
| Scanning | Subtle corner reticle pulsing |
| Detecting | Thin dashed ring, no label |
| HIGH verdict | Solid ring, full label, confidence bar filled |
| MEDIUM verdict | Dashed amber ring, "Precision — hold steady" |
| Unknown | Grey ring, "Unknown — report to add" |
| Mating hazard | Red banner, haptic warning, modal |

### One-handed ergonomics

- All tappable UI in bottom third of screen.
- Specs card slides up, swipe-down to dismiss.
- Confirmation prompts dock at bottom, ≥44pt buttons.
- No hamburger menu. Settings via two-finger tap or long-press on status bar.

### Confirmation prompts (55–75% band)

- Non-blocking card at bottom of screen.
- Three options: Yes / No / Not sure.
- Max one prompt per 60 seconds.
- No gamification. This is a work tool.

### Mating warnings

Modal only for safety — the one exception to the non-modal rule. Distinct haptic pattern. "Why?" button explains the incompatibility in one short sentence. Builds trust and teaches junior techs.

### Specs card

On tap, the label expands into a full-width card:

```
SMA Female (SMA-F)
Frequency:    DC – 18 GHz
Impedance:    50 Ω
Standard:     MIL-STD-348
Mating torque: 8 in-lb (0.9 N·m)
Thread:       1/4-36 UNS-2B
Dielectric:   PTFE
Mates with:   SMA-M ✓
Forbidden:    2.4mm-M ⚠  2.92mm-M ⚠

[ Copy specs ]  [ Report issue ]
```

### Accessibility

- Full VoiceOver / TalkBack support for all verdicts.
- Dynamic type.
- High-contrast auto-adapts to OS.
- Haptic + shape redundancy for colorblind safety (solid/dashed/dotted rings paired with colors).

### Lifecycle

- Pauses on background; resumes in <200ms.
- Thermal throttling drops detector to 15 Hz before killing anything.
- Session memory (last 5 minutes) kept for mating checks; cleared on background.
- No network activity when telemetry is opted out.

### Self-improvement transparency

Status bar at the bottom of the screen: "Model v7 • Updated 3 days ago • 247 confirmations contributed." Tap opens an improvement history card. This is the single most important UI element for the pitch — the tech sees the app getting better because of their participation.

## 11. Testing Strategy

### Layer 1 — ML validation (Python)

- Regression set: 50 images/class, frozen, never trained on.
- Golden demo set: 5 images/class, perfect conditions. Every model ≥99%. (Built in Phase 1 once real connectors arrive; Phase 0 uses a synthetic-only proxy golden set.)
- Synthetic adversarial set: extreme lighting, rotation, occlusion renders.
- Calibration: reliability diagram; ECE tracked per release.
- Measurement stage: synthetic images with known Ø, required ±0.1mm on clean renders.

### Layer 2 — Unity validation

Play-mode tests (CI, no device):

- Mock camera feeds known video, pipeline emits expected detections
- Empty frames → no exceptions, no ghost overlays
- Multi-connector frames → multiple anchors
- Forced low confidence → ConfirmationPrompt triggers
- Forced ML/measurement disagreement → MEDIUM verdict
- Mating rule hit → warning modal
- OTA model swap → new model loads without restart

Device tests (manual, per release, recorded):

- Launch to first scan < 3s
- 30 FPS sustained × 60s
- Thermal stays ≤ "Fair" × 5 min
- All 8 demo connectors ID'd in ≤5s each on golden set
- Two-connector scene: both rings stable × 30s
- Background/foreground resume < 200ms
- Airplane mode: scanner works, confirmations queue
- AR tracking loss/recovery: no orphaned anchors
- Low light: confidence drops, no false commits
- Mating warning fires correctly, haptic fires, dismisses cleanly

### Layer 3 — Demo validation

The 90-second pitch scenario is a tested feature:

1. Launch → scanner live (0:00)
2. SMA-F → instant ID (0:10)
3. Tap label → specs card (0:20)
4. 2.92mm male → precision-size ID with measured Ø (0:35)
5. Proximate to SMA-F → mating warning (0:50)
6. Untrained connector → "Unknown — report to add" (1:00)
7. Tap "Add this" → report flow (1:10)
8. Improvement history card (1:20)
9. Close (1:30)

Demo-mode toggle pre-seeds "recently scanned" state so mating warning fires on cue.

### Post-launch signals

- Correction rate (target: decreasing over time)
- Unknown rate (indicates new connectors appearing)
- Abandonment rate (UX regression signal)
- Field calibration (80% prediction should be ~80% correct)

## 12. Phases and Rough Timeline

| Phase | Weeks | Output |
|---|---|---|
| 0. Scaffolding + proxy data | 1–3 | Unity app skeleton, Python training pipeline, proxy-data-trained V0 model |
| 1. Real connectors arrive | 4–5 | Capture + label real-image dataset; retrain; regression set frozen |
| 2. UX polish + demo prep | 6–7 | Specs cards, mating warnings, demo scenario rehearsal |
| 3. Self-improvement plumbing | 8 | OTA pipeline, signing, eval gate, confirmation logging |
| 4. Field pilot (internal) | 9–10 | Demo to Anduril, iterate on real feedback |

## 13. Open Questions

- Which specific iPhone Pro and Android flagship will be the primary test devices? (Affects LiDAR vs ToF tuning.)
- Is there an Anduril-owned connector catalog (with specs) we can license for `connector_specs.json`, or do we assemble from public sources?
- Who owns the backend for OTA model hosting? Cloud provider? Anduril environment?
- Data retention policy for uploaded confirmations — how long, where stored, who can access?
- Signing key management — whose HSM holds the model-signing key?

These are resolvable during engagement and do not block the MVP.

## 14. Future Work (Explicitly Deferred)

- Tier-B device support (mid-range phones, no LiDAR/ToF).
- Damage detection.
- Inventory / shelf-scanning mode.
- Multi-language support.
- Anduril parts database integration.
- Class activation map ("why this answer?") visualization.
- Photogrammetry rig for automated reference capture.
