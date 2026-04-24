# Spec Amendment — On-Device Enroll Architecture

**Date:** 2026-04-24
**Amends:** `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`
**Status:** Draft — supersedes the data-strategy and training-pipeline portions of the original spec

## Summary of change

The original spec assumed a server-trained classifier shipped with each app build, with the reference database baked in at training time. This amendment pivots to an **on-device enroll** architecture: the app ships with a frozen general-purpose embedder; technicians teach the app each connector type by holding it up to the camera in an Enroll mode. Identification mode then uses an entirely on-device reference database that the tech (or their team) built locally.

The original perception pipeline, fusion rules, AR overlay, and UX principles are unchanged. Only the data flow and lifecycle around the reference database change.

## Why

1. **Web-scraped training data is noisy and unrepresentative.** Connectors photographed in catalogs do not look like connectors held under lab lighting. On-device enroll captures the actual connector under the actual lighting where it will be identified. Domain match is perfect.
2. **Adding the 9th, 10th, Nth connector type required a retrain + redeploy.** With on-device enroll, a tech adds a new connector in 30 seconds and the app identifies it immediately. Critical for any deployment where the connector inventory grows.
3. **The original architecture had a chicken-and-egg problem.** We needed a trained model before we could ship a useful app, but we needed the app deployed to gather real training data. On-device enroll inverts this: ship the app with a generic backbone; let it earn its reference database in the field.
4. **The "self-improvement" pitch becomes literal and immediate.** The first interaction a tech has with the app is teaching it. The flywheel runs entirely on-device with no server dependency on day one.
5. **It changes the demo dramatically.** Instead of "we trained a model and it works," the demo is "hand me an unknown connector, watch me enroll it on the spot, watch me identify it." That is the artifact Anduril buys.

## What stays the same

- Two-stage perception (detect → embed → match → measure → fuse).
- Confidence-fusion rule for precision-size connectors (ML class + physical measurement must agree).
- World-anchored AR overlay (ring + label).
- All UX principles (glanceable, hold-and-read, honest, visibly improving).
- All testing strategy.
- ONNX → Sentis runtime path.

## What changes

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNITY APP (iOS + Android)                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Two scene modes                       │   │
│  │                                                          │   │
│  │   SCAN scene  ─── unchanged from original spec ────────▶ │   │
│  │                                                          │   │
│  │   ENROLL scene ── NEW                                    │   │
│  │     • Tech picks a class label (existing or new)         │   │
│  │     • App captures ~150 frames over 5 s                  │   │
│  │     • Pipeline embeds each frame; clusters into K=3      │   │
│  │       canonical references                               │   │
│  │     • Writes to on-device reference DB                   │   │
│  │                                                          │   │
│  │   CURATE scene ── NEW                                    │   │
│  │     • Browse enrolled classes + sample reference frames  │   │
│  │     • Delete bad enrollments; re-enroll                  │   │
│  │     • Optional export/import of reference DB             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│              On-device reference database                       │
│                  (RFCE binary on disk)                          │
└─────────────────────────────────────────────────────────────────┘

         App ships with this:                Built on-device:
         ──────────────────────              ─────────────────────
         • embedder.onnx (frozen)            • reference_embeddings.bin
         • detector.onnx (frozen)            • Per-class capture history
         • connector_specs.json (defaults)
         • mating_matrix.json (defaults)
```

### Backbone training

Now optional. The app ships with a backbone trained on whatever data is available — in practice we'll use **pretrained DINOv2 (or MobileViT) with no fine-tuning** for V1. Pretrained ImageNet/DINOv2 features are good enough for general visual discrimination; the *class signal* comes from the on-device reference database.

The Python training pipeline (Plan 1) is retained for future backbone refinement: once enough enrollments accumulate across deployments, we can fine-tune the embedder on captured RF-connector imagery to sharpen its features specifically for this domain. But that's optimization, not blocker.

### Reference database lifecycle

**Original spec:** built offline by `build_references.py`, baked into `StreamingAssets/reference_embeddings.bin`, shipped with the app.

**Amended spec:** the on-device reference database lives in `Application.persistentDataPath` and is mutable. Initial state is empty; the tech enrolls each class on first use. Format remains the same RFCE binary so the existing `ReferenceDatabase` reader works unchanged.

### New: enrollment data structure

Each enrolled class can now have **multiple reference vectors** (not just one mean). During enrollment the 150 frames produce a buffer of embeddings; we cluster into K=3–5 prototypes (k-means on cosine distance) and store them all. Identification matches against any of them; the highest cosine similarity wins.

Why multiple prototypes per class: a connector looks meaningfully different from the front than from the side. One average-of-everything vector smears these together; K prototypes preserve them.

Binary-format extension is backward-compatible: each class can have N vectors instead of 1, with a new `vectors_per_class` field in the header. Bumps `FORMAT_VERSION` from 1 to 2. The Unity reader will accept both.

### Self-improvement loop

**Original spec:** confirmation prompts → server upload → human review → retrain → OTA deploy.

**Amended spec:** confirmation prompts → on-device update of reference DB. Specifically:
- "Yes, that's an SMA-M" → embedding goes into the SMA-M cluster, prototypes updated
- "No, it's actually an SMA-F" → embedding moves to SMA-F cluster, both classes' prototypes updated
- "It's a new connector type" → opens the Enroll flow with the captured embedding pre-seeded

Server upload remains optional for cross-device sharing and for backbone-refinement training data, but is no longer on the critical path for any user-visible improvement.

### Changes to phased timeline

Original 4-phase plan compresses substantially:

| Phase | Original | Amended |
|---|---|---|
| 0 | Build pipeline + proxy-data model (3 wk) | Wrap pretrained backbone, no training (1 wk) |
| 1 | Capture real connector data + retrain (1–2 wk) | Built into the Enroll flow itself |
| 2 | UX polish + demo prep (2 wk) | Same |
| 3 | Self-improvement plumbing (1 wk) | Reduced — most logic now on-device |
| 4 | Field pilot (1–2 wk) | Same |

Net: ~2 weeks faster to demoable artifact.

## Plan implications

- **Plan 1 (Python training pipeline):** retained but no longer blocking. The embedder ONNX shipped to Unity will be a pretrained-only export (new helper script: `make_pretrained_embedder.py`). The full training pipeline becomes a future-work optimization.
- **Plan 2 (Unity scanner MVP):** unchanged.
- **Plan 2b (NEW — replaces the "rich UI" plan that was Plan 2b):** Unity Enroll + Curate scenes + on-device reference DB writer. Specs card + mating warning + confirmation prompt UIs are deferred to Plan 2c.
- **Plan 3 (model integration):** simplified — only the embedder ONNX and detector ONNX need integration; references are built on-device.
- **Plan 4 (self-improvement):** scope shrinks dramatically. On-device updates land in Plan 2b; server-side flywheel becomes optional.

## Risks introduced by this pivot

1. **Pretrained backbone may not separate visually-similar precision connectors.** Mitigation: relies on the metrology cross-check (already in the design). If the embedder collapses 2.92mm and 2.4mm to the same vector, measurement breaks the tie.
2. **Enrollment quality is uneven.** A tech who enrolls badly produces a bad reference. Mitigation: Curate UX with sample-frame previews, easy delete + re-enroll, "test against current reference set" sanity check at end of enrollment.
3. **Cross-device variance.** Two phones in the same lab might enroll the same connector slightly differently. Mitigation: optional reference-DB export/import for team-wide sharing.
4. **No "ground truth" reference set.** Without a centrally-curated database, there's no benchmark to evaluate against. Mitigation: each enrollment captures 150 frames; we hold out 20 as a per-deployment validation set.

## Approval needed

This amendment changes Plan 1's deliverables, supersedes the original Plan 2b, and shrinks Plan 4. Implementation continues against this amended spec.
