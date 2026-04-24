# Spec Amendment — CAD Synthetic Data + Scale Marker Workflow

**Date:** 2026-04-24
**Amends:** `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md` and `docs/superpowers/specs/2026-04-24-on-device-enroll-amendment.md`
**Status:** Draft

## Summary

Three changes based on an expert review of the original architecture:

1. **Backbone training data switches from web-scraped photos to CAD-rendered synthetic data.** Using publicly-available STEP files from Southwest Microwave, Rosenberger, Huber+Suhner etc., rendered in Blender with physically-accurate materials and full domain randomization, we produce thousands of dimensionally-exact training images per class overnight on a single GPU.
2. **ArUco scale-marker requirement added to the identification flow for precision-size discrimination.** Monocular vision without scale reference cannot reliably separate 2.4/2.92/3.5 mm; a printed ArUco marker in the capture frame gives pixel-to-mm ground truth and pushes accuracy from ~80% to ~95%.
3. **Honest accuracy ceiling stated explicitly in the spec** to set expectations with stakeholders before engagement.

The on-device enroll architecture (previous amendment) is unchanged. These amendments affect how we train the backbone and what the capture UX requires — not the enroll/identify flow itself.

## Why

**Web-scraped data has three irreducible problems:** it's dimensionless (no ground-truth scale), it's unevenly distributed across classes, and distribution sites actively block scraping. CAD-rendered data fixes all three: STEP files are authoritative, can be rendered in any quantity, and are free of legal/technical scraping constraints.

**Monocular scale ambiguity is physics.** A 2.4 mm connector at close range and a 3.5 mm connector at farther range produce identical pixel patterns. Without a second sensor (LiDAR) or a known-size reference in frame (ArUco marker), the accuracy ceiling for precision-size discrimination is ~80%. Adding a scale marker to the capture workflow is the lowest-friction way to raise this ceiling without requiring hardware upgrades.

**Unstated accuracy expectations hurt engagements.** If Anduril signs expecting >98% on bare connectors in arbitrary photos, the deliverable fails on a physics problem, not a modeling problem. Stating the ceiling up front lets the requirement negotiation happen before the contract.

## Changes

### Backbone training data pipeline

**Was:** Train the embedder on ~30 images/class scraped from Bing/Digikey + procedurally-rendered PIL shapes.

**Is now:** Train the embedder on 2,000–5,000 images/class rendered from CAD in Blender with:

- Physically-accurate materials: brass/stainless-steel bodies, gold-plated contacts, PTFE dielectric (refractive index 1.38) for SMA, air for precision connectors
- Domain randomization: HDRI environment maps from Poly Haven, random background textures, randomized camera position + focal length + depth of field, sensor noise, motion blur, JPEG compression
- Exact camera intrinsics logged per render → downstream model can learn the true pixel-to-mm relationship
- Per-class coverage: every pose, every lighting, every focal length — no gaps that show up in the field

Catalog scraping and the labeler tool are retained but de-emphasized. They remain useful for hard-case supplementation and sanity validation, not as primary data.

### Mating-face keypoint crop

The current pipeline crops by YOLO bbox (variable aspect, variable scale). The amendment adds a lightweight mating-face keypoint detector: find the center of the mating face, crop a fixed pixel region around it. This standardizes the scale problem — every crop is always showing the same ~1 cm region at the same resolution. Bore-diameter ratios become directly learnable.

Implementation: extend YOLO with a keypoint head, or run a small regression network on the YOLO crop to predict the face centroid.

### Hierarchical classification structure

**Was:** Single 128-d embedding + nearest-neighbor match in one stage.

**Is now:** Embedding still single-shot, but training loss is hierarchical:
- Stage-1 auxiliary loss: classify family (SMA vs precision) and gender (male vs female) — high-contrast signals any model nails easily
- Stage-2 main loss: metric learning (ArcFace) over all 8 classes as before

At inference, only nearest-neighbor matching is used. The hierarchical auxiliary loss just helps the embedder organize its latent space during training.

### ArUco scale marker in the capture flow

Print a 25 mm × 25 mm ArUco tag (dictionary 4×4, ID 0). Techs place it next to the connector during capture. App requires the marker to be detected before committing an identification.

In ENROLL mode: marker required. Enrollment cannot complete without a detected marker in every frame. This pins the embedding space to a known physical scale.

In SCAN mode: marker preferred but not required. Without marker, the app still returns a verdict but labels it MEDIUM confidence for precision connectors (the "75-85% honest ceiling" regime). With marker, confidence can go HIGH.

Implementation: OpenCV's ArUco module in Python (for training-time augmentation with marker rendering), and a small Sentis-friendly marker detector in Unity. Or leverage ARKit/ARCore image-tracking if target detection is viable there.

### Accuracy ceilings to publish in the SOW

| Task | Expected accuracy | Caveat |
|---|---|---|
| Gender (M/F) | >99% | trivial |
| Family (SMA vs precision) | >98% | trivial — dielectric visible |
| Within precision (3.5/2.92/2.4 mm) *with* ArUco marker | 93–97% | achievable |
| Within precision *without* ArUco marker | 75–85% | monocular physics ceiling |
| Within precision *with* LiDAR depth | 95–99% | Pro-model iPhones only |

Recommendation: require ArUco marker in V1 capture workflow. LiDAR-depth cross-check remains a Pro-device bonus path.

## What doesn't change

- On-device enroll architecture
- Unity three-scene UX (Scanner, Enroll, Curate)
- ArcFace loss (already shipped)
- Metric learning + nearest-neighbor matching (already shipped)
- ConfidenceFuser rule for precision connectors (already shipped)
- INT8 quantization for deployment (already shipped)
- Testing strategy (though the test set structure needs upgrading; see Plan 3 rewrite)

## Plan impact

- **Plan 1** (Python training pipeline): retained as-is. Still trains on whatever data we provide; CAD-rendered data becomes the new primary source.
- **Plan 2** (Unity scanner MVP): unchanged.
- **Plan 2b** (Unity Enroll + Curate): unchanged, except the ENROLL flow will gate on ArUco detection. That's a small additional UX step in the Enroll scene.
- **Plan 3** (model integration): **rewritten** to be "CAD→Blender synthetic pipeline + mating-face keypoint + backbone training + field test set + ArUco integration." See new plan document.
- **Plan 4** (self-improvement): unchanged.

## Open questions

- Which CAD vendors to prioritize? (Southwest Microwave confirmed to publish STEP files; verify Rosenberger + Huber+Suhner offer them.)
- Blender Python API or a headless CLI wrapper? (Probably `bpy` pip-installable for Blender 4.x + headless EGL.)
- Keypoint detection architecture? (YOLO-Pose vs. separate lightweight regressor.)
- ArUco marker size on the physical tag — must be large enough to detect reliably at typical capture distance but small enough to fit alongside a connector on a lab bench. 25 mm feels right.

These are resolvable during Plan 3 execution.
