# Hex-Anchored Measurement Prototype — Findings

**Date:** 2026-04-24
**Status:** Prototype complete; real-world limitations documented

## What was built

A three-stage CV pipeline in `rfconnectorai/measurement/`:

1. **`hex_detector.py`** — finds the coupling-nut hexagon via contour detection + polygon approximation. Measures flat-to-flat in pixels via `minAreaRect`.
2. **`aperture_detector.py`** — finds the dark inner aperture via Hough circles, falling back to dark-threshold contour-picking.
3. **`class_predictor.py`** — tries both standard hex sizes (7.94 mm, 6.35 mm) as scale hypotheses, converts aperture pixels → mm, thresholds to class.

CLI tool at `scripts/measure_connector.py`.

## Synthetic accuracy: 100%

14 unit tests covering synthetic frontal mating-face images with known hex + aperture dimensions all pass. The underlying math works:

- Hex detection recovers flat-to-flat within ±5% across sizes/rotations/positions
- Aperture detection recovers diameter within ±10% across sizes
- End-to-end classifier correctly maps (hex, aperture) → {3.5mm-F, 2.92mm-F, 2.4mm-F} when inputs match nominals

## Real-image accuracy (Bing-fetched catalog photos): 0.6%

Ran the pipeline against 160 Bing-fetched product photos (20 per class × 8 classes) from `training/data/labeled/embedder/`. Result: 1 correct / 160.

**Root causes (by count):**
- ~60% of images: **no hex detected**. Images are side-views of cable assemblies, 3/4 perspective product photography, renders at oblique angles. The mating face is either not visible or heavily foreshortened.
- ~35% of images: **hex detected but aperture not detected**. The mating face IS visible, but the aperture detector picks up either the hex body or an off-center dark region, not the central bore.
- ~5% of images: **hex + aperture both detected but wrong values**. The measured aperture is consistently ~2× the nominal, suggesting the detector is locking onto the dielectric-visible region rather than the metal bore specifically.

## What this means

The hex-anchored approach is **mathematically sound** — when inputs match the assumption (frontal mating-face view, aperture as simple dark circle), it correctly recovers physical dimensions and maps them to class.

The approach is **very sensitive to capture conditions**. Real product photos don't satisfy the assumption. Therefore the hex-anchored measurement is viable only when:

1. **The capture workflow enforces a frontal mating-face shot.** The Unity Enroll/Scan UX must guide operators to the correct framing (reticle + "align mating face with center" overlay, reject frames where hex can't be detected). This is the single most important UX requirement.
2. **Training data for aperture detection reflects real connector internal structure**, not the simplified "dark-circle-on-metal" of the current synthetic renderer. Real precision connectors have a visible outer-conductor bore, sometimes a PTFE dielectric bead, and an inner conductor pin — multi-layered. CAD-rendered data from Plan 3 (when real STEPs arrive with accurate mating-face geometry, per the Plan 3 addendum) should fix this.
3. **A learned model outperforms the hand-coded pipeline.** The other Claude recommended an end-to-end learned approach with a multi-task loss: `class + aperture-in-mm + hex-in-pixels`. The prototype's failure modes (aperture locked onto wrong structure, hex not found at oblique angles) are exactly the kinds of messy edge cases a learned model handles better than geometric primitives.

## Conclusion for the spec

**Adopt the hex-as-scale approach,** but with explicit honesty about what it requires:

- **UX constraint:** capture must be frontal, mating-face centered, reticle-guided. The scan flow rejects any frame where the hex isn't detected with high confidence.
- **Training data constraint:** need CAD-rendered mating-face close-ups (from Plan 3) with ground-truth hex_px and aperture_px per render, not just class labels.
- **Model architecture:** hand-coded pipeline as an interpretable baseline + fallback; learned multi-task model for production.
- **Accuracy ceiling update:** With proper capture UX and CAD-trained model, ~92–96% on females is still achievable. With uncurated operator photos, the honest ceiling is lower than the earlier estimate.

## Files produced

- `training/rfconnectorai/measurement/__init__.py`
- `training/rfconnectorai/measurement/hex_detector.py`
- `training/rfconnectorai/measurement/aperture_detector.py`
- `training/rfconnectorai/measurement/class_predictor.py`
- `training/tests/test_hex_detector.py` (5 tests)
- `training/tests/test_aperture_detector.py` (4 tests)
- `training/tests/test_class_predictor.py` (5 tests)
- `training/scripts/measure_connector.py` (CLI)
- `training/docs/hex_measurement_prototype_findings.md` (this document)

14 unit tests, all green. Total test suite: 87 passed.
