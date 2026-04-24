# Spec Amendment — Hex-Anchored Measurement as Primary Scale Reference

**Date:** 2026-04-24
**Amends:** `docs/superpowers/specs/2026-04-24-cad-synthetic-and-scale-marker-amendment.md` (which added ArUco)
**Status:** Draft

## Summary

Two architectural changes driven by a review insight and a working prototype:

1. **The coupling-nut hex IS a built-in scale reference.** Every precision RF connector in our initial scope has a standardized hex (5/16 inch for 3.5/2.92 mm; 1/4 inch for 2.4 mm). Detecting the hex in the image yields a known physical length; converting aperture pixels to millimeters then maps directly to the connector class. No external scale marker required for female connectors.
2. **This only works with disciplined capture.** A prototype (committed this session) confirms the math is exactly right on ideal frontal mating-face images but collapses on uncurated product photography. The Unity capture UX must enforce a frontal mating-face framing — reticle-guided, hex-detection-gated — for the approach to deliver. This is a meaningful constraint on the product.

## What changes

### ArUco becomes a fallback, not the primary

Previous amendment: scale marker required for precision-size discrimination. New decision: hex is primary for female connectors. ArUco remains:

- **Required** for male connectors (no coupling nut hex on the plug side)
- **Optional fallback** for female connectors when the hex isn't reliably detected (edge of frame, heavy occlusion, oblique angle)

### Capture UX gains a framing gate

The Scan and Enroll scenes must:
1. Run the hex detector on every camera frame at ~30 Hz.
2. Display a reticle that turns green only when a hex is detected within the central region.
3. Refuse to commit an identification or enrollment frame unless the hex is green.
4. Show a text hint: "Center the mating face in the reticle" when hex not detected; "Hold steady" when detected but verdict not yet stable.

This is the single most important UX addition. Without it, the identification pipeline is vastly less accurate.

### Accuracy ceiling — updated again

| Task | Previous estimate | Revised estimate (with hex-anchored + enforced framing) |
|---|---|---|
| Gender (M/F) | >99% | >99% |
| Family (SMA vs precision) | >98% | >98% |
| Within precision (3.5/2.92/2.4 mm), female, with hex visible | 93–97% | **95–98%** |
| Within precision, male | 93–97% with ArUco | **80–90% with ArUco; ~75% without** |
| Within precision without hex or ArUco | 75–85% | **should not ship — require one or the other** |

The honest ceiling for females goes up (hex is a free, always-available scale reference). The honest ceiling for males goes down (no coupling nut; either need ArUco marker, a cover-off shot showing body hex, or accept lower confidence).

### Plan 3 (CAD synthetic) gains a mating-face-prominence requirement

The current Plan 3 renders generic views of connector meshes. The hex-anchored pipeline needs training data that's almost entirely **frontal mating-face close-ups** with known ground-truth hex_px and aperture_px per render.

Specifically the pipeline config should add:
- Primary render mode: connector axis pointed at the camera, mating face centered, frame-fitted
- Secondary augmentation: small rotations and perspective jitter around this primary axis (±15° yaw/pitch), not full random rotation
- Metadata manifest per render: `hex_flat_to_flat_px`, `aperture_diameter_px`, `mating_face_normal` for keypoint supervision

### Plan 4 — Hex-Anchored Measurement Pipeline

New plan file: `docs/superpowers/plans/2026-04-24-plan-4-hex-measurement.md`. Scope:

- Productionize the prototype detectors from `rfconnectorai/measurement/`
- Add a learned aperture segmentation head (replaces the hand-coded Hough-circles approach with a small U-Net-style network trained on CAD renders)
- Add the framing-gate component to the Unity Scanner and Enroll scenes
- Add ArUco detection for male connectors and as fallback
- Integrate measurement into the `ConfidenceFuser`: HIGH when both ML embedding and hex-aperture measurement agree; MEDIUM when only one is confident; LOW otherwise

## Prototype findings (reference)

See `training/docs/hex_measurement_prototype_findings.md` for the full writeup. Headline numbers:

- Synthetic frontal mating-face images: 14/14 tests pass, end-to-end correct
- Real Bing-fetched catalog photos: 0.6% correct (mostly side views and cable-assembly shots)

The 0.6% result is not a failure of the approach — it's a demonstration that uncurated photos don't satisfy the approach's assumptions. The spec amendment exists to make those assumptions contractual in the capture UX.

## What doesn't change

- On-device enroll architecture
- Metric learning + nearest-neighbor matching
- Sentis runtime on Unity
- Confidence-fusion rule for precision connectors
- All existing testing strategy
- Ship-ready status of Plans 1, 2, 2b, and 3

## Open questions for Plan 4 execution

- Is learned aperture segmentation worth the training cost vs. a heavy-duty hand-coded approach that handles threads + dielectric + pin explicitly? (Probably yes once CAD renders are plentiful; decide during Plan 4.)
- Should framing-gate detection run on-device or use a snapshot-then-verify model? (Likely on-device, lightweight YOLO-pose or similar.)
- What's the threshold for "hex detected with enough confidence to commit"? (Empirically set from CAD-rendered test images once available.)

These resolve during Plan 4 execution, not in this amendment.
