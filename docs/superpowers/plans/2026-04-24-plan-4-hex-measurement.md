# Plan 4: Hex-Anchored Measurement Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the prototype hex-aperture measurement into a production-capable system. Adds a learned aperture segmentation model, a Unity capture-framing gate, ArUco fallback for male connectors, and integration with the existing `ConfidenceFuser`.

**Architecture:** The prototype at `training/rfconnectorai/measurement/` stays as the interpretable CPU-CV fallback. We add a compact learned aperture segmentation model (U-Net Mini) trained on Plan 3 CAD renders. Unity gets a `FramingGate` component that runs the hex detector on camera frames at 30 Hz and gates the Scan/Enroll commit path. Male connectors require an ArUco marker whose detector is added as an OpenCV pre-pass (runs in Python for training-data preparation; ported to Unity via a small C# wrapper around OpenCV for Unity or reimplemented in C#).

**Tech Stack:** Python 3.11, PyTorch, OpenCV (already installed). Unity 6 + C#. Spec: `docs/superpowers/specs/2026-04-24-hex-anchored-measurement-amendment.md`.

---

## File structure

```
training/
├── rfconnectorai/
│   ├── measurement/
│   │   ├── hex_detector.py             (existing)
│   │   ├── aperture_detector.py        (existing — stays as CPU fallback)
│   │   ├── class_predictor.py          (existing — extended for learned path)
│   │   ├── aruco_detector.py           (NEW)
│   │   ├── aperture_unet.py            (NEW — PyTorch model)
│   │   └── train_aperture.py           (NEW — training script)
│   └── data/
│       └── aperture_dataset.py         (NEW — loads CAD renders + masks)
└── tests/
    ├── test_aruco_detector.py
    ├── test_aperture_unet.py
    └── test_aperture_dataset.py

unity/RFConnectorAR/Assets/Scripts/
├── AR/
│   └── FramingGate.cs                  (NEW)
└── App/
    └── AppBootstrap.cs                 (MODIFY — integrate gate)
```

---

## Task 1: ArUco detector (Python)

**Files:**
- Create: `training/rfconnectorai/measurement/aruco_detector.py`
- Create: `training/tests/test_aruco_detector.py`

OpenCV's `aruco` module is the natural implementation. 4×4 dictionary, ID 0, 25 mm tag. Returns pixel-size of the tag edge for scale calibration.

- [ ] Write failing test that renders a synthetic 4×4 ArUco tag into a blank image and verifies the detector recovers the tag edge in pixels within 5% of the drawn size.
- [ ] Implement `aruco_detector.py` with a single function `detect_aruco_marker(image, marker_size_mm=25.0) -> ArucoDetection | None` returning edge_px + pixels_per_mm.
- [ ] Verify tests pass.
- [ ] Commit: `feat(training): ArUco marker detector for male-connector scale fallback`

## Task 2: Class predictor — male-connector path

**Files:**
- Modify: `training/rfconnectorai/measurement/class_predictor.py`

Extend the predictor to accept a male/female gender hint:
- Female: hex-anchored (current behavior)
- Male: use ArUco if present; fall back to pin-diameter measurement (inner pin ≈ the class signature)

- [ ] Add `predict_class_male(image, ...) -> Prediction` that uses ArUco for scale and finds the pin-diameter at the image center.
- [ ] Add top-level `predict_class_with_gender(image, gender: str) -> Prediction` that routes to the appropriate sub-pipeline.
- [ ] Update tests.
- [ ] Commit: `feat(training): class predictor handles male connectors via ArUco + pin measurement`

## Task 3: CAD-rendered aperture dataset + masks

**Files:**
- Create: `training/rfconnectorai/data/aperture_dataset.py`
- Create: `training/tests/test_aperture_dataset.py`
- Modify: `training/rfconnectorai/synthetic/render.py` to emit aperture-mask PNG per render

Plan 3's CAD renderer needs to also save a per-render mask of the aperture region (derived from the mesh UV or a Blender-side vertex-color annotation). The training dataset reads these pairs.

- [ ] Extend `render_single` in Plan 3's renderer to output `mask_<seed>.png` alongside `render_<seed>.png` — white where aperture, black elsewhere.
- [ ] Write `ApertureDataset` that loads (rgb, mask) pairs per class.
- [ ] Tests.
- [ ] Commit.

## Task 4: Aperture segmentation U-Net

**Files:**
- Create: `training/rfconnectorai/measurement/aperture_unet.py`
- Create: `training/rfconnectorai/measurement/train_aperture.py`
- Create: `training/tests/test_aperture_unet.py`

Small U-Net (32 base channels, 4 stages) that takes a 256×256 RGB crop centered on the detected hex and outputs a binary aperture mask. Trained on the Plan 3 CAD data with the masks from Task 3.

- [ ] Implement U-Net in `aperture_unet.py` with forward test (input shape in → output shape out, gradients flow).
- [ ] Training script with smoke-test flag (2 epochs, tiny data).
- [ ] Tests pass.
- [ ] Commit.

## Task 5: End-to-end measurement with learned aperture

**Files:**
- Modify: `training/rfconnectorai/measurement/class_predictor.py`

The learned aperture segmentation replaces the Hough-circles step. Pipeline becomes:
1. Hex detection (CPU)
2. Crop around hex
3. Learned aperture segmentation on crop
4. Measure mask diameter in pixels → mm via hex scale
5. Classify

- [ ] Add `predict_class_learned(image, unet_path, ...) -> Prediction`
- [ ] Integration test: train on a tiny synthetic dataset, run inference, verify accuracy on held-out synthetic.
- [ ] Commit.

## Task 6: FramingGate in Unity

**Files:**
- Create: `unity/RFConnectorAR/Assets/Scripts/AR/FramingGate.cs`
- Modify: `unity/RFConnectorAR/Assets/Scripts/App/AppBootstrap.cs`
- Modify: `unity/RFConnectorAR/Assets/Editor/SceneBuilder.cs` to add reticle UI elements

`FramingGate` runs on every camera frame. Uses a port of the hex detector (or a small learned detector exported to ONNX and run via Sentis). Emits a boolean "hex in center" signal consumed by AppBootstrap and EnrollController, and drives a reticle color change in the UI.

- [ ] Implement `FramingGate` (C#) using Unity's built-in OpenCV alternative, or call out to a small Sentis model. (Decide during implementation.)
- [ ] Integration in AppBootstrap and EnrollController: commit verdict only when framing gate is green.
- [ ] SceneBuilder adds a reticle image + color-change wiring.
- [ ] Commit.

## Task 7: ConfidenceFuser integration

**Files:**
- Modify: `unity/RFConnectorAR/Assets/Scripts/Perception/ConfidenceFuser.cs`
- Modify: `unity/RFConnectorAR/Assets/Tests/EditMode/ConfidenceFuserTests.cs`

Extend the fuser to incorporate the hex-aperture measurement alongside the ML embedding classifier:

- ML says A, measurement says A → HIGH
- ML says A, measurement says "Unknown" → MEDIUM
- ML says A, measurement says B (different class) → LOW
- Only measurement, no embedding → MEDIUM
- Only embedding, no measurement → MEDIUM

- [ ] Update the fuser logic.
- [ ] Update tests.
- [ ] Commit.

## Task 8: End-to-end integration test

- [ ] Unity: open Scanner scene, camera at a known 2.4mm connector image, verify framing gate goes green, ML + measurement agree, verdict commits with HIGH confidence.
- [ ] Document the field-test protocol for the first real-device run.

## Plan Self-Review

(After writing the plan, check: spec coverage? placeholders? type consistency? scope reasonable?)

- Covers the hex-anchored measurement (Tasks 1, 2, 5), learned model path (Tasks 3, 4), Unity integration (Tasks 6, 7), end-to-end verification (Task 8).
- Dependencies on Plan 3: Task 3 needs Plan 3 renderer to emit masks. If Plan 3 hasn't produced CAD renders yet when Plan 4 starts, Task 3 blocks. Otherwise independent.
- Task 6 has an open architectural question (built-in CV vs Sentis-exported model) — intentionally left to implementation decision, not a placeholder.

## Execution handoff

Plan complete. Next step is one of:
1. Start Task 1 (ArUco detector — small, low-risk, no blocking dependencies).
2. Wait until real CAD renders exist (Plan 3 prerequisite for Tasks 3–5) and then execute in order.
3. Build the Unity FramingGate first (Task 6) since that's the user-visible UX piece and doesn't depend on training.

My recommendation: **Task 6 first.** Even without the learned aperture model, the framing gate alone fixes the capture-discipline problem — which the prototype findings show is the #1 limitation. After that, the learned aperture pipeline layers on top when CAD data is available.
