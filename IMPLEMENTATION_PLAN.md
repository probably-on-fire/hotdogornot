# SMA Connector AI — Implementation Plan

## 1. Mission

Build a production-grade RF/SMA connector identification system that allows an end-user to point a mobile or desktop camera at an SMA-family coaxial connector and immediately receive:

1. Connector family/type.
2. Standard vs reverse polarity.
3. Gender/contact configuration.
4. Physical/mechanical subtype.
5. Estimated size/geometry attributes.
6. Confidence, ambiguity warnings, and top alternative matches.
7. Cross-referenced connector specifications.

The project is not simply a “hotdog/not-hotdog” classifier. It must evolve into a connector-specific object detection, fine-grained classification, measurement-assist, and specification lookup system.

Current repository baseline:

```text
.
├── flutter/                 # iOS + Android Flutter camera app
├── training/                # FastAPI + PyTorch training/serving stack
├── docs/                    # cross-cutting docs
└── unity/                   # older AR app, historical/sidelined
```

Current production baseline from the repository README:

- Flutter app already exists.
- FastAPI predict service already exists.
- Current model is ImageNet-pretrained ResNet-18 with a 6-class linear head.
- Current held-out performance is approximately:
  - Full class: 75%
  - Family: 75%
  - Gender: 87.5%
  - Background false positives: 0%
- Held-out set is only 8 phone shots, which is not enough to prove the true accuracy of the model.

This plan assumes the existing codebase should be audited and reused where sound, but the model strategy should be upgraded from pure image classification to a more robust fine-grained detection + attribute classification pipeline.

---

## 2. Core Product Outcome

The end-user opens the app, points the camera at an RF connector, and gets a result card such as:

```json
{
  "detected": true,
  "connector_family": "SMA",
  "precision_family": "2.92mm / K / SMK",
  "polarity": "Standard SMA",
  "gender": "Female",
  "mount_style": "Bulkhead",
  "orientation": "Straight",
  "attachment": "Panel mount",
  "estimated_geometry": {
    "outer_thread_diameter_mm": "approximate / unknown until calibrated",
    "thread_count_or_pitch": "unknown / requires calibrated macro reference",
    "body_length_mm": "approximate / unknown until calibrated"
  },
  "confidence": 0.94,
  "top_alternatives": [
    {"label": "3.5mm female bulkhead", "confidence": 0.51},
    {"label": "SMA female bulkhead", "confidence": 0.47}
  ],
  "warning": "Visually similar precision connectors may require reference scale or manufacturer marking.",
  "spec_lookup": {
    "frequency_range": "up to 46 GHz",
    "impedance": "50 Ohm",
    "mating_compatibility": "mechanically compatible with SMA and 3.5mm, but performance limited when mated down"
  }
}
```

---

## 3. Classification Taxonomy

A single flat class label is too brittle. The correct architecture should produce multiple attributes.

### 3.1 Primary Families

Initial target families:

| Family | Notes |
|---|---|
| SMA | Standard SMA connectors, up to approximately 18 GHz |
| RP-SMA | Reverse polarity SMA; center contact gender is inverted |
| 3.5mm | Precision connector, mechanically SMA-compatible, higher frequency |
| 2.92mm / K / SMK | Precision connector, mechanically compatible with SMA/3.5mm |
| 2.4mm | Higher precision, not mechanically compatible with SMA/3.5mm/2.92mm |
| 1.85mm / V | Millimeter-wave precision connector |
| 1.0mm / W | Very high frequency precision connector |
| SSMA | Smaller SMA-related connector |
| SMB | Snap-on connector, lower-frequency family |
| SMC | Threaded smaller connector, lower-frequency family |
| QMA | Quick-lock SMA-like connector |
| TNC | Threaded RF connector |
| BNC | Bayonet RF connector |
| MCX | Small-form-factor snap-on connector |
| 7/16 DIN | Larger high-power RF connector |
| Unknown / unsupported | Explicit rejection class |

### 3.2 Attribute Heads

The model should infer these attributes separately:

| Attribute | Values |
|---|---|
| Presence | connector, no connector/background, uncertain |
| Family | SMA, RP-SMA, 3.5mm, 2.92mm, 2.4mm, 1.85mm, 1.0mm, SSMA, SMB, SMC, QMA, TNC, BNC, MCX, 7/16 DIN, unknown |
| Gender / contact | male pin, female socket, reverse-polarity male body/female contact, reverse-polarity female body/male contact, unknown |
| Polarity | standard, reverse polarity, not applicable, unknown |
| Mount style | cable mount, panel mount, bulkhead, PCB through-hole, PCB edge mount, PCB surface mount, adapter, terminator, unknown |
| Orientation | straight, right-angle, tee, adapter stack, unknown |
| Cable termination | solder, crimp, clamp, molded cable, unknown |
| Finish/material cue | gold, nickel/silver, black body, mixed/unknown |
| Size/geometry | estimated diameter, thread count/pitch, body length, connector aperture, requires calibrated reference |
| Confidence state | high confidence, ambiguous, insufficient view, requires second angle/reference scale |

### 3.3 Spec Lookup Database

Do **not** force the neural network to memorize specs. The neural network should classify visual identity and attributes. Then the app should cross-reference a structured database.

Create:

```text
training/rfconnectorai/specs/connectors.yaml
```

Example schema:

```yaml
SMA:
  impedance_ohm: 50
  frequency_range: "DC-18 GHz typical"
  coupling: "threaded"
  compatibility:
    mates_with: ["SMA", "3.5mm", "2.92mm mechanical compatibility with caution"]
  notes:
    - "Performance degrades at higher microwave frequencies."
RP-SMA:
  impedance_ohm: 50
  frequency_range: "varies by manufacturer"
  coupling: "threaded"
  visual_distinguishers:
    - "Center contact gender is reversed relative to standard SMA."
```

---

## 4. Architecture Recommendation

### 4.1 Replace “classifier-only” with a staged vision pipeline

Current ResNet classification is useful as a baseline but insufficient for fine-grained connector identification. The recommended production pipeline:

```text
Camera frame
  -> connector/background detector
  -> connector bounding box crop
  -> optional segmentation/mask refinement
  -> multi-head fine-grained classifier
  -> optional measurement/calibration module
  -> confidence calibration + ambiguity handling
  -> spec database lookup
  -> mobile/desktop result card
```

### 4.2 Model Strategy

Run these tracks in parallel and let metrics decide:

| Track | Purpose | Recommended models |
|---|---|---|
| Baseline preservation | Keep current ResNet path working | ResNet-18/50, EfficientNet/MobileNet classifier |
| Object detection | Locate connector in live camera frame | YOLO11n/YOLO26n, RT-DETR small, MobileNet-SSD as fallback |
| Fine-grained attributes | Classify family/gender/polarity/mount | EfficientNetV2, ConvNeXt-Tiny, MobileViT, YOLO classifier head, custom multi-head PyTorch |
| Segmentation | Clean crop/mask and measurement support | YOLO segmentation, SAM/SAM2-assisted labeling, rembg only as fallback |
| Mobile deployment | On-device inference | TFLite/LiteRT, ONNX Runtime Mobile, Core ML |
| Server fallback | Maximum accuracy path | FastAPI + PyTorch/ONNX Runtime |
| LLM/VLM assist | Explanations, spec text, second-opinion reasoning | Gemma/Gemini-style language layer, **not** the primary detector |

### 4.3 Recommendation on Gemma 3/4 or LLM/VLM use

Use Gemma-style mobile AI as an assistant layer, not as the primary object detector.

Correct use:

- Explain why the result is SMA vs RP-SMA.
- Convert model outputs into human-readable engineering guidance.
- Query/reason over the connector spec database.
- Ask user for another angle if ambiguity is high.
- Provide “likely vs not enough evidence” responses.

Do **not** depend on an LLM/VLM as the first-pass object detector for production. Fine-grained connector identity needs deterministic, measurable computer vision performance with a labeled dataset, confusion matrix, holdout discipline, and controlled deployment artifacts.

---

## 5. Accuracy Strategy

### 5.1 About the 99.99% target

99.99% should be treated as a strategic aspiration, not the first contractual acceptance criterion.

A true 99.99% claim requires very large, diverse, independently held-out test coverage. For example, if the test set only has 8 samples, one miss changes accuracy by 12.5 percentage points. Even 1,000 test images cannot robustly prove 99.99% field accuracy.

Use staged gates:

| Gate | Target | Meaning |
|---|---:|---|
| G0 | Current baseline documented | Reproduce current 75% result |
| G1 | 90%+ family accuracy | Enough to show credible improvement |
| G2 | 95%+ family/gender/polarity | Demo-worthy for client |
| G3 | 98%+ on curated holdout | Strong production candidate |
| G4 | 99%+ on large real-phone holdout | Production-grade |
| G5 | 99.9%+ with abstention | Achieved by returning “uncertain/needs another angle” instead of forcing guesses |
| G6 | 99.99% target | Requires large-scale field validation and probably active learning loop |

### 5.2 Use Abstention to Improve Real-World Correctness

The app should never guess confidently when visual evidence is insufficient.

Add output states:

- `high_confidence`
- `ambiguous_similar_connectors`
- `need_second_angle`
- `need_scale_reference`
- `unsupported_connector`
- `no_connector_detected`

For contract/demo purposes, “I need another angle” is much better than confidently misidentifying 2.92mm as SMA.

---

## 6. Dataset Plan

### 6.1 Immediate Dataset Audit

Codex should inventory:

```text
training/data/
training/data/labeled/
training/data/test_holdout/
training/docs/classifier_journey.md
training/docs/capture_protocol.md
```

Generate:

```text
docs/DATASET_AUDIT.md
```

Include:

- Existing classes.
- Number of images per class.
- Train/validation/test split.
- Real vs synthetic images.
- Phone-shot counts.
- Duplicate/near-duplicate detection.
- Blurry/low-light images.
- Background diversity.
- Angle diversity.
- Confusion-prone classes.
- Missing classes relative to the taxonomy above.

### 6.2 Minimum Data Targets

For a credible production demo:

| Data Type | Minimum target |
|---|---:|
| Real phone images per primary family | 300+ |
| Real phone images per attribute subtype | 100+ |
| Holdout images per primary family | 100+ |
| Background/no-connector negatives | 1,000+ |
| Confusing lookalikes | 100+ per lookalike family |
| Lighting/background variants | 10+ conditions |
| Angle variants | front, side, 45°, top, macro, partial occlusion |

### 6.3 Capture Protocol

For each physical connector:

1. Assign unique `specimen_id`.
2. Record known truth:
   - manufacturer/part number if available
   - connector family
   - gender
   - polarity
   - mount style
   - orientation
   - nominal diameter/thread type if known
3. Capture:
   - 12 photos per connector minimum.
   - 3 lighting conditions.
   - 3 backgrounds.
   - 4 viewing angles.
   - at least one calibrated reference image with ruler/caliper/known coin/reference card.
4. Place images into train or holdout at the **specimen level**, not image level, to prevent leakage.

### 6.4 Annotation Format

Adopt YOLO-format detection labels for bounding boxes:

```text
datasets/rfconnectors/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── attributes.csv
└── data.yaml
```

`attributes.csv`:

```csv
image_id,specimen_id,bbox_id,family,precision_family,gender,polarity,mount_style,orientation,termination,known_part_number,split,is_synthetic
```

---

## 7. Training Pipeline Plan

### 7.1 Required CLI Commands

Create or standardize these commands:

```bash
python -m rfconnectorai.data.audit --data-dir data --out docs/DATASET_AUDIT.md

python -m rfconnectorai.data.build_yolo_dataset \
  --input data/labeled \
  --out datasets/rfconnectors \
  --attributes-out datasets/rfconnectors/attributes.csv

python -m rfconnectorai.detector.train_yolo \
  --data datasets/rfconnectors/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --out models/detector

python -m rfconnectorai.classifier.train_multihead \
  --dataset datasets/rfconnectors \
  --backbone efficientnet_v2_s \
  --epochs 80 \
  --out models/multihead_classifier

python -m rfconnectorai.eval.evaluate_all \
  --detector models/detector/best.pt \
  --classifier models/multihead_classifier/best.pt \
  --holdout datasets/rfconnectors/test \
  --out reports/eval_latest

python -m rfconnectorai.export.export_mobile \
  --detector models/detector/best.pt \
  --classifier models/multihead_classifier/best.pt \
  --formats onnx,tflite,coreml \
  --out exports/mobile
```

### 7.2 Evaluation Metrics

Track:

| Metric | Purpose |
|---|---|
| mAP@50 / mAP@50-95 | Object detection quality |
| Family accuracy | Primary visual family |
| Attribute accuracy | gender, polarity, mount, orientation |
| Macro F1 | Handles class imbalance |
| Confusion matrix | Finds lookalike failures |
| Expected calibration error | Confidence reliability |
| Abstention accuracy | Correctness when confidence gating is applied |
| Latency | Mobile UX requirement |
| Model size | Mobile deployability |
| Battery/thermal notes | Field app usability |

### 7.3 Reports

Every experiment should write:

```text
reports/experiments/<timestamp>/
├── metrics.json
├── confusion_matrix_family.png
├── confusion_matrix_attributes.png
├── failure_gallery.html
├── calibration_curve.png
├── latency_report.md
└── model_card.md
```

---

## 8. App Plan

### 8.1 Flutter App

Keep Flutter as the cross-platform mobile app foundation. Extend the existing result panel to support:

- Live detection box.
- Connector family.
- Gender/polarity/mount/orientation attributes.
- Confidence and ambiguity warning.
- “Take another angle” workflow.
- “Known part/spec lookup” section.
- Manual correction chips.
- Contributor/label capture mode.
- Offline-first inference when mobile export is available.
- Server fallback mode when enabled.

### 8.2 Desktop App Options

For Windows/Linux/macOS:

1. Fastest path: Flutter desktop using the existing app.
2. Stronger ML/debug tooling: Python desktop diagnostic app with Streamlit or PySide.
3. Production client demo: Flutter desktop + FastAPI fallback.

Recommendation: first make Flutter desktop run using same API/result models. Do not build a separate native desktop app until the model pipeline is stable.

---

## 9. Serving/API Plan

### 9.1 Predict Endpoint

Upgrade `/predict` to return structured multi-head output:

```json
{
  "request_id": "uuid",
  "detected": true,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "connector_family": {"label": "SMA", "confidence": 0.94},
      "precision_family": {"label": "standard_sma", "confidence": 0.88},
      "gender": {"label": "female", "confidence": 0.91},
      "polarity": {"label": "standard", "confidence": 0.84},
      "mount_style": {"label": "bulkhead", "confidence": 0.77},
      "orientation": {"label": "straight", "confidence": 0.93},
      "spec": {},
      "warnings": []
    }
  ],
  "latency_ms": {
    "preprocess": 12,
    "detector": 31,
    "classifier": 18,
    "total": 74
  }
}
```

### 9.2 Compatibility

Keep old clients working by preserving the old top-level fields where possible:

```json
{
  "top_class": "sma_female",
  "confidence": 0.94
}
```

Add the richer output beside it.

---

## 10. Milestones

### Milestone 0 — Repo Audit and Baseline Reproduction

Deliverables:

- `docs/REPO_AUDIT.md`
- `docs/DATASET_AUDIT.md`
- baseline training/eval reproduced
- current model card captured
- clear list of gaps

Acceptance:

- Existing tests run or failures documented.
- Current 75% baseline reproduced or explained.
- Codex does not remove working app/server behavior.

### Milestone 1 — Taxonomy and Spec Database

Deliverables:

- `docs/CONNECTOR_TAXONOMY.md`
- `training/rfconnectorai/specs/connectors.yaml`
- schema validation tests

Acceptance:

- App/server can map predicted class/attributes to specs.
- Taxonomy supports SMA, RP-SMA, 3.5mm, 2.92mm, 2.4mm, 1.85mm, 1.0mm, SSMA, SMB, SMC, QMA, TNC, BNC, MCX, 7/16 DIN, unknown.

### Milestone 2 — Dataset Builder and Annotation Conversion

Deliverables:

- YOLO dataset exporter
- attributes CSV builder
- split-by-specimen support
- data quality report

Acceptance:

- No train/test specimen leakage.
- Dataset report prints class counts and missing attributes.
- Synthetic images are tracked separately from real images.

### Milestone 3 — Detector Training Track

Deliverables:

- YOLO/RT-DETR training scripts
- mAP report
- detector export path

Acceptance:

- Connector/background detection works on phone frames.
- No-connector/background examples are rejected reliably.
- Detector exports to ONNX/TFLite/CoreML where supported.

### Milestone 4 — Multi-Head Classifier

Deliverables:

- multi-head classifier model
- attribute loss weighting
- per-attribute metrics
- confusion/failure reports

Acceptance:

- Family/gender/polarity metrics improve over baseline.
- Model returns calibrated confidence and top alternatives.

### Milestone 5 — API Integration

Deliverables:

- structured `/predict` response
- old response compatibility
- spec lookup
- confidence warnings

Acceptance:

- Flutter still works.
- New structured fields are available to app.
- Unit tests cover known cases and no-connector cases.

### Milestone 6 — Flutter Result UX

Deliverables:

- richer result card
- warning states
- spec details
- correction chips
- capture/contribute workflow upgrade

Acceptance:

- Demo user can point camera and understand the result.
- Low-confidence cases ask for another angle instead of guessing.

### Milestone 7 — Mobile/Edge Deployment

Deliverables:

- ONNX/TFLite/CoreML export pipeline
- Flutter integration switch for local/server inference
- latency report

Acceptance:

- On-device inference works on at least one Android device.
- Server fallback remains available.
- Latency target is under 500 ms server path and preferably under 250 ms local path for small models.

### Milestone 8 — Client Demo Package

Deliverables:

- demo script
- install/run instructions
- model card
- limitations page
- before/after accuracy report
- short client-facing README

Acceptance:

- One-command local demo path.
- Clear evidence of improvement beyond current 75%.
- Honest limitations and next data capture plan.

---

## 11. Engineering Rules for Codex

1. Do not delete or rewrite the whole project unless necessary.
2. Preserve current app/server functionality.
3. Make small commits per milestone.
4. Add tests for every new parser, schema, API response, and dataset function.
5. Keep old `/predict` compatibility.
6. Use clear docstrings and type hints.
7. Prefer deterministic scripts over notebooks for repeatability.
8. Put generated reports under `reports/`, not random folders.
9. Do not train huge models by default in CI.
10. Add `--dry-run` flags to dataset conversion scripts.
11. Do not claim 99.99% accuracy unless validated on a statistically meaningful holdout.
12. Any model improvement must include confusion matrix and failure gallery.

---

## 12. Recommended File Additions

```text
IMPLEMENTATION_PLAN.md
TASKS.md
docs/
  CONNECTOR_TAXONOMY.md
  DATASET_AUDIT.md
  REPO_AUDIT.md
  MODEL_CARD_TEMPLATE.md
training/rfconnectorai/
  specs/
    connectors.yaml
  data/
    audit.py
    build_yolo_dataset.py
    split.py
  detector/
    train_yolo.py
    export.py
  classifier/
    train_multihead.py
    model_multihead.py
  eval/
    evaluate_all.py
    reports.py
  schemas/
    prediction.py
    taxonomy.py
reports/
  experiments/
exports/
  mobile/
datasets/
  rfconnectors/
```

---

## 13. Immediate Development Priority

The first sprint should not jump straight into architecture arguments. It should prove the ground truth:

1. Audit repo.
2. Audit dataset.
3. Freeze taxonomy.
4. Preserve current baseline.
5. Add detector dataset format.
6. Train a first detector.
7. Train a first multi-head classifier.
8. Compare against ResNet baseline.

Only after that should the team decide whether ResNet stays, EfficientNet/YOLO wins, or an ensemble is justified.
