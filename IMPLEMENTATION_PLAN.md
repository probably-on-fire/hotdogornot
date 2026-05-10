# SMA Connector AI - Implementation Plan

## 1. Mission

Build a production-grade RF connector identification system that lets an end
user point a mobile or desktop camera at an SMA-family or related RF coaxial
connector and receive a reliable, engineering-useful result.

The product must identify:

1. Connector presence.
2. Connector family/type.
3. Standard vs reverse polarity.
4. Gender/contact configuration.
5. One-sided or two-sided adapter configuration.
6. Mount style and orientation.
7. Size and geometry cues when evidence permits.
8. Confidence, ambiguity, top alternatives, and next action.
9. Cross-referenced connector specifications.

This project is not a "hotdog/not-hotdog" classifier and is not a
ResNet-only project. ResNet-18 is the current baseline and fallback. The
target system is a multi-architecture vision pipeline:

```text
camera frame
  -> connector/background detector
  -> connector instance crop/mask
  -> multi-head attribute classifier
  -> geometry/measurement verification
  -> spec lookup
  -> optional 3D render verification
  -> confidence/abstention decision
  -> app result card
```

Heavy model training and architecture bake-offs should run in Kaggle, Colab,
or another cloud runtime after code is pushed to GitHub. The local PC is for
repo work, docs, schemas, tooling, and lightweight validation unless
explicitly approved.

## 2. Current Baseline

Current repository components:

```text
flutter/                 mobile Flutter app
training/                FastAPI + PyTorch training/serving stack
docs/                    architecture, taxonomy, plans, diagrams
unity/                   historical/sidelined Unity AR app
```

Current production behavior:

- Flutter app already exists.
- FastAPI `/predict` service already exists.
- Existing `/predict` JSON fields must remain compatible.
- Current classifier is ImageNet-pretrained ResNet-18 with a 6-class linear
  head.
- Current documented held-out performance is approximately:
  - full class: 75%
  - family: 75%
  - gender: 87.5%
  - background false positives: 0%
- Current held-out set is only 8 phone shots, which is not enough to prove
  true field accuracy.

Baseline rule:

```text
Keep ResNet-18 working for comparison and fallback.
Do not allow ResNet-18 to constrain the target architecture.
```

## 3. Core Product Outcome

Target API/app result:

```json
{
  "request_id": "uuid",
  "detected": true,
  "detections": [
    {
      "bbox": [120, 80, 420, 360],
      "family": {"label": "SMA", "confidence": 0.96},
      "precision_family": {"label": "standard_sma", "confidence": 0.91},
      "polarity": {"label": "standard", "confidence": 0.92},
      "side_a_gender": {"label": "male_pin", "confidence": 0.94},
      "side_b_gender": {"label": "female_socket", "confidence": 0.86},
      "mount_style": {"label": "adapter", "confidence": 0.90},
      "orientation": {"label": "right_angle", "confidence": 0.88},
      "termination": {"label": "not_applicable", "confidence": 0.83},
      "geometry": {
        "thread_diameter_mm": null,
        "thread_pitch_or_count": null,
        "body_length_mm": null,
        "requires_calibrated_reference": true
      },
      "confidence_state": "high_confidence",
      "warnings": [],
      "top_alternatives": [
        {"label": "RP-SMA right-angle adapter", "confidence": 0.41}
      ],
      "spec": {
        "impedance_ohms": 50,
        "frequency_range": "DC-18 GHz typical",
        "coupling": "threaded"
      }
    }
  ],
  "latency_ms": {
    "preprocess": 12,
    "detector": 31,
    "classifier": 18,
    "total": 74
  },
  "image_width": 1920,
  "image_height": 1080,
  "predictions": []
}
```

Compatibility rule:

```text
Existing Flutter clients must keep working.
Add richer fields beside old `/predict` fields, not instead of them.
```

## 4. Taxonomy

The taxonomy is authoritative in:

- `docs/CONNECTOR_TAXONOMY.md`
- `training/rfconnectorai/specs/connectors.yaml`
- `training/rfconnectorai/schemas/taxonomy.py`

Initial connector families:

| Family | Notes |
|---|---|
| SMA | Standard SMA, threaded, 50 ohm, commonly up to 18 GHz |
| RP-SMA | Reverse-polarity SMA; center contact gender inverted |
| 3.5mm | Precision SMA-compatible family |
| 2.92mm / K / SMK | Precision SMA/3.5mm-compatible family |
| 2.4mm | Higher precision; not SMA/3.5mm/2.92mm compatible |
| 1.85mm / V | Millimeter-wave precision family |
| 1.0mm / W | Very high frequency precision family |
| SSMA | Smaller SMA-related threaded connector |
| SMB | Snap-on lower-frequency connector |
| SMC | Threaded subminiature connector |
| QMA | Quick-lock SMA-like connector |
| TNC | Threaded RF connector |
| BNC | Bayonet RF connector |
| MCX | Small snap-on RF connector |
| 7/16 DIN | Large high-power threaded connector |
| unknown | Unsupported or insufficient evidence |

Primary attribute heads:

| Attribute | Values |
|---|---|
| presence | connector, no_connector, uncertain |
| family | taxonomy family list plus unknown |
| precision_family | standard_sma, rp_sma, 3.5mm, 2.92mm_k_smk, 2.4mm, 1.85mm_v, 1.0mm_w, not_applicable, unknown |
| side_a_gender/contact | male_pin, female_socket, RP variants, unknown |
| side_b_gender/contact | male_pin, female_socket, RP variants, not_applicable, unknown |
| polarity | standard, reverse_polarity, not_applicable, unknown |
| mount_style | cable_mount, panel_mount, bulkhead, PCB styles, adapter, terminator, unknown |
| orientation | straight, right_angle, tee, adapter_stack, unknown |
| termination | solder, crimp, clamp, molded_cable, not_applicable, unknown |
| finish/material | gold, nickel_silver, black_body, mixed, unknown |
| size/geometry | thread diameter, pitch/count, body length, aperture, hex size, requires reference |
| confidence_state | high_confidence, ambiguous, insufficient_view, need_second_angle, need_scale_reference, unsupported_connector, no_connector_detected |

Two-sided adapters must be represented explicitly. Folder names alone are not
enough for this project.

## 5. Target Architecture

### 5.1 Current Path To Preserve

```text
image
  -> OpenCV decode
  -> Hough crop detector
  -> rembg foreground gate
  -> optional clean crop
  -> ResNet-18 classifier
  -> old compatible /predict fields
```

This path remains useful for:

- regression comparison,
- baseline reproduction,
- fallback behavior,
- avoiding breakage while new systems are built.

### 5.2 Target Multi-Architecture Path

```text
image/video frame
  -> object detector
  -> per-connector instance crop/mask
  -> multi-head attribute classifier
  -> geometry/measurement module
  -> taxonomy/spec constraints
  -> optional 3D render verification
  -> confidence/abstention gate
  -> structured prediction response
```

### 5.3 Model Roles

| Role | Candidates | Purpose |
|---|---|---|
| Detector | YOLO11n, YOLO11s, RT-DETR small, MobileNet-SSD fallback | Locate connector instances, separate multi-connector images, reject background |
| Crop classifier | EfficientNetV2-S, MobileNetV3, MobileViT, ConvNeXt-Tiny, ResNet baseline | Fine-grained family/attribute classification |
| Multi-head model | custom PyTorch heads | Predict family, polarity, gender/contact, mount, orientation, termination |
| Geometry module | OpenCV/measurement utilities | Thread, aperture, hex, scale-reference reasoning |
| Spec verifier | YAML taxonomy/spec lookup | Apply compatibility and impossibility constraints |
| 3D verifier | parametric render/silhouette/edge matching | Second-pass verification for top candidates |
| Server deployment | FastAPI + PyTorch/ONNX Runtime | Highest accuracy fallback path |
| Mobile deployment | ONNX Runtime Mobile, TFLite/LiteRT, Core ML | On-device or edge inference |
| LLM/VLM assist | Gemma/Gemini-style language layer | Explanation/spec guidance only, not primary detection |

Promotion rule:

```text
Pick models by measured accuracy, reliability, latency, model size,
exportability, and failure behavior. Do not pick by architecture hype.
```

## 6. Dataset and Labeling Plan

### 6.1 Dataset Audit

Audit without modifying images:

- `training/Images/`
- `training/data/labeled/`
- `training/data/test_holdout/`
- `training/data/reference/`
- `training/data/videos/`
- future upload/incoming roots

Audit output:

```text
docs/DATASET_AUDIT.md
```

Required audit fields:

- image counts by folder/class,
- dimensions/file type,
- unreadable/corrupt files,
- duplicate hashes,
- near-duplicate risk if practical,
- current labels inferred from folder names,
- missing taxonomy classes,
- likely multi-connector images,
- tiny/blurry/low-quality images,
- leakage risk between train/val/test/holdout.

### 6.2 Connector Instance Catalog

Multi-connector images must be split into single-connector instances while
preserving originals.

Target manifest:

```text
datasets/rfconnectors/instances.jsonl
```

Each row represents one connector instance:

```json
{
  "instance_id": "stable-id",
  "source_image": "training/Images/example.webp",
  "crop_path": "datasets/rfconnectors/crops/example_0001.jpg",
  "bbox_xyxy": [120, 80, 420, 360],
  "label_confidence": "human_verified",
  "source_type": "real_photo",
  "family": "SMA",
  "precision_family": "standard_sma",
  "side_a_gender": "male_pin",
  "side_b_gender": "female_socket",
  "polarity": "standard",
  "mount_style": "adapter",
  "orientation": "straight",
  "termination": "not_applicable",
  "geometry": {
    "thread_diameter_mm": null,
    "thread_pitch_or_count": null,
    "body_length_mm": null,
    "hex_size_mm": null,
    "aperture_mm": null,
    "requires_calibrated_reference": true
  }
}
```

### 6.3 Dataset Format

Standard dataset root:

```text
datasets/rfconnectors/
  crops/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  attributes.csv
  instances.jsonl
  data.yaml
```

Rules:

- originals stay untouched,
- every crop tracks source lineage,
- labels validate against taxonomy,
- synthetic images are flagged,
- split by specimen/source group where possible,
- holdout is never used for training,
- no train/test leakage through duplicate or near-duplicate crops.

## 7. 3D and Synthetic Data Plan

### 7.1 Parametric 3D Suite

Build a parametric 3D suite for:

- SMA male/female straight,
- RP-SMA male/female,
- right-angle adapters,
- tee/splitter adapters,
- bulkhead/panel mount,
- cable/crimp/solder connectors,
- SMA-to-SMA adapters,
- SMA-to-BNC/TNC/MCX/UHF/N adapters,
- 3.5mm and 2.92mm/K/SMK lookalikes,
- 2.4mm/1.85mm/1.0mm precision families,
- confusing negatives.

Model parameters:

- family,
- polarity,
- side A/B gender/contact,
- thread diameter,
- thread pitch/count,
- body length,
- hex size,
- aperture,
- orientation,
- mount style,
- material/finish.

### 7.2 Synthetic Rendering

Render variations:

- front/side/top/45-degree/oblique views,
- macro and non-macro camera distances,
- focal length variation,
- lighting/background variation,
- blur/noise/compression,
- occlusion,
- scale marker/no scale marker,
- single-connector and multi-connector scenes,
- confusing lookalikes.

Synthetic labels must include:

- bbox,
- mask if available,
- attributes,
- geometry,
- render seed,
- camera pose,
- source model ID.

Synthetic data supports training and robustness; it does not replace a real
phone-photo holdout.

## 8. Training Pipeline

### 8.1 Required CLIs

```bash
python -m rfconnectorai.data.audit \
  --data-dir data \
  --out docs/DATASET_AUDIT.md

python -m rfconnectorai.data.crop_instances \
  --input training/Images \
  --manifest datasets/rfconnectors/instances.jsonl \
  --out datasets/rfconnectors/crops \
  --dry-run

python -m rfconnectorai.data.build_yolo_dataset \
  --input datasets/rfconnectors/instances.jsonl \
  --out datasets/rfconnectors \
  --dry-run

python -m rfconnectorai.detector.train_yolo \
  --data datasets/rfconnectors/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --out models/detector

python -m rfconnectorai.classifier.train_multihead \
  --dataset datasets/rfconnectors \
  --backbone efficientnet_v2_s \
  --epochs 80 \
  --out models/multihead_classifier

python -m rfconnectorai.synthetic.render_suite \
  --spec training/rfconnectorai/specs/connectors.yaml \
  --out datasets/rfconnectors_synthetic \
  --per-model 500

python -m rfconnectorai.eval.evaluate_all \
  --detector models/detector/best.pt \
  --classifier models/multihead_classifier/best.pt \
  --holdout datasets/rfconnectors/images/test \
  --out reports/experiments/latest

python -m rfconnectorai.export.export_mobile \
  --detector models/detector/best.pt \
  --classifier models/multihead_classifier/best.pt \
  --formats onnx,tflite,coreml \
  --out exports/mobile
```

### 8.2 Cloud Training Rule

Do not run expensive detector/classifier bake-offs on the local PC.

Workflow:

```text
commit scripts/docs
push to GitHub
pull in Kaggle/Colab
run training/evaluation
save metrics/model cards/failure galleries
bring results back into reports/experiments/
```

### 8.3 Evaluation Metrics

Track:

- mAP@50 and mAP@50-95,
- background false positive rate,
- family accuracy,
- polarity accuracy,
- side A/B gender/contact accuracy,
- mount/orientation/termination accuracy,
- macro F1,
- top-k accuracy,
- confusion matrices,
- calibration error,
- abstention-aware correctness,
- latency,
- model size,
- export compatibility,
- mobile thermal/battery notes if available.

### 8.4 Reports

Every experiment writes:

```text
reports/experiments/<timestamp>/
  metrics.json
  confusion_matrix_family.png
  confusion_matrix_attributes.png
  failure_gallery.html
  calibration_curve.png
  latency_report.md
  model_card.md
  config.yaml
```

## 9. Accuracy Strategy

99.99% is an aspiration, not the first acceptance gate. True claims require
large, diverse, independently held-out validation.

| Gate | Target | Meaning |
|---|---:|---|
| G0 | baseline documented | current 75% reproduced or explained |
| G1 | 90%+ family accuracy | credible improvement |
| G2 | 95%+ family/gender/polarity | demo-worthy |
| G3 | 98%+ curated holdout | strong candidate |
| G4 | 99%+ large real-phone holdout | production-grade |
| G5 | 99.9%+ with abstention | correct when confident, abstain when not |
| G6 | 99.99% | requires large field validation |

Forced wrong guesses are unacceptable. The app should ask for another angle
or a scale reference when needed.

## 10. API Plan

Preserve old fields:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "predictions": []
}
```

Add new fields beside old fields:

- `request_id`,
- `detected`,
- `detections`,
- bbox,
- all attribute heads,
- geometry,
- spec lookup,
- confidence state,
- warnings,
- top alternatives,
- latency.

Add future endpoints:

- `/health`,
- `/version`,
- `/taxonomy`,
- `/specs/{connector_family}`.

## 11. Flutter/App Plan

Keep the current camera flow and screens. Add:

- bounding box overlay,
- richer result card,
- confidence/ambiguity visual states,
- second-angle capture prompt,
- scale-reference prompt,
- spec summary,
- top alternatives,
- correction chips for taxonomy attributes,
- contributor metadata fields,
- server/local inference setting when mobile export exists.

## 12. Deployment Plan

Server path:

- FastAPI,
- PyTorch or ONNX Runtime,
- highest-accuracy fallback,
- compatibility with existing Flutter client.

Mobile/edge path:

- ONNX Runtime Mobile,
- TFLite/LiteRT,
- Core ML,
- latency and model-size constraints,
- server fallback remains available.

Desktop path:

- Flutter desktop first,
- optional Streamlit/Python diagnostic UI for ML debugging,
- no separate native desktop app until model pipeline stabilizes.

## 13. Milestones

1. Repo audit and safety baseline.
2. Taxonomy/spec database.
3. Dataset audit.
4. Connector instance catalog and crop manifest.
5. Standard dataset builder.
6. Detector training track.
7. Multi-head classifier track.
8. 3D model suite and synthetic render generation.
9. Evaluation and reporting harness.
10. Structured `/predict` response.
11. Flutter result UX upgrade.
12. Mobile/server export and deployment.
13. Client demo package.

## 14. Engineering Rules

1. Preserve existing app/server behavior.
2. Keep old `/predict` compatibility.
3. Keep ResNet-18 as baseline/fallback.
4. Do not train heavy models locally.
5. Add tests for parsers, schemas, dataset utilities, and API schemas.
6. Add dry-run/config-validation modes for data and training commands.
7. Keep generated reports under `reports/`.
8. Keep datasets/models/exports out of commits unless intentionally small docs/artifacts.
9. Track every crop back to its source image.
10. Split by specimen/source group to avoid leakage.
11. Never claim 99.99% without statistically meaningful validation.
12. Prefer abstention over confident wrong answers.
13. Use LLM/VLM only for explanation/spec guidance, not primary detection.

## 15. Recommended File Additions

```text
docs/
  DATASET_AUDIT.md
  DETECTOR_TRAINING.md
  MULTIHEAD_CLASSIFIER.md
  MODEL_CARD_TEMPLATE.md
  CLIENT_DEMO_README.md
  DEMO_SCRIPT.md
  LIMITATIONS_AND_NEXT_STEPS.md

training/rfconnectorai/
  data/
    audit.py
    crop_instances.py
    build_yolo_dataset.py
    split.py
  detector/
    __init__.py
    train_yolo.py
    export.py
  classifier/
    model_multihead.py
    train_multihead.py
    label_encoding.py
  eval/
    __init__.py
    evaluate_all.py
    reports.py
  synthetic/
    render_suite.py
    model_catalog.py
  export/
    export_mobile.py
  schemas/
    prediction.py

datasets/
  rfconnectors/

reports/
  experiments/

exports/
  mobile/
```
