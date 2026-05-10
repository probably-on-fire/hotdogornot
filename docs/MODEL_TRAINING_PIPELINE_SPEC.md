# Model Training Pipeline Design Spec

This spec defines the target training pipeline for extraordinary-accuracy
SMA/RF connector identification. It expands the roadmap in
`IMPLEMENTATION_PLAN.md` into a concrete data, labeling, synthetic rendering,
training, and verification strategy.

The core principle: preserve the existing app and `/predict` service while
building a much stronger model pipeline underneath it.

## 1. Goal

Build a connector-specific object detection and classification pipeline that
can inspect a camera image, identify each visible SMA/RF connector instance,
infer its visible attributes, cross-check against known geometry/specs, and
return a reliable result or a clear abstention state.

Target user outcome:

```text
User points camera at connector
-> app finds the connector
-> app reports type, polarity, gender/contact, shape/orientation, size cues,
   confidence, warnings, alternatives, and spec lookup
```

The system should not force a guess when visual evidence is insufficient.
For client-grade reliability, `need_second_angle`, `need_scale_reference`,
`ambiguous`, `unsupported_connector`, and `no_connector_detected` are valid
outcomes.

## 2. Recommended Architecture

Use 3D models and synthetic rendering to improve training and verification,
but do not make brute-force 3D pose matching the only primary inference
method. The primary production path should be staged computer vision:

```text
real image
  -> detect each connector instance
  -> crop each connector
  -> classify family and attributes
  -> compare against specs and geometry constraints
  -> optionally run 3D render matching as a verification/refinement layer
  -> return result or abstain/request another view
```

Why this order:

- object detection handles multi-connector images,
- crops reduce background bias,
- attribute heads are more robust than one flat class label,
- geometry/spec constraints catch impossible predictions,
- 3D render matching is useful as a second-pass verifier,
- abstention prevents confident wrong answers.

## 3. Pipeline Phases

### Phase 1: Dataset Audit

Inventory every current image source without modifying originals:

- `training/Images/`
- `training/data/labeled/`
- `training/data/test_holdout/`
- `training/data/reference/`
- `training/data/videos/`
- any future upload/incoming folders

The audit must report:

- image count by folder/class,
- file type and dimensions,
- unreadable/corrupt files,
- duplicate files by hash,
- likely near-duplicates where practical,
- current labels inferred from folder names,
- missing taxonomy classes,
- images likely containing multiple connectors,
- images too small/blurry/low quality for training,
- train/holdout leakage risks.

Deliverable:

```text
docs/DATASET_AUDIT.md
```

### Phase 2: Connector Instance Catalog

Images with multiple connectors should be split into individual connector
instances. Originals must stay untouched. Each detected or manually cropped
connector becomes a catalog item.

Recommended manifest:

```text
datasets/rfconnectors/instances.jsonl
```

Each row should represent one connector instance:

```json
{
  "instance_id": "uuid-or-stable-id",
  "source_image": "training/Images/example.webp",
  "crop_path": "datasets/rfconnectors/crops/sma_adapter_0001.jpg",
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
  "finish_material_cue": "gold",
  "geometry": {
    "thread_diameter_mm": null,
    "thread_pitch_or_count": null,
    "body_length_mm": null,
    "hex_size_mm": null,
    "aperture_mm": null,
    "requires_calibrated_reference": true
  },
  "notes": "SMA male to SMA female adapter, product image"
}
```

Key requirements:

- One physical connector instance per row.
- Preserve source-image lineage.
- Track whether label is human-verified, weak, auto-generated, or unknown.
- Support two-sided adapters, not just one mating face.
- Support unknown fields without crashing training.

### Phase 3: Attribute Taxonomy

Use `training/rfconnectorai/specs/connectors.yaml` and
`docs/CONNECTOR_TAXONOMY.md` as the label authority.

Core label groups:

```text
presence:
  connector / no_connector / uncertain

family:
  SMA / RP-SMA / 3.5mm / 2.92mm / 2.4mm / 1.85mm / 1.0mm /
  SSMA / SMB / SMC / QMA / TNC / BNC / MCX / 7/16 DIN / unknown

gender/contact:
  male_pin / female_socket /
  rp_male_body_female_contact /
  rp_female_body_male_contact / unknown

polarity:
  standard / reverse_polarity / not_applicable / unknown

mount_style:
  cable_mount / panel_mount / bulkhead /
  pcb_through_hole / pcb_edge_mount / pcb_surface_mount /
  adapter / terminator / unknown

orientation:
  straight / right_angle / tee / adapter_stack / unknown

termination:
  solder / crimp / clamp / molded_cable / not_applicable / unknown
```

For adapters, capture both sides where visible:

```text
side_a_family
side_a_gender
side_a_polarity
side_b_family
side_b_gender
side_b_polarity
```

This matters for examples such as:

- SMA male to SMA female,
- RP-SMA male to SMA female,
- SMA to BNC,
- SMA to MCX,
- SMA right-angle adapters,
- tee/splitter adapters.

### Phase 4: Crop and Label Workflow

Use a non-destructive crop workflow:

```text
source image
  -> detector/manual crop candidates
  -> one crop per connector instance
  -> human verification/correction
  -> instance manifest
  -> standardized training dataset
```

Required outputs:

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

- Do not split near-duplicate crops across train/val/test.
- Split by specimen or source group where possible.
- Keep holdout physically and logically separate.
- Do not use generated synthetic images in the final real-photo holdout.
- Multi-connector source images can produce many crops, but their lineage
  must be tracked to avoid leakage.

### Phase 5: 3D Connector Model Suite

After the instance catalog is established, construct a parametric 3D model
suite for supported connector types.

Initial targets:

- SMA male straight
- SMA female straight
- RP-SMA male
- RP-SMA female
- SMA right-angle adapter
- SMA tee/splitter adapter
- SMA bulkhead/panel mount
- SMA cable/crimp/solder connector
- SMA-to-SMA adapter variants
- SMA-to-BNC, SMA-to-TNC, SMA-to-MCX, SMA-to-UHF, SMA-to-N adapter variants
- 2.92mm/K/SMK precision lookalikes
- 3.5mm precision lookalikes
- 2.4mm, 1.85mm, and 1.0mm precision families
- confusing non-SMA negatives

Each model should be parameterized by:

```text
family
polarity
side_a_gender
side_b_gender
thread diameter
thread pitch/count
body length
hex size
aperture diameter
orientation
mount style
material/finish
```

Possible implementation locations:

```text
training/rfconnectorai/synthetic/procedural_connectors.py
training/rfconnectorai/synthetic/render.py
training/rfconnectorai/synthetic/angled_renderer.py
```

### Phase 6: Synthetic Rendering

Render every model under broad camera/environment variation:

- front, side, top, 45-degree, oblique, partial views,
- macro and non-macro distances,
- multiple focal lengths,
- multiple resolutions,
- lighting variation,
- background variation,
- blur/noise/compression,
- partial occlusion,
- different finishes/material cues,
- with and without scale reference,
- with multiple connectors in the frame,
- with confusing adapters nearby.

Every synthetic image must have perfect labels:

```text
bbox
mask if available
family
gender/contact
polarity
mount style
orientation
termination
geometry values
camera pose
render seed
source model id
```

Synthetic data should not replace real photos. It should:

- fill angle/lighting gaps,
- improve detector robustness,
- teach rare connector variants,
- provide exact geometry labels,
- create hard negatives and lookalikes.

### Phase 7: Training Tracks

Train in stages and compare each stage against the current ResNet baseline.

#### 7.1 Detector

Purpose:

- find connector instances,
- separate multiple connectors in one image,
- reject no-connector/background images.

Candidate models:

- YOLO11n/s,
- RT-DETR small,
- MobileNet-SSD fallback for edge cases.

Metrics:

- mAP@50,
- mAP@50-95,
- false positive rate on backgrounds,
- missed detections,
- latency.

#### 7.2 Multi-Head Attribute Classifier

Purpose:

- classify each cropped connector instance by attributes.

Heads:

- family,
- precision family,
- side A gender/contact,
- side B gender/contact,
- polarity,
- mount style,
- orientation,
- termination,
- finish/material cue,
- confidence state.

Backbones:

- ResNet-18 baseline,
- ResNet-50 only if data supports it,
- EfficientNetV2 small,
- MobileNetV3/MobileViT for deployment candidates.

Metrics:

- per-head accuracy,
- macro F1,
- confusion matrix,
- top-k accuracy,
- calibration error,
- abstention-aware correctness.

#### 7.3 Measurement/Geometry Module

Purpose:

- estimate measurable geometry when scale is available,
- reject geometry-impossible classifications,
- separate visually similar precision families where possible.

Inputs:

- ArUco marker,
- ruler/caliper/reference object,
- known printed scale card,
- thread pitch/edge cues,
- aperture/hex/body measurements.

Outputs:

- diameter estimate,
- thread pitch/count estimate,
- body length estimate,
- aperture estimate,
- measurement confidence,
- `need_scale_reference` when unavailable.

### Phase 8: 3D Verification Layer

Use 3D render matching as a second-pass verifier, not the only detector.

Workflow:

```text
model predicts top candidates
  -> choose candidate 3D models
  -> render nearby poses
  -> compare silhouette, edges, thread region, connector ends
  -> adjust confidence or request another angle
```

Good uses:

- distinguish SMA vs RP-SMA when center contact is visible,
- verify right-angle vs straight adapter,
- compare thread/body proportions,
- detect when the view is insufficient,
- identify when a second side/angle is required.

Avoid:

- brute-force matching against every model on every frame,
- claiming exact physical dimensions without scale,
- overriding low-quality visual evidence with a confident synthetic match.

### Phase 9: Evaluation Gates

Accuracy must be measured honestly.

Recommended gates:

| Gate | Target | Meaning |
|---|---:|---|
| G0 | current baseline documented | reproduce or explain 75% baseline |
| G1 | 90%+ family accuracy | credible improvement |
| G2 | 95%+ family/gender/polarity | demo-worthy |
| G3 | 98%+ curated holdout | strong candidate |
| G4 | 99%+ large real-phone holdout | production-grade |
| G5 | 99.9%+ with abstention | correct when confident, abstain when not |
| G6 | 99.99% aspiration | requires large field validation |

Do not claim 99.99% unless validated on a statistically meaningful,
independently held-out dataset.

Required reports per experiment:

```text
reports/experiments/<timestamp>/
  metrics.json
  confusion_matrix_family.png
  confusion_matrix_attributes.png
  failure_gallery.html
  calibration_curve.png
  latency_report.md
  model_card.md
```

### Phase 10: App/API Integration

Preserve existing `/predict` behavior while adding richer fields.

Current compatible fields must remain:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "predictions": []
}
```

Future structured fields should be added beside them:

```json
{
  "request_id": "uuid",
  "detected": true,
  "detections": [
    {
      "bbox": [120, 80, 420, 360],
      "family": {"label": "SMA", "confidence": 0.96},
      "polarity": {"label": "standard", "confidence": 0.91},
      "side_a_gender": {"label": "male_pin", "confidence": 0.94},
      "side_b_gender": {"label": "female_socket", "confidence": 0.88},
      "orientation": {"label": "right_angle", "confidence": 0.84},
      "mount_style": {"label": "adapter", "confidence": 0.87},
      "geometry": {
        "thread_diameter_mm": null,
        "thread_pitch_or_count": null,
        "requires_calibrated_reference": true
      },
      "confidence_state": "high_confidence",
      "warnings": [],
      "top_alternatives": [],
      "spec": {}
    }
  ]
}
```

Flutter should eventually show:

- connector family,
- gender/contact,
- polarity,
- orientation,
- mount style,
- confidence,
- warnings,
- top alternatives,
- spec summary,
- correction chips,
- second-angle prompt when needed.

## 4. Implementation Order

Recommended next batches:

1. Dataset audit.
2. Multi-connector image inventory.
3. Instance/crop manifest design.
4. Non-destructive crop extraction tooling.
5. Human verification workflow.
6. Standard YOLO dataset builder.
7. First detector training.
8. Multi-head classifier.
9. Synthetic 3D model suite.
10. Synthetic render generation.
11. 3D verification layer.
12. Structured `/predict` response.
13. Flutter richer result UI.

## 5. Non-Negotiable Rules

- Keep originals untouched.
- Track source lineage for every crop.
- Split train/val/test by specimen/source group, not random image only.
- Keep real holdout separate from synthetic data.
- Preserve current `/predict` endpoint behavior.
- Preserve current Flutter screens while adding capability.
- Treat unknown/ambiguous as valid outputs.
- Use 3D rendering to improve and verify, not to hide poor labels.
- Do not claim impossible accuracy without statistically meaningful proof.
