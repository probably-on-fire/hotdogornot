# SMA Connector AI — Tasks Backlog

## Legend

- `P0` = required immediately
- `P1` = required for demo-quality implementation
- `P2` = production hardening
- `P3` = advanced/future enhancement

Status values:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked

---

## Epic 0 — Repository Audit and Safety Baseline

### P0 Tasks

- [ ] Create `docs/REPO_AUDIT.md`.
- [ ] Inventory root folders: `flutter/`, `training/`, `docs/`, `unity/`.
- [ ] Read and summarize:
  - [ ] root `README.md`
  - [ ] `training/README.md`
  - [ ] `training/docs/architecture.md`
  - [ ] `training/docs/classifier_journey.md`
  - [ ] `training/docs/runbook.md`
  - [ ] `training/docs/capture_protocol.md`
  - [ ] `flutter/README.md`
- [ ] Identify current API entry points.
- [ ] Identify current model load path.
- [ ] Identify current dataset paths.
- [ ] Identify current test suite.
- [ ] Run available Python tests:
  - [ ] `cd training && pytest`
- [ ] Run static import check for training package.
- [ ] Run Flutter dependency check:
  - [ ] `cd flutter && flutter pub get`
  - [ ] `cd flutter && flutter analyze`
- [ ] Document all failures without hiding them.
- [ ] Confirm current `/predict` response shape.
- [ ] Add a baseline smoke test if none exists.

### Acceptance Criteria

- [ ] Repo can be explained in one page.
- [ ] Existing breakages are documented.
- [ ] No existing behavior was removed.
- [ ] There is a known command to run the current server.
- [ ] There is a known command to run the current app.

---

## Epic 1 — Taxonomy and Connector Specification Model

### P0 Tasks

- [ ] Create `docs/CONNECTOR_TAXONOMY.md`.
- [ ] Define primary connector families:
  - [ ] SMA
  - [ ] RP-SMA
  - [ ] 3.5mm
  - [ ] 2.92mm / K / SMK
  - [ ] 2.4mm
  - [ ] 1.85mm / V
  - [ ] 1.0mm / W
  - [ ] SSMA
  - [ ] SMB
  - [ ] SMC
  - [ ] QMA
  - [ ] TNC
  - [ ] BNC
  - [ ] MCX
  - [ ] 7/16 DIN
  - [ ] unknown / unsupported
- [ ] Define attribute labels:
  - [ ] presence
  - [ ] family
  - [ ] precision family
  - [ ] gender/contact
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] cable termination
  - [ ] finish/material cue
  - [ ] size/geometry
  - [ ] confidence state
- [ ] Create `training/rfconnectorai/specs/connectors.yaml`.
- [ ] Add frequency range, impedance, coupling, compatibility, and visual notes where known.
- [ ] Add `unknown` and `not_applicable` values explicitly.
- [ ] Create `training/rfconnectorai/schemas/taxonomy.py`.
- [ ] Add unit tests for taxonomy loading and validation.

### Acceptance Criteria

- [ ] Taxonomy validates from YAML.
- [ ] Unknown/unsupported connector is a first-class outcome.
- [ ] Spec lookup is separate from model inference.
- [ ] Server can import the taxonomy without starting training code.

---

## Epic 2 — Dataset Audit

### P0 Tasks

- [ ] Create `training/rfconnectorai/data/audit.py`.
- [ ] Add CLI:
  - [ ] `python -m rfconnectorai.data.audit --data-dir data --out docs/DATASET_AUDIT.md`
- [ ] Count images by folder/class.
- [ ] Count train/val/test/holdout if present.
- [ ] Detect real vs synthetic if path metadata permits.
- [ ] Detect image dimensions and file types.
- [ ] Detect unreadable/corrupt images.
- [ ] Detect duplicate and near-duplicate images.
- [ ] Identify classes with fewer than target minimum samples.
- [ ] Identify missing taxonomy classes.
- [ ] Generate `docs/DATASET_AUDIT.md`.

### P1 Tasks

- [ ] Add blur score.
- [ ] Add brightness/contrast stats.
- [ ] Add background diversity approximation.
- [ ] Add per-class example contact sheet.
- [ ] Add leakage check by filename/specimen ID if metadata exists.

### Acceptance Criteria

- [ ] Dataset audit runs without training.
- [ ] Audit report is deterministic.
- [ ] Audit report clearly states why current held-out accuracy is not statistically meaningful if the holdout is tiny.
- [ ] No images are moved or modified by audit script.

---

## Epic 3 — Dataset Standardization and Annotation Conversion

### P0 Tasks

- [ ] Create standard dataset directory:
  - [ ] `datasets/rfconnectors/images/train`
  - [ ] `datasets/rfconnectors/images/val`
  - [ ] `datasets/rfconnectors/images/test`
  - [ ] `datasets/rfconnectors/labels/train`
  - [ ] `datasets/rfconnectors/labels/val`
  - [ ] `datasets/rfconnectors/labels/test`
- [ ] Create `datasets/rfconnectors/attributes.csv`.
- [ ] Create `datasets/rfconnectors/data.yaml`.
- [ ] Create `training/rfconnectorai/data/build_yolo_dataset.py`.
- [ ] Add `--dry-run`.
- [ ] Add split-by-specimen support.
- [ ] Add background/no-connector support.
- [ ] Add synthetic/real metadata field.
- [ ] Add validation for class names against taxonomy.

### P1 Tasks

- [ ] Add simple annotation helper for full-image labels when bounding boxes are unavailable.
- [ ] Add script to convert old classification folders into weak YOLO boxes.
- [ ] Add script to produce a labeling manifest for CVAT/Label Studio/Roboflow/manual annotation.

### Acceptance Criteria

- [ ] Dataset builder does not leak same specimen into train and holdout.
- [ ] Every exported image has corresponding metadata.
- [ ] Missing labels are reported clearly.
- [ ] Output can be consumed by YOLO training.

---

## Epic 4 — Detection Model Track

### P0 Tasks

- [ ] Add dependency documentation for Ultralytics or selected detector framework.
- [ ] Create `training/rfconnectorai/detector/train_yolo.py`.
- [ ] Support model selection:
  - [ ] `yolo11n.pt`
  - [ ] `yolo11s.pt`
  - [ ] future YOLO26 if installed/supported
- [ ] Add CLI args:
  - [ ] `--data`
  - [ ] `--model`
  - [ ] `--epochs`
  - [ ] `--imgsz`
  - [ ] `--batch`
  - [ ] `--device`
  - [ ] `--out`
- [ ] Save training run metadata.
- [ ] Save mAP metrics.
- [ ] Save detector model card.

### P1 Tasks

- [ ] Add RT-DETR-small experiment option if dependency is practical.
- [ ] Add detector failure gallery.
- [ ] Add export helper for ONNX/TFLite/CoreML where supported.
- [ ] Add no-connector rejection tests.

### Acceptance Criteria

- [ ] Detector can locate connector in a phone image.
- [ ] Background-only images are rejected.
- [ ] Model artifacts are saved predictably.
- [ ] Training script has a dry-run/config validation mode.

---

## Epic 5 — Multi-Head Attribute Classifier

### P0 Tasks

- [ ] Create `training/rfconnectorai/classifier/model_multihead.py`.
- [ ] Create `training/rfconnectorai/classifier/train_multihead.py`.
- [ ] Implement attribute heads:
  - [ ] family
  - [ ] gender/contact
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] termination
- [ ] Support backbones:
  - [ ] current ResNet-18 baseline
  - [ ] ResNet-50
  - [ ] EfficientNetV2 small or MobileNetV3
- [ ] Add weighted loss for missing or imbalanced attributes.
- [ ] Add top-k output.
- [ ] Add confidence calibration output.
- [ ] Save model card.

### P1 Tasks

- [ ] Add hard-negative mining.
- [ ] Add focal loss option.
- [ ] Add mixup/cutmix augmentation option.
- [ ] Add class-balanced sampler.
- [ ] Add multi-crop/test-time augmentation option.
- [ ] Add per-attribute confusion matrices.

### Acceptance Criteria

- [ ] Multi-head classifier trains on standardized dataset.
- [ ] Missing attributes do not crash training.
- [ ] Per-attribute metrics are reported.
- [ ] Baseline ResNet result is still available for comparison.

---

## Epic 6 — Evaluation and Reporting

### P0 Tasks

- [ ] Create `training/rfconnectorai/eval/evaluate_all.py`.
- [ ] Evaluate detector and classifier together.
- [ ] Produce:
  - [ ] `metrics.json`
  - [ ] family confusion matrix
  - [ ] attribute confusion matrices
  - [ ] failure gallery
  - [ ] latency report
  - [ ] model card
- [ ] Add abstention thresholds:
  - [ ] family confidence threshold
  - [ ] margin between top-1 and top-2
  - [ ] low image quality flag
  - [ ] multiple similar connector warning

### P1 Tasks

- [ ] Add calibration curve.
- [ ] Add expected calibration error.
- [ ] Add bootstrap confidence interval for accuracy.
- [ ] Add per-class precision/recall/F1.
- [ ] Add “demo readiness” scorecard.

### Acceptance Criteria

- [ ] Every experiment is comparable to prior runs.
- [ ] Report distinguishes forced-choice accuracy from abstention-aware correctness.
- [ ] Failure cases are visible, not hidden.
- [ ] No 99.99% claim is made without sufficient validation.

---

## Epic 7 — FastAPI Prediction Service Upgrade

### P0 Tasks

- [ ] Locate current FastAPI app.
- [ ] Preserve existing endpoint path and old response fields.
- [ ] Add structured prediction schema in `training/rfconnectorai/schemas/prediction.py`.
- [ ] Add output fields:
  - [ ] request ID
  - [ ] detected
  - [ ] bbox
  - [ ] connector family
  - [ ] precision family
  - [ ] gender/contact
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] termination
  - [ ] confidence state
  - [ ] warnings
  - [ ] spec lookup
  - [ ] latency
  - [ ] top alternatives
- [ ] Add no-connector response.
- [ ] Add ambiguous response.
- [ ] Add second-angle recommendation response.
- [ ] Add unit tests for response schema.

### P1 Tasks

- [ ] Add `/health`.
- [ ] Add `/version`.
- [ ] Add `/taxonomy`.
- [ ] Add `/specs/{connector_family}`.
- [ ] Add server-side image quality diagnostics.
- [ ] Add request logging without storing sensitive images by default.

### Acceptance Criteria

- [ ] Existing Flutter client does not break.
- [ ] New structured response is available.
- [ ] Unit tests cover old and new response compatibility.
- [ ] Latency is measured and returned.

---

## Epic 8 — Flutter App Upgrade

### P0 Tasks

- [ ] Locate existing identify screen.
- [ ] Preserve current camera flow.
- [ ] Update client data model to parse rich prediction response.
- [ ] Add richer result card:
  - [ ] family
  - [ ] gender
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] confidence
  - [ ] top alternatives
  - [ ] warnings
  - [ ] spec summary
- [ ] Add visual states:
  - [ ] high confidence
  - [ ] ambiguous
  - [ ] no connector
  - [ ] need another angle
  - [ ] unsupported connector
- [ ] Keep correction chips.
- [ ] Add capture/contribute metadata fields for taxonomy attributes.

### P1 Tasks

- [ ] Add bounding-box overlay on preview/result image.
- [ ] Add offline/server inference setting.
- [ ] Add app-side latency display in dev mode.
- [ ] Add “capture second angle” guided workflow.
- [ ] Add “known part number” field in contributor mode.
- [ ] Add export/share result as JSON.

### Acceptance Criteria

- [ ] User can identify connector from camera.
- [ ] Result is understandable to non-ML client.
- [ ] Low-confidence result does not look like a confident answer.
- [ ] Contributor mode improves future dataset quality.

---

## Epic 9 — Mobile/Desktop Deployment

### P0 Tasks

- [ ] Add export command for detector/classifier.
- [ ] Test ONNX export.
- [ ] Test TFLite/LiteRT export where supported.
- [ ] Test Core ML export where supported.
- [ ] Document export compatibility.
- [ ] Add `exports/mobile/README.md`.

### P1 Tasks

- [ ] Integrate local inference into Flutter Android first.
- [ ] Add server fallback.
- [ ] Add model version selection.
- [ ] Add latency benchmark on target device.
- [ ] Add Flutter desktop run path for Windows/Linux/macOS.

### Acceptance Criteria

- [ ] At least one local mobile inference path works.
- [ ] Server fallback remains stable.
- [ ] Desktop app can run or the limitation is documented.
- [ ] Exported model artifacts are versioned.

---

## Epic 10 — Client Demo Package

### P0 Tasks

- [ ] Create `docs/CLIENT_DEMO_README.md`.
- [ ] Create `docs/DEMO_SCRIPT.md`.
- [ ] Create `docs/LIMITATIONS_AND_NEXT_STEPS.md`.
- [ ] Create latest model card.
- [ ] Create before/after metrics table.
- [ ] Include screenshots or GIFs if available.
- [ ] Include exact commands:
  - [ ] run server
  - [ ] run Flutter app
  - [ ] run evaluation
  - [ ] export model

### Acceptance Criteria

- [ ] Demo can be run by someone besides the original developer.
- [ ] Contract client can understand what improved.
- [ ] Limitations are framed professionally.
- [ ] Next data collection plan is clear.

---

## Epic 11 — CI, Quality, and Project Hygiene

### P1 Tasks

- [ ] Add or update GitHub Actions for Python tests.
- [ ] Add lint/format commands.
- [ ] Add `ruff` or equivalent if not present.
- [ ] Add `mypy` selectively if feasible.
- [ ] Add Flutter analyze workflow.
- [ ] Add small fixture images for tests.
- [ ] Add `.gitignore` entries for:
  - [ ] datasets
  - [ ] reports
  - [ ] model weights
  - [ ] exports
  - [ ] local envs
- [ ] Add model artifact naming convention.

### Acceptance Criteria

- [ ] Tests run predictably.
- [ ] Large datasets/models are not accidentally committed.
- [ ] CI does not require GPU.
- [ ] Local GPU training remains documented.

---

## Epic 12 — Advanced Enhancements

### P2/P3 Tasks

- [ ] Add active learning loop:
  - [ ] collect low-confidence examples
  - [ ] route to labeling queue
  - [ ] retrain periodically
- [ ] Add calibrated measurement mode:
  - [ ] user places reference object
  - [ ] estimate diameter/body length/thread pitch
- [ ] Add segmentation model for precise connector outline.
- [ ] Add SAM/SAM2-assisted annotation pipeline.
- [ ] Add VLM/LLM explanation assistant.
- [ ] Add manufacturer part-number lookup.
- [ ] Add barcode/QR scan for labeled inventory workflows.
- [ ] Add AR overlay for connector dimensions.
- [ ] Add cloud dashboard for collected corrections.

### Acceptance Criteria

- [ ] Advanced features do not block core demo.
- [ ] Measurement mode clearly labels estimates vs confirmed specs.
- [ ] LLM/VLM does not override measured classifier confidence without evidence.

---

## First Codex Execution Batch

Give Codex this exact first batch:

```text
Start with IMPLEMENTATION_PLAN.md and TASKS.md as authoritative root planning docs.

Do not rewrite the whole app. First audit the repository and preserve all existing behavior.

Implement only Epic 0 and Epic 1 first:

1. Create docs/REPO_AUDIT.md by inspecting the repo structure, current README docs, current training pipeline, current FastAPI service, current Flutter app, and tests.
2. Create docs/CONNECTOR_TAXONOMY.md with the connector families and attributes listed in IMPLEMENTATION_PLAN.md.
3. Create training/rfconnectorai/specs/connectors.yaml with structured specs for SMA, RP-SMA, 3.5mm, 2.92mm/K/SMK, 2.4mm, 1.85mm, 1.0mm, SSMA, SMB, SMC, QMA, TNC, BNC, MCX, 7/16 DIN, unknown.
4. Add a lightweight taxonomy loader/validator in training/rfconnectorai/schemas/taxonomy.py.
5. Add tests for loading and validating the taxonomy YAML.
6. Run pytest for the training package if possible.
7. Run flutter analyze if Flutter is available.
8. Do not delete, rename, or break the existing predict endpoint or Flutter screens.
9. Commit changes in small logical units or show a patch summary if committing is unavailable.

Return:
- files changed
- commands run
- test results
- any blockers
- next recommended task batch
```

---

## Second Codex Execution Batch

```text
Continue from the completed repo audit and taxonomy.

Implement Epic 2 dataset audit:

1. Create training/rfconnectorai/data/audit.py.
2. Add a CLI runnable as:
   python -m rfconnectorai.data.audit --data-dir data --out docs/DATASET_AUDIT.md
3. The audit must count images by class/folder, detect unreadable files, summarize dimensions, detect duplicates by hash, and identify classes missing relative to the taxonomy.
4. Do not move or modify any image files.
5. Generate docs/DATASET_AUDIT.md.
6. Add tests using temporary fixture folders/images.
7. Run pytest.

Return:
- files changed
- generated audit summary
- commands run
- test results
- blockers
```

---

## Third Codex Execution Batch

```text
Continue from the taxonomy and dataset audit.

Implement Epic 3 dataset standardization:

1. Create training/rfconnectorai/data/build_yolo_dataset.py.
2. Add CLI:
   python -m rfconnectorai.data.build_yolo_dataset --input data/labeled --out datasets/rfconnectors --dry-run
3. Support export to:
   datasets/rfconnectors/images/train
   datasets/rfconnectors/images/val
   datasets/rfconnectors/images/test
   datasets/rfconnectors/labels/train
   datasets/rfconnectors/labels/val
   datasets/rfconnectors/labels/test
   datasets/rfconnectors/attributes.csv
   datasets/rfconnectors/data.yaml
4. If no bounding boxes exist, support weak full-image boxes and clearly mark them as weak labels.
5. Prevent train/test leakage by specimen_id if available.
6. Track real vs synthetic if detectable from path or metadata.
7. Add tests with toy fixture data.
8. Run pytest.

Return:
- files changed
- command examples
- test results
- known limitations
```

---

## Fourth Codex Execution Batch

```text
Continue from the standardized dataset output.

Implement the first detection training track:

1. Add training/rfconnectorai/detector/train_yolo.py.
2. Support Ultralytics YOLO model selection with yolo11n.pt and yolo11s.pt if available.
3. Add CLI args for data, model, epochs, imgsz, batch, device, and out.
4. Save run metadata and metrics under reports/experiments/<timestamp>.
5. Add export helper stub for ONNX/TFLite/CoreML but do not block training if export dependencies are missing.
6. Add documentation to training/README.md or docs/DETECTOR_TRAINING.md.
7. Add tests for CLI argument parsing and config validation.
8. Do not run expensive training by default in tests.

Return:
- files changed
- command examples
- test results
- how to run first detector training
```

---

## Fifth Codex Execution Batch

```text
Continue from detector training.

Implement the multi-head classifier path:

1. Create training/rfconnectorai/classifier/model_multihead.py.
2. Create training/rfconnectorai/classifier/train_multihead.py.
3. Use a configurable backbone initially supporting current ResNet-18 and one mobile-friendly option if already available.
4. Implement output heads for family, gender, polarity, mount style, orientation, and termination.
5. Support missing labels safely.
6. Report per-head accuracy and macro F1.
7. Save metrics and confusion matrix data under reports/experiments/<timestamp>.
8. Add tests for model forward pass and dataset label encoding.
9. Keep current ResNet classifier path intact.

Return:
- files changed
- command examples
- test results
- next integration steps
```

---

## Sixth Codex Execution Batch

```text
Continue from detector and multi-head classifier.

Upgrade the FastAPI prediction response without breaking old clients:

1. Add training/rfconnectorai/schemas/prediction.py.
2. Preserve current /predict endpoint path and old fields.
3. Add structured fields for detections, attributes, confidence, warnings, top alternatives, spec lookup, and latency.
4. Add no-connector and ambiguous result states.
5. Add tests for old response compatibility and new structured response.
6. Update server docs.

Return:
- files changed
- example JSON response
- test results
- any Flutter changes needed next
```

---

## Seventh Codex Execution Batch

```text
Continue from upgraded API response.

Upgrade the Flutter UI:

1. Parse the richer /predict response while preserving compatibility with old response fields.
2. Update result card to show family, gender, polarity, mount style, orientation, confidence, warnings, top alternatives, and spec summary.
3. Add visual states for high confidence, ambiguous, no connector, need another angle, and unsupported connector.
4. Keep existing identify/camera flow working.
5. Keep correction chips.
6. Add contributor metadata fields if practical.
7. Run flutter analyze.

Return:
- files changed
- UI behavior summary
- analyze/test results
- screenshots instructions if applicable
```

---

## Eighth Codex Execution Batch

```text
Prepare client demo package:

1. Create docs/CLIENT_DEMO_README.md.
2. Create docs/DEMO_SCRIPT.md.
3. Create docs/LIMITATIONS_AND_NEXT_STEPS.md.
4. Create docs/MODEL_CARD_TEMPLATE.md if not already present.
5. Add exact commands for server, app, training, eval, and export.
6. Include a before/after metrics table populated with available actual results; use TBD only where no result exists yet.
7. Make the limitations professional and confidence-building, not apologetic.

Return:
- files changed
- demo instructions
- remaining blockers before client presentation
```
