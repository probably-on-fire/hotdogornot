# SMA Connector AI - Development Task Blueprint

This file is the execution checklist from the current repository state to the
target multi-architecture RF connector identification system.

Core rules:

- Preserve existing Flutter and FastAPI behavior.
- Preserve `/predict` compatibility.
- Treat ResNet-18 as baseline/fallback, not the final architecture.
- Build detector + multi-head classifier + geometry/spec verification.
- Run heavy model training and bake-offs in Kaggle/Colab/cloud, not on the
  local PC.
- Keep local and GitHub `master` synced after each completed batch.

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

## Epic 0 - Repository Audit and Safety Baseline

### P0 Tasks

- [x] Create `docs/REPO_AUDIT.md`.
- [x] Inventory `flutter/`, `training/`, `docs/`, `unity/`.
- [x] Read current README/training/Flutter architecture docs.
- [x] Identify current FastAPI entry points.
- [x] Identify current model load path.
- [x] Identify current dataset paths.
- [x] Identify current test suite.
- [x] Confirm current `/predict` response shape.
- [x] Document local test/tooling blockers.

### Remaining P0 Tasks

- [ ] Add/confirm a lightweight baseline smoke test for existing `/predict`
      compatibility if not already covered.
- [ ] In cloud or a properly provisioned Python 3.11 env, run the full
      training pytest suite and record results.
- [ ] In a Flutter-enabled environment, run `flutter analyze` and record
      results.

### Acceptance Criteria

- [x] Existing behavior was not removed.
- [x] Current server/app paths are documented.
- [x] Known local blockers are documented.
- [ ] Full tests are run in the correct environment or documented as blocked.

---

## Epic 1 - Taxonomy and Connector Specification Model

### P0 Tasks

- [x] Create `docs/CONNECTOR_TAXONOMY.md`.
- [x] Create `training/rfconnectorai/specs/connectors.yaml`.
- [x] Add SMA, RP-SMA, 3.5mm, 2.92mm/K/SMK, 2.4mm, 1.85mm, 1.0mm, SSMA,
      SMB, SMC, QMA, TNC, BNC, MCX, 7/16 DIN, and unknown.
- [x] Define attribute values for presence, family, precision family,
      gender/contact, polarity, mount, orientation, termination, finish,
      confidence state.
- [x] Create `training/rfconnectorai/schemas/taxonomy.py`.
- [x] Add unit tests for taxonomy loading and validation.

### P1 Tasks

- [ ] Add side-A/side-B adapter fields explicitly to taxonomy docs and schemas.
- [ ] Define nested `side_a` / `side_b` blocks (family, precision_family,
      gender, polarity, threaded, coupling) alongside the flat
      `side_a_gender` / `side_b_gender` fields used by simple consumers.
- [ ] Document the annotation protocol in `docs/ANNOTATION_PROTOCOL.md`,
      including adapter labeling rules and unknown/insufficient_view rules.
- [ ] Add examples for SMA-to-SMA, RP-SMA-to-SMA, SMA-to-BNC, SMA-to-MCX,
      right-angle, tee, and bulkhead adapters.
- [ ] Add a stable ID/name normalization helper for taxonomy labels.

### Acceptance Criteria

- [x] Taxonomy validates from YAML.
- [x] Unknown/unsupported is first-class.
- [x] Spec lookup is separate from model inference.
- [x] Server-safe taxonomy import exists.

---

## Epic 2 - Dataset Audit

### P0 Tasks

- [ ] Create `training/rfconnectorai/data/audit.py`.
- [ ] Add CLI:

  ```bash
  python -m rfconnectorai.data.audit --data-dir data --out docs/DATASET_AUDIT.md
  ```

- [ ] Support additional roots:
  - [ ] `training/Images`
  - [ ] `training/data/labeled`
  - [ ] `training/data/test_holdout`
  - [ ] `training/data/reference`
  - [ ] `training/data/videos`
- [ ] Count images by folder/class.
- [ ] Count videos/reference files separately.
- [ ] Detect image file types and dimensions.
- [ ] Detect unreadable/corrupt images.
- [ ] Detect duplicates by hash.
- [ ] Detect near-duplicate risk if practical.
- [ ] Detect real vs synthetic from path/name metadata where possible.
- [ ] Identify likely multi-connector images.
- [ ] Identify missing classes relative to taxonomy.
- [ ] Identify classes with fewer than minimum target samples.
- [ ] Identify train/holdout leakage risks.
- [ ] Generate `docs/DATASET_AUDIT.md`.

### P1 Tasks

- [ ] Add blur score.
- [ ] Add brightness/contrast stats.
- [ ] Add approximate background diversity.
- [ ] Add per-class contact sheet.
- [ ] Add specimen/source-group inference.
- [ ] Add JSON output beside markdown.

### Acceptance Criteria

- [ ] Audit does not modify or move images.
- [ ] Audit report is deterministic.
- [ ] Audit states why the current 8-image holdout is statistically weak.
- [ ] Audit identifies missing taxonomy families and multi-connector images.

---

## Epic 3 - Connector Instance Catalog and Crop Workflow

### P0 Tasks

- [x] Author `docs/ANNOTATION_PROTOCOL.md` before any human labeling begins
      so weak/strong labels are tagged consistently from day one.
- [x] Create shared geometry schema before writing instance manifest rows
      (`GeometryLabel` in `training/rfconnectorai/schemas/instance.py`).
- [ ] Reuse the same geometry schema in instance labels and `/predict`
      response payloads so crops, manifests, and predictions stay in sync.
- [x] Create `training/rfconnectorai/schemas/instance.py` with
      `ConnectorSide`, `GeometryLabel`, `ConnectorInstance`,
      `LabelConfidence`, and `SourceType` models/enums.
- [ ] Validate every row of `datasets/rfconnectors/instances.jsonl` through
      the instance schema before training.
- [ ] Create `training/rfconnectorai/data/crop_instances.py`.
- [ ] Define `datasets/rfconnectors/instances.jsonl`.
- [ ] Define one row per connector instance.
- [ ] Preserve source image path and bbox for every crop.
- [ ] Add fields:
  - [ ] `instance_id`
  - [ ] `source_image`
  - [ ] `crop_path`
  - [ ] `bbox_xyxy`
  - [ ] `label_confidence`
  - [ ] `source_type`
  - [ ] `family`
  - [ ] `precision_family`
  - [ ] `side_a_gender`
  - [ ] `side_b_gender`
  - [ ] `side_a` (nested: family, precision_family, gender, polarity, threaded, coupling)
  - [ ] `side_b` (nested: family, precision_family, gender, polarity, threaded, coupling)
  - [ ] `polarity`
  - [ ] `mount_style`
  - [ ] `orientation`
  - [ ] `termination`
  - [ ] `finish_material_cue`
  - [ ] `geometry`
- [ ] Add CLI:

  ```bash
  python -m rfconnectorai.data.crop_instances \
    --input training/Images \
    --manifest datasets/rfconnectors/instances.jsonl \
    --out datasets/rfconnectors/crops \
    --dry-run
  ```

- [ ] Support manual/full-image weak crop entries.
- [ ] Support detector-generated crop candidates.
- [ ] Do not modify original source images.

### P1 Tasks

- [ ] Add simple local review manifest for human correction.
- [ ] Add CVAT/Label Studio/Roboflow export manifest.
- [ ] Add import path for corrected labels.
- [ ] Add duplicate crop/source leakage detection.

### Acceptance Criteria

- [ ] Multi-connector images can produce multiple instance rows.
- [ ] Every crop traces back to a source image.
- [ ] Unknown/missing attributes are allowed and explicit.
- [ ] Weak labels are marked as weak.

---

## Epic 4 - Dataset Standardization and YOLO Conversion

### P0 Tasks

- [ ] Create `training/rfconnectorai/data/build_yolo_dataset.py`.
- [ ] Create output tree:
  - [ ] `datasets/rfconnectors/images/train`
  - [ ] `datasets/rfconnectors/images/val`
  - [ ] `datasets/rfconnectors/images/test`
  - [ ] `datasets/rfconnectors/labels/train`
  - [ ] `datasets/rfconnectors/labels/val`
  - [ ] `datasets/rfconnectors/labels/test`
- [ ] Create `datasets/rfconnectors/attributes.csv`.
- [ ] Create `datasets/rfconnectors/data.yaml`.
- [ ] Read from `instances.jsonl`.
- [ ] Validate labels against taxonomy.
- [ ] Support `--dry-run`.
- [ ] Support split-by-specimen/source group.
- [ ] Track synthetic vs real.
- [ ] Support background/no-connector examples.
- [ ] Support weak full-image boxes when true boxes are unavailable.

### P1 Tasks

- [ ] Add label quality report.
- [ ] Add class imbalance warning.
- [ ] Add missing-attribute warning.
- [ ] Add conversion tests with toy fixture data.

### Acceptance Criteria

- [ ] No same specimen/source group leaks across train/val/test.
- [ ] YOLO dataset can be consumed by detector training.
- [ ] Attribute metadata is available for multi-head training.
- [ ] Dataset builder never silently drops labels.

---

## Epic 5 - Detector Training Track

### P0 Tasks

- [ ] Add `training/rfconnectorai/detector/__init__.py`.
- [ ] Add `training/rfconnectorai/detector/train_yolo.py`.
- [ ] Support model choices:
  - [ ] `yolo11n.pt`
  - [ ] `yolo11s.pt`
  - [ ] future YOLO variants if installed/supported
- [ ] Add CLI args:
  - [ ] `--data`
  - [ ] `--model`
  - [ ] `--epochs`
  - [ ] `--imgsz`
  - [ ] `--batch`
  - [ ] `--device`
  - [ ] `--out`
  - [ ] `--dry-run`
- [ ] Save run metadata under `reports/experiments/<timestamp>/`.
- [ ] Save mAP metrics.
- [ ] Save detector model card.
- [ ] Write a `ModelRecord` via
      `training/rfconnectorai/models/registry.py` to
      `reports/experiments/<timestamp>/model_record.json`.
- [ ] Add config validation tests.
- [ ] Do not run expensive training in tests.
- [ ] Unit tests must validate model construction and config parsing only;
      tests must not train real models or require GPU.

### P1 Tasks

- [ ] Add RT-DETR small experiment option if dependency is practical.
- [ ] Add detector failure gallery.
- [ ] Add no-connector/background rejection metrics.
- [ ] Add detector export helper for ONNX/TFLite/CoreML where supported.
- [ ] Add cloud notebook/run instructions for Kaggle/Colab.

### Acceptance Criteria

- [ ] Detector locates connector instances in phone/product images.
- [ ] Multi-connector images produce multiple detections.
- [ ] Background-only images are rejected reliably.
- [ ] Detector run is reproducible from config.
- [ ] Heavy training is performed in cloud, not local PC.

---

## Epic 6 - Multi-Head Attribute Classifier

### P0 Tasks

- [ ] Keep current ResNet-18 path intact as baseline/fallback.
- [ ] Create `training/rfconnectorai/classifier/model_multihead.py`.
- [ ] Create `training/rfconnectorai/classifier/train_multihead.py`.
- [ ] Create `training/rfconnectorai/classifier/label_encoding.py`.
- [ ] Implement heads:
  - [ ] family
  - [ ] precision family
  - [ ] side A gender/contact
  - [ ] side B gender/contact
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] termination
  - [ ] finish/material cue
- [ ] Support missing labels safely.
- [ ] Support candidate backbones:
  - [ ] ResNet-18 baseline
  - [ ] ResNet-50 if useful
  - [ ] EfficientNetV2-S
  - [ ] MobileNetV3
  - [ ] MobileViT if practical
  - [ ] ConvNeXt-Tiny if practical
- [ ] Add weighted/masked losses for missing and imbalanced attributes.
- [ ] Add top-k output.
- [ ] Add confidence calibration output.
- [ ] Save metrics and model card.
- [ ] Write a `ModelRecord` via
      `training/rfconnectorai/models/registry.py` for every classifier
      run so dataset hash and taxonomy hash stay matched to artifacts.
- [ ] Add tests for label encoding and forward pass.
- [ ] Unit tests must validate model construction, label encoding, loss
      masking, and one tiny forward pass only.
- [ ] Tests must not train real models or require GPU.

### P1 Tasks

- [ ] Add focal loss option.
- [ ] Add mixup/cutmix option.
- [ ] Add class-balanced sampler.
- [ ] Add hard-negative mining.
- [ ] Add multi-crop/test-time augmentation.
- [ ] Add per-head confusion matrices.
- [ ] Add cloud bake-off scripts/configs.

### Acceptance Criteria

- [ ] Multi-head classifier trains on standardized dataset.
- [ ] Missing attributes do not crash training.
- [ ] Per-attribute metrics are reported.
- [ ] ResNet baseline remains comparable.
- [ ] Candidate architectures are compared in Kaggle/Colab/cloud.

---

## Epic 7 - 3D Model Suite and Synthetic Rendering

### P0 Tasks

- [ ] Create `training/rfconnectorai/synthetic/model_catalog.py`.
- [ ] Create `training/rfconnectorai/synthetic/render_suite.py`.
- [ ] Define parametric models for:
  - [ ] SMA male/female straight
  - [ ] RP-SMA male/female
  - [ ] right-angle adapters
  - [ ] tee/splitter adapters
  - [ ] bulkhead/panel mount
  - [ ] cable/crimp/solder connectors
  - [ ] SMA-to-SMA adapters
  - [ ] SMA-to-BNC/TNC/MCX/UHF/N adapters
  - [ ] 3.5mm and 2.92mm/K/SMK lookalikes
  - [ ] 2.4mm, 1.85mm, 1.0mm precision families
  - [ ] confusing negatives
- [ ] Render variations:
  - [ ] angle/pose
  - [ ] lighting
  - [ ] background
  - [ ] focal length
  - [ ] blur/noise/compression
  - [ ] occlusion
  - [ ] with/without scale reference
  - [ ] single and multi-connector scenes
- [ ] Emit perfect labels:
  - [ ] bbox
  - [ ] mask if available
  - [ ] attributes
  - [ ] geometry
  - [ ] render seed
  - [ ] camera pose
  - [ ] source model ID

### P1 Tasks

- [ ] Add synthetic-vs-real balancing controls.
- [ ] Add render quality audit.
- [ ] Add synthetic failure/lookalike set.
- [ ] Add cloud render/training notes.

### Acceptance Criteria

- [ ] Synthetic renders are traceable to model IDs.
- [ ] Synthetic images never contaminate real holdout.
- [ ] Synthetic data can augment detector and classifier training.
- [ ] Generated labels validate against taxonomy.

---

## Epic 8 - Geometry, Measurement, and 3D Verification

### P0 Tasks

- [ ] Define geometry schema for predictions.
- [ ] Integrate existing ArUco/hex/aperture/thread modules where useful.
- [ ] Add `need_scale_reference` state.
- [ ] Add geometry plausibility checks.
- [ ] Add spec compatibility checks.
- [ ] Create optional 3D render verification interface:
  - [ ] receive top candidate labels
  - [ ] render candidate poses
  - [ ] compare silhouette/edges/thread/body proportions
  - [ ] return confidence adjustment or second-angle request

### P1 Tasks

- [ ] Add calibrated measurement mode.
- [ ] Add thread count/pitch estimator.
- [ ] Add second-angle fusion.
- [ ] Add measurement tests with fixture images/renders.

### Acceptance Criteria

- [ ] Geometry does not claim exact size without scale evidence.
- [ ] Impossible/spec-incompatible predictions are downgraded.
- [ ] 3D verification is second-pass only, not the primary detector.

---

## Epic 9 - Evaluation and Reporting Harness

### P0 Tasks

- [ ] Create `training/rfconnectorai/eval/__init__.py`.
- [ ] Create `training/rfconnectorai/eval/evaluate_all.py`.
- [ ] Create `training/rfconnectorai/eval/reports.py`.
- [ ] Evaluate detector and classifier together.
- [ ] Report:
  - [ ] mAP@50
  - [ ] mAP@50-95
  - [ ] family accuracy
  - [ ] polarity accuracy
  - [ ] side A/B gender accuracy
  - [ ] mount/orientation/termination accuracy
  - [ ] macro F1
  - [ ] top-k accuracy
  - [ ] calibration error
  - [ ] abstention-aware correctness
  - [ ] latency
  - [ ] model size
- [ ] Produce:
  - [ ] `metrics.json`
  - [ ] confusion matrices
  - [ ] failure gallery
  - [ ] calibration curve
  - [ ] latency report
  - [ ] model card
  - [ ] config snapshot
  - [ ] `model_record.json` from
        `training/rfconnectorai/models/registry.py`
  - [ ] dataset lock reference to
        `datasets/rfconnectors/dataset.lock.json` so every report ties back
        to a specific dataset revision

### P1 Tasks

- [ ] Add bootstrap confidence intervals.
- [ ] Add demo-readiness scorecard.
- [ ] Add before/after comparison against ResNet-18 baseline.
- [ ] Add cloud-run summary template.

### Acceptance Criteria

- [ ] Every experiment is comparable to prior runs.
- [ ] Forced-choice and abstention-aware metrics are separated.
- [ ] Failure cases are visible.
- [ ] No 99.99% claim is made without sufficient validation.

---

## Epic 10 - FastAPI Prediction Service Upgrade

### P0 Tasks

API schema and tests must land before any FastAPI handler change. Order:

1. Add `training/rfconnectorai/schemas/prediction.py`.
2. Add response fixture tests for:
   - old-compatible response,
   - no-connector response,
   - ambiguous response,
   - multi-detection adapter (with nested `side_a` / `side_b` blocks),
   - `need_second_angle` response,
   - `need_scale_reference` response.
3. Only after the schema and fixture tests pass, wire the schema into the
   FastAPI predict handler.

- [ ] Preserve existing `/predict` endpoint path.
- [ ] Preserve old response fields.
- [ ] Add `training/rfconnectorai/schemas/prediction.py`.
- [ ] Reuse `GeometryLabel` (and side-aware blocks) from
      `training/rfconnectorai/schemas/instance.py` so labels and predictions
      share one geometry/side-aware schema.
- [ ] Add structured fields:
  - [ ] request ID
  - [ ] detected
  - [ ] detections
  - [ ] bbox
  - [ ] family
  - [ ] precision family
  - [ ] side A/B gender/contact (flat fields, kept for legacy clients)
  - [ ] nested `side_a` / `side_b` blocks (family, precision_family, gender, polarity, threaded, coupling)
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] termination
  - [ ] geometry
  - [ ] confidence state
  - [ ] warnings
  - [ ] spec lookup
  - [ ] latency
  - [ ] top alternatives
- [ ] Add no-connector response.
- [ ] Add ambiguous response.
- [ ] Add second-angle recommendation.
- [ ] Add response schema tests.
- [ ] Add old-client compatibility tests.

### P1 Tasks

- [ ] Add `/health`.
- [ ] Add `/version`.
- [ ] Add `/taxonomy`.
- [ ] Add `/specs/{connector_family}`.
- [ ] Add server-side image quality diagnostics.
- [ ] Add privacy-safe request logging.

### Acceptance Criteria

- [ ] Existing Flutter client does not break.
- [ ] New structured response is available.
- [ ] Unit tests cover old and new response compatibility.
- [ ] Latency is measured and returned.

---

## Epic 11 - Flutter App Upgrade

### P0 Tasks

- [ ] Preserve existing identify/camera flow.
- [ ] Preserve current screens.
- [ ] Update API client to parse rich response beside old fields.
- [ ] Add richer result card:
  - [ ] family
  - [ ] side A/B gender/contact
  - [ ] polarity
  - [ ] mount style
  - [ ] orientation
  - [ ] confidence
  - [ ] warnings
  - [ ] top alternatives
  - [ ] spec summary
- [ ] Add visual states:
  - [ ] high confidence
  - [ ] ambiguous
  - [ ] no connector
  - [ ] need another angle
  - [ ] need scale reference
  - [ ] unsupported connector
- [ ] Keep correction chips.
- [ ] Add contributor metadata fields for taxonomy attributes.

### P1 Tasks

- [ ] Add bounding-box overlay.
- [ ] Add offline/server inference setting.
- [ ] Add app-side latency display in dev mode.
- [ ] Add guided second-angle workflow.
- [ ] Add known part number field.
- [ ] Add export/share result as JSON.

### Acceptance Criteria

- [ ] User can identify connector from camera.
- [ ] Low-confidence result does not look confident.
- [ ] Contributor mode improves future dataset quality.
- [ ] Flutter analyze passes in a Flutter-enabled environment.

---

## Epic 12 - Mobile, Edge, and Server Deployment

### P0 Tasks

- [ ] Create `training/rfconnectorai/export/export_mobile.py`.
- [ ] Export detector/classifier to ONNX.
- [ ] Test ONNX Runtime path in cloud/dev environment.
- [ ] Test TFLite/LiteRT where supported.
- [ ] Test Core ML where supported.
- [ ] Document export compatibility.
- [ ] Add `exports/mobile/README.md`.
- [ ] Each exported artifact must reference a `ModelRecord` from
      `training/rfconnectorai/models/registry.py` so mobile/server clients
      can identify exactly which trained model produced the export.

### P1 Tasks

- [ ] Integrate Android local inference first.
- [ ] Add server fallback setting.
- [ ] Add model version selection.
- [ ] Add target-device latency benchmark.
- [ ] Add Flutter desktop run path.

### Acceptance Criteria

- [ ] At least one local mobile inference path works.
- [ ] Server fallback remains stable.
- [ ] Desktop app can run or limitation is documented.
- [ ] Exported model artifacts are versioned.

---

## Epic 13 - Client Demo Package

### P0 Tasks

- [x] Create `docs/ACCEPTANCE_GATES.md` so each batch and each demo step
      maps to a concrete gate.
- [ ] Create `docs/CLIENT_DEMO_README.md`.
- [ ] Create `docs/DEMO_SCRIPT.md`.
- [ ] Create `docs/LIMITATIONS_AND_NEXT_STEPS.md`.
- [ ] Create `docs/MODEL_CARD_TEMPLATE.md`.
- [ ] Add exact commands for:
  - [ ] run server
  - [ ] run Flutter app
  - [ ] run evaluation
  - [ ] export model
  - [ ] reproduce cloud training
- [ ] Include before/after metrics table.
- [ ] Include screenshots/GIFs if available.
- [ ] Frame limitations professionally.

### Acceptance Criteria

- [ ] Demo can be run by someone besides the original developer.
- [ ] Client can understand what improved.
- [ ] Limitations are honest and confidence-building.
- [ ] Next data collection plan is clear.

---

## Epic 14 - CI, Quality, and Project Hygiene

### P1 Tasks

- [ ] Update GitHub Actions for Python tests.
- [ ] Add lint/format commands.
- [ ] Add `ruff` if not already present.
- [ ] Add selective `mypy` if feasible.
- [ ] Add Flutter analyze workflow.
- [ ] Add small fixture images for tests.
- [ ] Add `.gitignore` entries for:
  - [ ] datasets
  - [ ] reports
  - [ ] model weights
  - [ ] exports
  - [ ] local envs
- [ ] Add artifact naming convention.
- [ ] Reference `docs/ACCEPTANCE_GATES.md` in CI so each batch's PR
      description can be validated against a concrete gate.
- [ ] Track `datasets/rfconnectors/dataset.lock.json` even when the bulk
      dataset itself is gitignored, so runs remain reproducible.
- [ ] Ensure CI does not require GPU.

### Acceptance Criteria

- [ ] Tests run predictably.
- [ ] Large datasets/models are not accidentally committed.
- [ ] CI does not require GPU.
- [ ] Cloud training remains documented.

---

## Epic 15 - Advanced Enhancements

### P2/P3 Tasks

- [ ] Active learning loop.
- [ ] Low-confidence collection queue.
- [ ] Periodic retraining workflow.
- [ ] Calibrated measurement mode.
- [ ] SAM/SAM2-assisted annotation.
- [ ] VLM/LLM explanation assistant.
- [ ] Manufacturer part-number lookup.
- [ ] Barcode/QR inventory workflow.
- [ ] AR overlay for dimensions.
- [ ] Cloud dashboard for corrections.

### Acceptance Criteria

- [ ] Advanced features do not block core demo.
- [ ] Measurement estimates are labeled as estimates.
- [ ] LLM/VLM does not override visual/geometry confidence without evidence.

---

## Execution Batch 1 - Completed: Repo Audit and Taxonomy

Completed deliverables:

- `docs/REPO_AUDIT.md`
- `docs/CONNECTOR_TAXONOMY.md`
- `training/rfconnectorai/specs/connectors.yaml`
- `training/rfconnectorai/schemas/taxonomy.py`
- `training/tests/test_taxonomy.py`

---

## Execution Batch 2 - Dataset Audit and Schema Foundations

Implement Epic 2 plus the schema foundations needed before crops are
generated. Targets gate `G1 - Dataset Readiness` plus partial `G2`.

Scope:

1. `docs/ANNOTATION_PROTOCOL.md` (already authored - confirm and link).
2. `docs/ACCEPTANCE_GATES.md` (already authored - confirm and link).
3. `training/rfconnectorai/schemas/instance.py` (already authored - add
   tests for schema validation, including adapter rules).
4. `training/rfconnectorai/data/audit.py` plus CLI:

   ```bash
   python -m rfconnectorai.data.audit --data-dir data --out docs/DATASET_AUDIT.md
   ```

5. The audit must support:
   - `training/Images`
   - `training/data/labeled`
   - `training/data/test_holdout`
   - `training/data/reference`
   - `training/data/videos`
6. The audit must report:
   - image counts by folder/class,
   - file type and dimensions,
   - unreadable/corrupt files,
   - duplicates by hash,
   - likely synthetic vs real from path/name metadata,
   - likely multi-connector images (lightweight heuristics only),
   - missing taxonomy classes,
   - classes below target sample count,
   - train/val/test/holdout leakage risks.
7. Add JSON output beside markdown if practical:
   `docs/DATASET_AUDIT.json`.
8. Add tests for:
   - instance schema validation,
   - audit utility functions using tiny temp fixtures,
   - audit CLI dry run or fixture run.

Constraints:

- Do not modify or move original images.
- Do not train models.
- Do not change existing `/predict` behavior.
- Do not rewrite the Flutter app.
- Do not add GPU-only dependencies.
- Preserve all current tests.

Return:

- files changed
- generated audit summary
- commands run
- cloud/local environment used
- test results or blockers
- next recommended batch

---

## Execution Batch 3 - Instance Catalog and Crop Workflow

Implement Epic 3.

Return:

- files changed
- manifest schema summary
- crop workflow summary
- command examples
- known limitations

---

## Execution Batch 4 - Dataset Standardization

Implement Epic 4.

Return:

- files changed
- dataset tree created or dry-run report
- validation results
- command examples
- known limitations

---

## Execution Batch 5 - First Detector Track

Implement Epic 5.

Return:

- files changed
- cloud command examples
- config validation results
- how to run first detector training
- no local heavy training confirmation

---

## Execution Batch 6 - Multi-Head Classifier Track

Implement Epic 6.

Return:

- files changed
- supported backbones
- command examples
- label encoding tests
- next integration steps

---

## Execution Batch 7 - 3D/Synthetic Pipeline

Implement Epic 7.

Return:

- files changed
- model catalog summary
- render command examples
- synthetic label format
- known limitations

---

## Execution Batch 8 - Evaluation Harness

Implement Epic 9.

Return:

- files changed
- report schema
- metrics supported
- cloud run instructions
- blockers

---

## Execution Batch 9 - API Upgrade

Implement Epic 10.

Return:

- files changed
- example old-compatible JSON
- example rich JSON
- test results
- Flutter changes needed next

---

## Execution Batch 10 - Flutter Upgrade

Implement Epic 11.

Return:

- files changed
- UI behavior summary
- analyze/test results from Flutter-capable environment
- screenshots instructions if applicable

---

## Execution Batch 11 - Export/Deployment

Implement Epic 12.

Return:

- files changed
- export commands
- server/mobile compatibility notes
- latency benchmark plan

---

## Execution Batch 12 - Client Demo Package

Implement Epic 13.

Return:

- files changed
- demo instructions
- remaining blockers before client presentation
