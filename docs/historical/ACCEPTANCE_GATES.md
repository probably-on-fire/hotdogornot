# Acceptance Gates

Each execution batch must hit the gate listed below before the next batch
begins. Gates are deliberately concrete so every contributor knows what
"done" means without re-reading the full implementation plan.

These gates are operational checkpoints between batches. They are not the
same as the accuracy targets G0-G6 in `IMPLEMENTATION_PLAN.md` section 9,
which describe model quality milestones.

## G0 - Safety Baseline

- Existing `/predict` response remains valid.
- Existing Flutter app does not break.
- Current ResNet path still loads or fails gracefully.

## G1 - Dataset Readiness

- Dataset audit completed and `docs/DATASET_AUDIT.md` is generated.
- Duplicate and leakage risk reported.
- Missing taxonomy classes reported.
- Holdout is isolated from train/val/test.

## G2 - Instance Catalog Readiness

- Every crop traces to its source image.
- Multi-connector images support multiple instance rows.
- Unknown / missing attributes are explicit, not silently filled.
- `instances.jsonl` rows validate against
  `training/rfconnectorai/schemas/instance.py`.

## G3 - First Detector Readiness

- Detector produces bounding box predictions on real test images.
- Background false positive rate is measured and reported.
- Multi-connector images are handled (multiple detections returned).
- Run metadata is captured under `reports/experiments/<timestamp>/`.

## G4 - Multi-Head Readiness

- Per-head metrics are reported for family, polarity, side A/B gender,
  mount, orientation, termination.
- Missing labels do not break training.
- Baseline comparison against ResNet-18 exists.
- Calibration error and abstention-aware correctness are reported.

## G5 - Demo Readiness

- API returns both old-compatible and rich structured response fields.
- Flutter UI displays confidence and ambiguity states.
- A new client can run the demo from documented steps in
  `docs/CLIENT_DEMO_README.md`.
- Limitations are honest and a next-step plan is documented.

## How To Use This File

- Each Codex batch must reference the gate it is targeting.
- Pull request descriptions should restate the gate criteria and link the
  evidence (audit report, metrics file, screenshots, etc.).
- Failing gates block the next batch. Document blockers explicitly rather
  than working around them.
