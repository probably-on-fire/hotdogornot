# Model Card Template

Use this template for every promoted detector / classifier / multihead /
embedder run. The eval harness emits a populated copy at
``reports/experiments/<run>/model_card.md``; promote that copy when you
graduate the run.

## Identity

- ``model_id``:
- ``model_type``: detector | classifier | multihead_classifier | embedder | geometry_verifier
- ``architecture``:
- ``trained_on``: ``datasets/rfconnectors@<dataset_id>``
- ``taxonomy_version``: ``<connectors_yaml_sha256>``
- ``created_at``: ``<iso8601>``
- ``model_record``: ``reports/experiments/<run>/model_record.json``

## Training

- Dataset lock: ``datasets/rfconnectors/dataset.lock.json``
- Cloud env: Kaggle | Colab | other
- Hardware: GPU type, memory
- Hyperparameters: epochs, batch, imgsz, learning rate, loss, optimizer
- Augmentations
- Seed

## Evaluation Set

- Source: ``datasets/rfconnectors/images/test`` and / or
  ``training/data/test_holdout/``
- Real vs synthetic split
- Per-family counts
- Known leakage risks (must be empty)

## Metrics

| Metric | Value |
|---|---:|
| mAP@50 | |
| mAP@50-95 | |
| Background false positive rate | |
| Family accuracy | |
| Polarity accuracy | |
| Side A gender accuracy | |
| Side B gender accuracy | |
| Mount style accuracy | |
| Orientation accuracy | |
| Termination accuracy | |
| Macro F1 | |
| Top-2 accuracy | |
| Expected calibration error | |
| Abstention coverage | |
| Abstention selective score | |
| Latency p50 / p90 (ms) | |
| Model size (MB) | |

## Failure Modes

Embed or link the failure gallery
``reports/experiments/<run>/failure_gallery.html``.

- Visually similar families that the model confuses (e.g. SMA / 3.5mm).
- Conditions where the model abstains correctly.
- Conditions where the model abstains incorrectly (false negatives).

## Limitations

- Real-phone holdout size and what one miss shifts.
- Geometry confidence without a scale reference.
- Adapters with one occluded side.
- Mobile latency / thermal envelope (if measured).

## Compatibility

- ``/predict`` legacy fields preserved: yes / no.
- ONNX exported: yes / no.
- TFLite/LiteRT exported: yes / no.
- Core ML exported: yes / no.

## Promotion Decision

- Comparison to prior model:
- Acceptance gate: ``G0`` / ``G1`` / ``G2`` / ``G3`` / ``G4`` / ``G5``
  ([details](ACCEPTANCE_GATES.md)).
- Approver:
- Date:
