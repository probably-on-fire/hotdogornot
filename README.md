<div align="center">

<img src="flutter/assets/icon/icon.png" alt="Connector ID" width="128" height="128" />

# Connector ID

**RF connector identification from a phone or desktop camera.**

SMA, RP-SMA, 3.5mm, 2.92mm/K/SMK, 2.4mm, 1.85mm, 1.0mm, SSMA, SMB,
SMC, QMA, TNC, BNC, MCX, 7/16 DIN, and unknown/unsupported.

Powered by [aired.com](https://aired.com)

</div>

---

## What This Is

Connector ID is evolving from a proof-of-concept RF connector classifier
into a production-grade identification system for RF coaxial connectors.
The goal is simple for the end user: point a camera at a connector and get
a correct, useful result.

The target result is not just one flat class label. The system should infer:

- connector family/type,
- standard vs reverse polarity,
- gender/contact configuration,
- mount style,
- orientation,
- cable termination where visible,
- size/geometry cues when a scale reference is available,
- confidence, ambiguity, and top alternatives,
- cross-referenced engineering specs.

The authoritative roadmap is:

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- [`TASKS.md`](TASKS.md)

The first implementation batch completed the repo audit and taxonomy/spec
foundation:

- [`docs/REPO_AUDIT.md`](docs/REPO_AUDIT.md)
- [`docs/CONNECTOR_TAXONOMY.md`](docs/CONNECTOR_TAXONOMY.md)
- [`training/rfconnectorai/specs/connectors.yaml`](training/rfconnectorai/specs/connectors.yaml)
- [`training/rfconnectorai/schemas/taxonomy.py`](training/rfconnectorai/schemas/taxonomy.py)

---

## Current Baseline

Current production behavior is preserved.

- Flutter app in `flutter/`
- FastAPI predict service in `training/rfconnectorai/server/predict_service.py`
- Existing `/predict` endpoint shape remains:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "predictions": [
    {
      "class_name": "2.4mm-M",
      "confidence": 0.83,
      "probabilities": {},
      "bbox": {"x": 612, "y": 415, "w": 240, "h": 240}
    }
  ]
}
```

Current documented model:

| Metric | v18 Baseline |
|---|---:|
| Full class accuracy | 75% |
| Family accuracy | 75% |
| Gender accuracy | 87.5% |
| Background false positives | 0% |
| Held-out size | 8 phone shots |

The current model is an ImageNet-pretrained ResNet-18 with a linear head.
The 8-image holdout is too small to support strong accuracy claims; one
miss changes accuracy by 12.5 percentage points.

---

## Target Architecture

The planned production architecture is a staged computer vision pipeline:

```text
Camera frame
  -> connector/background detector
  -> connector crop or mask
  -> multi-head attribute classifier
  -> optional measurement/calibration module
  -> confidence and ambiguity logic
  -> connector spec lookup
  -> mobile/desktop result card
```

Graphviz source for the detailed architecture diagram:

- [`docs/SOFTWARE_ARCHITECTURE.dot`](docs/SOFTWARE_ARCHITECTURE.dot)
- [`docs/SOFTWARE_ARCHITECTURE.svg`](docs/SOFTWARE_ARCHITECTURE.svg)
- [`docs/SOFTWARE_ARCHITECTURE.png`](docs/SOFTWARE_ARCHITECTURE.png)

Render it with:

```bash
dot -Tpng docs/SOFTWARE_ARCHITECTURE.dot -o docs/SOFTWARE_ARCHITECTURE.png
dot -Tsvg docs/SOFTWARE_ARCHITECTURE.dot -o docs/SOFTWARE_ARCHITECTURE.svg
```

The full architecture notes remain in:

- [`training/docs/architecture.md`](training/docs/architecture.md)

---

## Repository Layout

```text
.
|-- IMPLEMENTATION_PLAN.md       authoritative product/architecture roadmap
|-- TASKS.md                     implementation backlog by epic
|-- README.md                    repo overview
|-- docs/
|   |-- REPO_AUDIT.md            current repo and safety baseline
|   |-- CONNECTOR_TAXONOMY.md    connector families and attribute heads
|   |-- SOFTWARE_ARCHITECTURE.dot detailed Graphviz architecture script
|   |-- printables/              ArUco marker assets
|   `-- superpowers/             historical plans/specs
|-- flutter/
|   |-- lib/src/api.dart         current /predict client parser
|   |-- lib/src/screens/         Identify, Contribute, About
|   `-- README.md                Flutter app guide
|-- training/
|   |-- rfconnectorai/
|   |   |-- classifier/          current ResNet classifier path
|   |   |-- measurement/         geometry/ArUco/hex/aperture tools
|   |   |-- server/              FastAPI predict and relay services
|   |   |-- schemas/             taxonomy and future response schemas
|   |   |-- specs/               connector spec YAML
|   |   |-- synthetic/           synthetic data generation
|   |   `-- data_fetch/          image/video data collection helpers
|   |-- tests/                   pytest suite
|   |-- docs/                    training architecture/runbook/history
|   `-- README.md                training-side guide
`-- unity/                       historical Unity AR app
```

Planned additions from later task batches:

```text
training/rfconnectorai/data/audit.py
training/rfconnectorai/data/build_yolo_dataset.py
training/rfconnectorai/detector/train_yolo.py
training/rfconnectorai/classifier/model_multihead.py
training/rfconnectorai/classifier/train_multihead.py
training/rfconnectorai/eval/evaluate_all.py
training/rfconnectorai/schemas/prediction.py
datasets/rfconnectors/
reports/experiments/
exports/mobile/
```

---

## Quick Start

### Training Package

Use Python 3.11 or newer.

```bash
cd training
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows
.venv/bin/pip install -e ".[dev]"          # macOS/Linux
python -m pytest tests/ -q
```

Run the current FastAPI predict service:

```bash
cd training
uvicorn rfconnectorai.server.predict_service:app --port 8503
```

Train the current ResNet baseline:

```bash
cd training
python -m rfconnectorai.classifier.train \
  --data-dir data/labeled/embedder \
  --out-dir models/connector_classifier \
  --epochs 20
```

### Flutter App

```bash
cd flutter
flutter pub get
flutter run
```

The app currently provides:

- Identify: camera/photo/video prediction flow.
- About: product info, privacy, request form, dev-mode unlock.
- Contribute: dev-only training and holdout capture flow.

---

## Development Rules

- Do not rewrite the whole app.
- Preserve existing `/predict` behavior and Flutter screens.
- Add new structured output beside old fields, not instead of them.
- Keep spec lookup separate from model inference.
- Treat `unknown`, `unsupported`, and `need another angle` as valid outcomes.
- Do not claim 99.99% accuracy without statistically meaningful validation.
- Every model improvement must include test data discipline, metrics, and
  visible failure cases.

---

## Documentation Index

| Doc | Purpose |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Product mission, architecture, accuracy gates, dataset/training/app/API plan |
| [`TASKS.md`](TASKS.md) | Epic-by-epic backlog and execution batches |
| [`docs/REPO_AUDIT.md`](docs/REPO_AUDIT.md) | Current repository audit and safety baseline |
| [`docs/CONNECTOR_TAXONOMY.md`](docs/CONNECTOR_TAXONOMY.md) | Connector family taxonomy and attribute labels |
| [`docs/SOFTWARE_ARCHITECTURE.dot`](docs/SOFTWARE_ARCHITECTURE.dot) | Graphviz source for the full I/O architecture diagram |
| [`training/docs/architecture.md`](training/docs/architecture.md) | Current v18 architecture plus roadmap architecture |
| [`training/docs/classifier_journey.md`](training/docs/classifier_journey.md) | Experiment history and lessons learned |
| [`training/docs/runbook.md`](training/docs/runbook.md) | Deploy/retrain operations |
| [`training/docs/capture_protocol.md`](training/docs/capture_protocol.md) | Capture protocol for new connector data |
| [`flutter/README.md`](flutter/README.md) | Flutter app behavior, backend coupling, and build notes |
| [`training/README.md`](training/README.md) | Training and serving stack guide |

---

<div align="center">

**Built and operated by [aired.com](https://aired.com)**

</div>
