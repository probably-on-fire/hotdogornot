# Repository Audit

This audit covers the current repository state before implementing the
connector-specific roadmap in `IMPLEMENTATION_PLAN.md` and `TASKS.md`.
It is intentionally descriptive: no existing FastAPI, Flutter, training,
or prediction behavior has been changed.

## Current Structure

| Path | Role |
|---|---|
| `flutter/` | Flutter camera app for iOS/Android, with Identify, About, and dev-only Contribute flows. |
| `training/` | Python package containing FastAPI services, PyTorch training, inference, data ingestion, synthetic data, measurement utilities, deployment files, and tests. |
| `docs/` | Cross-cutting handoff, architecture, procurement, printable marker, and implementation planning docs. |
| `unity/` | Historical Unity AR app kept in the repo but described by current docs as sidelined. |

## Documentation Baseline

The root `README.md` describes a two-part Connector ID system:

- Flutter app in `flutter/`, sending camera photos to the backend.
- Training and serving stack in `training/`, using a fine-tuned ResNet-18.
- Public predict endpoint: `https://aired.com/rfcai/predict`.

Current documented production model:

| Metric | Value |
|---|---:|
| Full class accuracy | 75% |
| Family accuracy | 75% |
| Gender accuracy | 87.5% |
| Background false positives | 0% |
| Held-out size | 8 phone shots |

The docs repeatedly warn that the 8-image holdout is too small to support
strong accuracy claims. A single incorrect prediction moves accuracy by
12.5 percentage points.

Important existing docs:

- `training/README.md`: training setup, model serving, synthetic data, Streamlit tools, data layout, and test command.
- `training/docs/architecture.md`: v18 inference flow and training recipe.
- `training/docs/classifier_journey.md`: experiment history and data quality findings.
- `training/docs/runbook.md`: production deployment, retraining, model files, and operational gotchas.
- `training/docs/capture_protocol.md`: image/video capture protocol for improving the dataset.
- `flutter/README.md`: app behavior, camera lifecycle, API coupling, dev mode, and build commands.

## Current Training Pipeline

The current classifier path is under `training/rfconnectorai/classifier/`.

- `train.py` trains an ImageNet-pretrained ResNet-18 with a final linear head.
- `predict.py` loads `weights.pt` and `labels.json`, applies ImageNet transforms,
  and returns `class_name`, `confidence`, and per-class probabilities.
- `dataset.py` provides folder-based image dataset handling and transforms.
- `export_onnx.py` supports export for deployment.

The current classifier is a flat class model over labels such as
`2.4mm-M`, `2.4mm-F`, `2.92mm-M`, `2.92mm-F`, `3.5mm-M`, and `3.5mm-F`.
Docs state SMA and 1.85mm are wired at the data/UI layer but not populated
enough for the trained head.

Additional training-support modules exist for:

- measurement: `training/rfconnectorai/measurement/`
- synthetic data: `training/rfconnectorai/synthetic/`
- data fetching: `training/rfconnectorai/data_fetch/`
- ingestion: `training/rfconnectorai/ingest/`
- inference/build references: `training/rfconnectorai/inference/`
- model export: `training/rfconnectorai/export/`

Dataset paths documented by the repo:

- `training/data/labeled/embedder/<class>/`
- `training/data/test_holdout/<class>/`
- `training/data/videos/`
- `training/data/reference/pasternack/`
- generated or operational data under `training/data/archive/`,
  `training/data/synthetic_faces/`, `training/data/synthetic_angled/`,
  and `training/data/cad/`

## Current FastAPI Services

There are two FastAPI app modules:

| Module | Purpose |
|---|---|
| `training/rfconnectorai/server/predict_service.py` | Production-style detect/crop/classify service with `/predict`, `/predict-video`, `/healthz`, and mounted `/labeler/*` routes. |
| `training/rfconnectorai/server/app.py` | Relay-style app for uploads and model artifact distribution, with `/uploads`, `/model/version`, `/model/latest`, `/model/weights`, `/model/weights.onnx`, and `/model/labels`. |

Current `/predict` behavior in `predict_service.py`:

- Requires `X-Device-Token`.
- Accepts multipart `image`.
- Decodes with OpenCV.
- Detects connector crops with `detect_connector_crops`.
- Optionally gates crops through the rembg foreground filter.
- Optionally classifies on a rembg-cleaned white-background crop.
- Uses the loaded ResNet classifier or ensemble.
- Returns:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "predictions": [
    {
      "class_name": "2.4mm-M",
      "confidence": 0.83,
      "probabilities": {},
      "bbox": {"x": 612, "y": 415, "w": 240, "h": 240},
      "_diag": {}
    }
  ]
}
```

The existing endpoint path and response shape must remain compatible while
future structured predictions are added beside it.

## Current Flutter App

The Flutter app is under `flutter/`.

Key files:

- `flutter/lib/main.dart`
- `flutter/lib/src/app.dart`
- `flutter/lib/src/api.dart`
- `flutter/lib/src/settings.dart`
- `flutter/lib/src/screens/main_shell.dart`
- `flutter/lib/src/screens/identify_screen.dart`
- `flutter/lib/src/screens/contribute_screen.dart`
- `flutter/lib/src/screens/about_screen.dart`

Current API parsing in `api.dart` expects the existing `/predict` JSON:

- `image_width`
- `image_height`
- `predictions`
- per prediction: `class_name`, `confidence`, `probabilities`, `bbox`

The Identify screen currently:

- captures a photo or video,
- sends to `/predict` or `/predict-video`,
- applies local confidence and bbox-size gates,
- shows a result panel,
- exposes correction chips for family and gender.

The Contribute screen is dev-only and uploads training or holdout captures
to labeler endpoints.

## Current Tests

The training package has a pytest suite under `training/tests/`.
At audit time, there are 41 test files covering:

- classifier dataset/train/predict/export behavior,
- FastAPI relay and predict service paths,
- measurement detectors and utilities,
- synthetic rendering,
- ingestion,
- data fetching helpers,
- ensemble and averaging utilities.

The documented command is:

```bash
cd training
python -m pytest tests/ -q
```

Flutter has tests under `flutter/test/`, including API and widget tests.
The expected static analysis command is:

```bash
cd flutter
flutter analyze
```

## Safety Baseline

No existing behavior should be removed in Epic 0 or Epic 1.

Preserve:

- `training/rfconnectorai/server/predict_service.py` `/predict`
- `training/rfconnectorai/server/predict_service.py` `/predict-video`
- existing top-level prediction fields consumed by Flutter
- existing Flutter screens and camera flow
- existing ResNet classifier code path
- existing training scripts and documented commands

## Known Gaps Before Epic 2+

- Current classifier is a flat label model, not a multi-attribute taxonomy.
- SMA/RP-SMA and many related RF connector families are not represented in
  the current trained head.
- Dataset coverage is not sufficient to prove very high accuracy.
- The holdout set is too small for statistically meaningful claims.
- The current model can classify crops but does not provide structured spec
  lookup, ambiguity handling, or calibrated measurement output.
- Dataset audit, standardized detection dataset export, detector training,
  multi-head classifier training, and structured prediction schemas are
  planned but not implemented in this first batch.

## Local Validation Notes

Commands run during this audit batch:

```bash
cd training
python -m pytest tests/test_taxonomy.py -q
python -m pytest tests/ -q
python -c "from rfconnectorai.schemas.taxonomy import load_taxonomy; t=load_taxonomy(); print(len(t.connectors))"

cd flutter
flutter --version
```

Results:

- `tests/test_taxonomy.py`: passed, 5 tests.
- taxonomy import check: passed, loaded 16 connector specs.
- full `training/tests/`: blocked during collection in this local shell.
  The active Python is 3.9, while `training/pyproject.toml` requires
  Python `>=3.11,<3.14`; several dependencies are also missing locally
  (`timm`, `onnxruntime`, `trimesh`, `fastapi`).
- `flutter --version`: blocked because Flutter is not on PATH.
- documented `C:\flutter\bin\flutter.bat`: not present on this machine.
