# Client Demo README

This file is the entry point for the client-facing demo of the RF
connector identification system. Anyone besides the original developer
should be able to run the demo using only this document.

## What the Demo Shows

- Real-time-ish identification of an RF connector from a phone or desktop
  camera.
- Family + side A / side B gender + polarity + mount style + orientation
  + termination, with confidence and top alternatives.
- Honest abstention states (``need_second_angle``,
  ``need_scale_reference``, ``unsupported_connector``,
  ``no_connector_detected``).
- Spec lookup joined onto the prediction.

The demo is *not* a 99.99% accuracy claim. See
[`LIMITATIONS_AND_NEXT_STEPS.md`](LIMITATIONS_AND_NEXT_STEPS.md) for the
honest accuracy framing.

## Prerequisites

- Python 3.11+
- Flutter (for the mobile/desktop client) — optional if running the API only
- A connector to test on, ideally with a small ruler or printed scale
  marker (see `docs/printables/`).

## Run the Server

```bash
cd training
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows
.venv/bin/pip install -e ".[dev]"          # macOS/Linux

uvicorn rfconnectorai.server.predict_service:app --port 8503
```

The legacy ``/predict`` endpoint stays compatible with the existing
Flutter client. Once the API schema upgrade in
``rfconnectorai/schemas/prediction.py`` is wired into the FastAPI
handler, the same endpoint will additionally return the rich structured
fields described in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

## Run the Flutter App

```bash
cd flutter
flutter pub get
flutter run
```

Point the app at the server (settings screen). Identify, About, and
Contribute screens stay where they were.

## Run an Evaluation

```bash
python -m rfconnectorai.eval.evaluate_all \
    --predictions reports/experiments/<run>/predictions.jsonl \
    --dataset-lock datasets/rfconnectors/dataset.lock.json \
    --detector-record reports/experiments/<run>/model_record.json \
    --classifier-record reports/experiments/<run>/model_record.json \
    --out reports/experiments/<run>
```

Generates ``metrics.json`` and ``model_card.md``.

## Re-run the Cloud Pipeline

Local-PC instructions stop at ``--dry-run``. Heavy training runs in
Kaggle/Colab — see [`docs/MULTI_ARCHITECTURE_TRANSITION.md`](MULTI_ARCHITECTURE_TRANSITION.md)
for the cloud workflow.

## Demo Script

Walk-through, including talking points and what to do when the model
abstains, lives in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

## Honest Limitations

[`LIMITATIONS_AND_NEXT_STEPS.md`](LIMITATIONS_AND_NEXT_STEPS.md) is
required reading before showing the demo to a client.
