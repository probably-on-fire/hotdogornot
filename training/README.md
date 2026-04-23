# RF Connector AI — Training Pipeline

Python pipeline that trains a YOLOv11n connector detector and a MobileViT-v2 RGBD embedder, then exports both to ONNX for consumption by the Unity Sentis runtime.

Spec: `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`

## Setup

    cd training
    uv venv
    uv pip install -e ".[dev]"

If `uv` is not installed, the standard equivalent works too:

    cd training
    python -m venv .venv
    .venv/Scripts/pip install -e ".[dev]"      # Windows
    .venv/bin/pip install -e ".[dev]"          # macOS/Linux

## Data layout

For the **embedder**, organize images into per-class directories matching `configs/classes.yaml`:

    data/labeled/embedder/
      SMA-M/
        img0001.png
        ...
      SMA-F/
      3.5mm-M/
      ...

For the **detector**, use standard Ultralytics YOLO layout pointed at by `configs/detector.yaml`:

    data/labeled/detector/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt   # "0 cx cy w h" per line, normalized
      labels/val/*.txt

During Phase 0 (before real connectors arrive), populate both with a mix of:
- `python -m rfconnectorai.data.scrape` (catalog images)
- `python -m rfconnectorai.data.synthetic` (procedural renders)

## Full pipeline

    bash scripts/run_pipeline.sh

Produces under `runs/`:
- `detector.onnx`
- `embedder.onnx`
- `reference_embeddings.bin`
- `eval_report.json`

## Individual steps

    python -m rfconnectorai.training.train_detector --data-yaml configs/detector.yaml --output-dir runs/detector
    python -m rfconnectorai.training.train_embedder --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output-dir runs/embedder
    python -m rfconnectorai.inference.build_references --checkpoint runs/embedder/embedder.pt --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output runs/reference_embeddings.bin
    python -m rfconnectorai.inference.eval --checkpoint runs/embedder/embedder.pt --references runs/reference_embeddings.bin --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output runs/eval_report.json
    python -m rfconnectorai.export.onnx_export --embedder-checkpoint runs/embedder/embedder.pt --embedder-out runs/embedder.onnx --detector-weights runs/detector/detector.pt --detector-out runs/detector.onnx

## Tests

    pytest                                 # all
    pytest tests/test_end_to_end.py -v     # full-pipeline smoke test

## Configuration

- `configs/classes.yaml` — the 8 RF connector classes with metadata
- `configs/detector.yaml` — YOLO dataset spec
- `configs/embedder.yaml` — hyperparameters for embedder training

## Notes

- On Windows, the synthetic-data renderer (`rfconnectorai.data.synthetic`) uses a PIL-based procedural generator instead of a GL-backed 3D rasterizer, because `pyrender` does not load a GL context cleanly on headless Windows. The trimesh mesh builder is retained for downstream geometric reasoning. On Linux with EGL available, a pyrender-based renderer could be swapped back in; Phase-0 proxy data does not depend on the higher-quality path.
- The embedder projection head uses `nn.LayerNorm` rather than `nn.BatchNorm1d`. BatchNorm crashes on batch size 1 and leaks batch statistics in metric-learning setups; LayerNorm is the idiomatic choice.
- ONNX export uses `torch.onnx.export(..., dynamo=False)` to stay on the legacy TorchScript exporter, which does not require the optional `onnxscript` dependency.
