# RF Connector AI — Training Pipeline

Python pipeline that trains a YOLOv11n connector detector and a MobileViT-v2 RGBD embedder, then exports both to ONNX for Unity Sentis.

## Setup

    uv venv
    uv pip install -e ".[dev]"

## Pipeline

1. Acquire proxy data: `python -m rfconnectorai.data.scrape` and `python -m rfconnectorai.data.synthetic`
2. Train detector: `python -m rfconnectorai.training.train_detector`
3. Train embedder: `python -m rfconnectorai.training.train_embedder`
4. Build references: `python -m rfconnectorai.inference.build_references`
5. Export ONNX: `python -m rfconnectorai.export.onnx_export`
6. Evaluate: `python -m rfconnectorai.inference.eval`

## Test

    pytest
