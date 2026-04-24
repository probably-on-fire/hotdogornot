#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline driver. Expects a labeled dataset at data/labeled/ with
# per-class subdirectories for the embedder, and Ultralytics YOLO format for
# the detector.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/labeled/embedder}"
DETECTOR_YAML="${DETECTOR_YAML:-configs/detector.yaml}"
CLASSES_YAML="${CLASSES_YAML:-configs/classes.yaml}"
RUNS="${RUNS:-runs}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$RUNS"

echo "[1/5] Training detector"
python -m rfconnectorai.training.train_detector \
    --data-yaml "$DETECTOR_YAML" \
    --output-dir "$RUNS/detector" \
    --device "$DEVICE"

echo "[2/5] Training embedder"
python -m rfconnectorai.training.train_embedder \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output-dir "$RUNS/embedder" \
    --device "$DEVICE"

echo "[3/5] Building reference embeddings"
python -m rfconnectorai.inference.build_references \
    --checkpoint "$RUNS/embedder/embedder.pt" \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output "$RUNS/reference_embeddings.bin" \
    --device "$DEVICE"

echo "[4/5] Evaluating"
python -m rfconnectorai.inference.eval \
    --checkpoint "$RUNS/embedder/embedder.pt" \
    --references "$RUNS/reference_embeddings.bin" \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output "$RUNS/eval_report.json" \
    --device "$DEVICE"

echo "[5/5] Exporting to ONNX (with INT8 quantized variants)"
python -m rfconnectorai.export.onnx_export \
    --embedder-checkpoint "$RUNS/embedder/embedder.pt" \
    --embedder-out "$RUNS/embedder.onnx" \
    --detector-weights "$RUNS/detector/detector.pt" \
    --detector-out "$RUNS/detector.onnx" \
    --quantize

echo "Done. Artifacts in $RUNS/:"
ls -la "$RUNS/"*.onnx "$RUNS/reference_embeddings.bin" "$RUNS/eval_report.json"
