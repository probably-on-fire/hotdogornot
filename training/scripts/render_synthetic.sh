#!/usr/bin/env bash
set -euo pipefail

# Drive the full synthetic render pipeline for all 8 classes.
# Expects CAD files under data/cad/ matching configs/cad_sources.yaml.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PER_CLASS="${PER_CLASS:-2000}"
IMG_SIZE="${IMG_SIZE:-384}"
SAMPLES="${SAMPLES:-32}"
HDRI_DIR="${HDRI_DIR:-}"

# Accept KEY=VALUE pairs passed as positional args (e.g. PER_CLASS=3 IMG_SIZE=64).
for arg in "$@"; do
    case "$arg" in
        PER_CLASS=*) PER_CLASS="${arg#*=}" ;;
        IMG_SIZE=*)  IMG_SIZE="${arg#*=}" ;;
        SAMPLES=*)   SAMPLES="${arg#*=}" ;;
        HDRI_DIR=*)  HDRI_DIR="${arg#*=}" ;;
    esac
done

# Prefer the local venv python if available, otherwise fall back to `python`.
if [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

CMD=( "$PY" -m rfconnectorai.synthetic.pipeline
    --config configs/cad_sources.yaml
    --cad-root data/cad
    --output data/synthetic
    --per-class "$PER_CLASS"
    --image-size "$IMG_SIZE"
    --samples "$SAMPLES"
)
if [ -n "$HDRI_DIR" ]; then
    CMD+=( --hdri-dir "$HDRI_DIR" )
fi

echo "Rendering $PER_CLASS samples/class at ${IMG_SIZE}x${IMG_SIZE}..."
"${CMD[@]}"
echo "Done. Output in data/synthetic/"
