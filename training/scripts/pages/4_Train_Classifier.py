"""
Streamlit page to train + evaluate the connector image classifier.

Workflow:
  1. Pick a labeled-data root (defaults to data/labeled/embedder)
  2. See per-class counts (this is the dataset size you're training on)
  3. Set epochs / batch / lr / val-fraction
  4. Hit Train — runs in-process, prints per-epoch metrics
  5. Hit "Run on a sample image" — load a test image and see the classifier's
     prediction with the full per-class probability breakdown.

Trained weights save to `models/connector_classifier/` and are picked up by
the predict module (and any code that wants to ensemble with the measurement
pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from rfconnectorai.classifier.dataset import ConnectorFolderDataset
from rfconnectorai.classifier.predict import ConnectorClassifier
from rfconnectorai.classifier.train import TrainConfig, train


REPO_TRAINING = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_TRAINING / "data" / "labeled" / "embedder"
DEFAULT_MODEL_DIR = REPO_TRAINING / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]


st.set_page_config(page_title="Train classifier", layout="wide")
st.title("Train connector classifier")
st.caption(
    "Fine-tunes a ResNet-18 (pretrained on ImageNet) on the labeled folders "
    "under `data/labeled/embedder/<CLASS>/`. The classifier complements the "
    "geometric measurement pipeline — it predicts class on any image, even "
    "non-perpendicular product photos that the measurement pipeline can't fire on."
)

with st.sidebar:
    st.markdown("### Dataset")
    data_dir = Path(st.text_input("Data root", value=str(DEFAULT_DATA_DIR)))

    counts = {}
    if data_dir.is_dir():
        ds = ConnectorFolderDataset(data_dir, class_names=CANONICAL_CLASSES)
        counts = ds.class_counts()
    for cls in CANONICAL_CLASSES:
        st.write(f"- **{cls}**: {counts.get(cls, 0)}")
    st.write(f"_Total: {sum(counts.values())} images_")

    if sum(counts.values()) == 0:
        st.warning(
            "No labeled images found. Use the Fetch Images / Process Video pages "
            "to populate this folder first."
        )

st.markdown("### Train")
col1, col2, col3, col4 = st.columns(4)
epochs = col1.number_input("Epochs", min_value=1, max_value=50, value=8)
batch_size = col2.number_input("Batch size", min_value=2, max_value=64, value=16)
learning_rate = col3.number_input(
    "Learning rate", min_value=1e-5, max_value=1e-2, value=3e-4, format="%.5f", step=1e-4,
)
val_fraction = col4.slider("Val split", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

model_dir = Path(st.text_input("Model output dir", value=str(DEFAULT_MODEL_DIR)))

if st.button("Train", type="primary", disabled=sum(counts.values()) == 0):
    config = TrainConfig(
        data_dir=data_dir,
        out_dir=model_dir,
        class_names=CANONICAL_CLASSES,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        val_fraction=float(val_fraction),
    )
    progress = st.progress(0.0, text="Starting…")
    status = st.empty()
    try:
        # Wire training prints to the Streamlit status area by patching `print`
        # for the duration of the call. Train still prints to stdout for CLI use.
        original_print = print
        history_lines: list[str] = []

        def _capture_print(*args, **kwargs):
            line = " ".join(str(a) for a in args)
            history_lines.append(line)
            status.code("\n".join(history_lines))
            # Bump the progress bar each epoch line.
            for i in range(1, int(epochs) + 1):
                if line.startswith(f"epoch {i:>2}/"):
                    progress.progress(i / int(epochs), text=line)
                    break
            original_print(*args, **kwargs)

        import builtins
        builtins.print = _capture_print
        try:
            metrics = train(config)
        finally:
            builtins.print = original_print

        progress.progress(1.0, text="Done")
        st.success(f"Trained. Weights saved to `{model_dir}`")
        st.markdown("**Final metrics**")
        last = metrics["history"][-1]
        st.write(
            f"- train_acc: {last['train_acc']:.3f}  |  val_acc: {last['val_acc']:.3f}\n"
            f"- train_loss: {last['train_loss']:.3f}  |  val_loss: {last['val_loss']:.3f}"
        )
    except Exception as e:
        st.error(f"Training failed: {e}")

st.divider()
st.markdown("### Predict on a sample image")

if not (model_dir / "weights.pt").exists():
    st.info(f"Train a model first (no weights at `{model_dir}/weights.pt`).")
else:
    uploaded = st.file_uploader(
        "Upload an image to classify", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is not None:
        nparr = np.frombuffer(uploaded.getvalue(), np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode image.")
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with st.spinner("Predicting…"):
                classifier = ConnectorClassifier.load(model_dir)
                pred = classifier.predict(rgb)
            col_a, col_b = st.columns([1, 1])
            col_a.image(rgb, caption="Input", use_container_width=True)
            with col_b:
                st.success(f"**{pred.class_name}** (confidence {pred.confidence:.0%})")
                st.markdown("**Per-class probabilities**")
                for cls_name, prob in sorted(
                    pred.probabilities.items(), key=lambda kv: -kv[1]
                ):
                    st.write(f"- {cls_name}: {prob:.3f}")
