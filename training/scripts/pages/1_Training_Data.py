"""
Training Data — one place to manage the labeled dataset.

Three tabs:
  - Upload + Label: drop a video, auto-detect connector crops, label each one,
    save to data/labeled/embedder/<CLASS>/.
  - Review: walk through a class folder with the current classifier's
    prediction alongside the on-disk label; bulk keep / delete / move.
  - Train: trigger a fine-tune of the ResNet-18 classifier on the current
    labeled folders, then test it on a sample image.

The continuous-learning loop also writes new crops here from phone uploads,
so this is the single point of truth for what the model sees.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st

from rfconnectorai.classifier.dataset import ConnectorFolderDataset
from rfconnectorai.classifier.predict import ConnectorClassifier
from rfconnectorai.classifier.train import TrainConfig, train
from rfconnectorai.data_fetch.connector_crops import detect_connector_crops


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data" / "labeled" / "embedder"
DEFAULT_MODEL_DIR = REPO / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]
LABEL_CHOICES = ["(skip)"] + CANONICAL_CLASSES
ACTION_CHOICES = ["Keep", "Delete (false positive)"] + [f"Move to {c}" for c in CANONICAL_CLASSES]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _class_counts() -> dict[str, int]:
    counts = {}
    for cls in CANONICAL_CLASSES:
        d = DATA_ROOT / cls
        counts[cls] = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
    return counts


def _next_video_idx(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    indices = []
    for p in class_dir.glob("video_*.jpg"):
        tail = p.stem[len("video_"):]
        if tail.isdigit():
            indices.append(int(tail))
    return max(indices) + 1 if indices else 0


def _extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Path]:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out_pat = str(out_dir / "frame_%04d.jpg")
    subprocess.run(
        [ff, "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "4", out_pat],
        capture_output=True, check=False,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


@st.cache_resource(show_spinner=False)
def _load_classifier(model_dir_str: str | None):
    if model_dir_str is None: return None
    p = Path(model_dir_str)
    if not (p / "weights.pt").exists() or not (p / "labels.json").exists():
        return None
    try:
        return ConnectorClassifier.load(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _predict_path(_clf_id: str, img_path_str: str) -> tuple[str, float]:
    clf = _load_classifier(_clf_id) if _clf_id else None
    if clf is None: return ("(no model)", 0.0)
    bgr = cv2.imread(img_path_str)
    if bgr is None: return ("(unreadable)", 0.0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    p = clf.predict(rgb)
    return (p.class_name, float(p.confidence))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Training Data", layout="wide")
st.title("Training Data")
st.caption(
    "One place to grow and curate the labeled dataset. Upload videos to "
    "extract new crops, review existing labels for accuracy, and retrain "
    "the classifier when the data is ready."
)

# Sidebar dataset summary, always visible.
with st.sidebar:
    st.markdown("### Current dataset")
    counts = _class_counts()
    for cls in CANONICAL_CLASSES:
        st.write(f"- **{cls}**: {counts.get(cls, 0)}")
    st.write(f"_Total: {sum(counts.values())} images_")

tab_upload, tab_review, tab_train = st.tabs(
    ["Upload + Label", "Review", "Train"]
)

# ===========================================================================
# Tab 1: Upload + Label
# ===========================================================================

with tab_upload:
    st.markdown("### Drop a connector video")
    st.caption(
        "The labeler auto-detects each connector in each frame, you pick "
        "the class for each crop, and the labeled crops land in "
        "`data/labeled/embedder/<CLASS>/` ready for training."
    )

    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="td_video",
    )

    col1, col2 = st.columns(2)
    fps = col1.number_input(
        "Extraction fps", min_value=0.5, max_value=15.0, value=2.0, step=0.5,
        help="Higher fps = more frames = more data, but more clicks.",
        key="td_fps",
    )
    max_crops = col2.number_input(
        "Max connectors per frame", min_value=1, max_value=6, value=3,
        key="td_maxcrops",
    )

    if "lbl_crops" not in st.session_state:
        st.session_state.lbl_crops = []
        st.session_state.lbl_labels = {}

    if uploaded is not None and st.button("Extract & detect", type="primary", key="td_extract"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="rfcai_label_") as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / f"clip{suffix}"
            video_path.write_bytes(uploaded.getvalue())

            with st.spinner(f"Extracting frames at {fps} fps…"):
                frames = _extract_frames(video_path, tmp_path, float(fps))
            if not frames:
                st.error("Couldn't decode any frames. Wrong codec?")
                st.stop()
            st.success(f"Extracted {len(frames)} frames")

            with st.spinner("Detecting connectors…"):
                collected = []
                for frame_idx, fp in enumerate(frames):
                    bgr = cv2.imread(str(fp))
                    if bgr is None: continue
                    results = detect_connector_crops(bgr, max_crops=int(max_crops))
                    for crop_idx, r in enumerate(results):
                        ok, buf = cv2.imencode(".jpg", r.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if not ok: continue
                        collected.append((frame_idx, crop_idx, buf.tobytes()))
            st.session_state.lbl_crops = collected
            st.session_state.lbl_labels = {}
            st.success(f"Detected {len(collected)} connector crops across {len(frames)} frames")

    crops = st.session_state.lbl_crops
    if crops:
        st.divider()
        st.markdown(f"### Label crops ({len(crops)} total)")
        st.caption("Pick a class for each crop. Use **(skip)** for false positives.")

        per_row = 4
        for row_start in range(0, len(crops), per_row):
            cols = st.columns(per_row)
            for col, idx in zip(cols, range(row_start, min(row_start + per_row, len(crops)))):
                frame_idx, crop_idx, jpeg_bytes = crops[idx]
                key = (frame_idx, crop_idx)
                with col:
                    st.image(jpeg_bytes, caption=f"f{frame_idx}.c{crop_idx}", use_container_width=True)
                    current = st.session_state.lbl_labels.get(key, "(skip)")
                    choice = st.selectbox(
                        "class",
                        options=LABEL_CHOICES,
                        index=LABEL_CHOICES.index(current) if current in LABEL_CHOICES else 0,
                        key=f"td_lbl_{frame_idx}_{crop_idx}",
                        label_visibility="collapsed",
                    )
                    st.session_state.lbl_labels[key] = choice

        st.divider()
        tally = {}
        for v in st.session_state.lbl_labels.values():
            tally[v] = tally.get(v, 0) + 1
        st.write(", ".join(f"**{n}** {k}" for k, n in sorted(tally.items())) or "(no choices yet)")

        if st.button("Save labeled crops", type="primary", key="td_save"):
            saved = 0
            for (frame_idx, crop_idx, jpeg_bytes) in crops:
                cls = st.session_state.lbl_labels.get((frame_idx, crop_idx), "(skip)")
                if cls == "(skip)" or cls not in CANONICAL_CLASSES:
                    continue
                cls_dir = DATA_ROOT / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                idx = _next_video_idx(cls_dir)
                (cls_dir / f"video_{idx:04d}.jpg").write_bytes(jpeg_bytes)
                saved += 1
            if saved == 0:
                st.warning("No crops marked with a class — nothing saved.")
            else:
                st.success(f"Saved {saved} crops into `data/labeled/embedder/`")
                st.session_state.lbl_crops = []
                st.session_state.lbl_labels = {}
                st.balloons()

# ===========================================================================
# Tab 2: Review
# ===========================================================================

with tab_review:
    st.markdown("### Review labels")
    st.caption(
        "Walk through each image in a class folder. The classifier's "
        "prediction shows alongside the on-disk label — disagreements bubble "
        "to the top. Pick an action per image, then tap Apply."
    )

    review_col_left, review_col_right = st.columns([1, 3])
    with review_col_left:
        cls_options = [f"{c} ({counts.get(c, 0)})" for c in CANONICAL_CLASSES]
        sel = st.radio("Class to review", options=cls_options, index=0, key="td_review_class")
        target_class = sel.split(" (")[0]

        sort_mode = st.selectbox(
            "Sort by",
            options=["disagreements first", "lowest classifier confidence", "filename"],
            index=0,
            key="td_review_sort",
        )
        page_size = st.number_input(
            "Per page", min_value=8, max_value=64, value=24, step=8, key="td_review_pagesize",
        )
        use_classifier = st.checkbox(
            "Use classifier predictions",
            value=(DEFAULT_MODEL_DIR / "weights.pt").exists(),
            key="td_review_useclf",
        )

    with review_col_right:
        class_dir = DATA_ROOT / target_class
        if not class_dir.is_dir() or counts.get(target_class, 0) == 0:
            st.info(f"`{target_class}` has no images yet.")
        else:
            clf_id = str(DEFAULT_MODEL_DIR) if use_classifier else None

            images = sorted([p for p in class_dir.iterdir() if p.is_file()])
            records = []
            with st.spinner(f"Scoring {len(images)} images..."):
                for img_path in images:
                    if clf_id is not None:
                        pred, conf = _predict_path(clf_id, str(img_path))
                    else:
                        pred, conf = "(no model)", 0.0
                    records.append({
                        "path": img_path,
                        "name": img_path.name,
                        "pred": pred,
                        "conf": conf,
                        "disagree": pred != target_class and pred not in ("(no model)", "(unreadable)"),
                    })

            if sort_mode == "disagreements first":
                records.sort(key=lambda r: (not r["disagree"], r["conf"]))
            elif sort_mode == "lowest classifier confidence":
                records.sort(key=lambda r: r["conf"])
            else:
                records.sort(key=lambda r: r["name"])

            total_pages = max(1, (len(records) + page_size - 1) // page_size)
            page = st.number_input(
                "Page", min_value=1, max_value=total_pages, value=1, key="td_review_page",
            )
            page_start = (page - 1) * page_size
            page_end = min(page_start + page_size, len(records))
            visible = records[page_start:page_end]

            n_disagree = sum(1 for r in records if r["disagree"])
            st.markdown(
                f"**`{target_class}`** — {len(records)} images, "
                f"**{n_disagree}** disagree with classifier "
                f"(showing {page_start + 1}–{page_end})"
            )

            if "review_actions" not in st.session_state:
                st.session_state.review_actions = {}

            per_row = 4
            for row_start in range(0, len(visible), per_row):
                cols = st.columns(per_row)
                for col, rec in zip(cols, visible[row_start:row_start + per_row]):
                    with col:
                        try:
                            st.image(str(rec["path"]), use_container_width=True)
                        except Exception as e:
                            st.error(str(e))
                            continue

                        if rec["pred"] in ("(no model)", "(unreadable)"):
                            st.write(f"`{rec['name']}`")
                        else:
                            badge = "✗" if rec["disagree"] else "✓"
                            color = "red" if rec["disagree"] else "green"
                            st.markdown(
                                f"`{rec['name']}` — :{color}[{badge} **{rec['pred']}** {rec['conf']:.0%}]"
                            )

                        current = st.session_state.review_actions.get(str(rec["path"]), "Keep")
                        choice = st.selectbox(
                            "Action",
                            options=ACTION_CHOICES,
                            index=ACTION_CHOICES.index(current) if current in ACTION_CHOICES else 0,
                            key=f"td_act_{rec['path']}",
                            label_visibility="collapsed",
                        )
                        st.session_state.review_actions[str(rec["path"])] = choice

            st.divider()

            delete_paths = []
            move_pairs = []
            for path_str, action in st.session_state.review_actions.items():
                if action == "Keep":
                    continue
                src = Path(path_str)
                if not src.exists():
                    continue
                if action == "Delete (false positive)":
                    delete_paths.append(src)
                elif action.startswith("Move to "):
                    tgt = action[len("Move to "):]
                    if tgt != target_class:
                        move_pairs.append((src, tgt))

            n_pending = len(delete_paths) + len(move_pairs)
            if n_pending == 0:
                st.caption("No pending changes.")
            else:
                st.markdown(
                    f"**{n_pending} pending changes** "
                    f"(delete: {len(delete_paths)}, move: {len(move_pairs)})"
                )
                if st.button("Apply", type="primary", key="td_review_apply"):
                    deleted = moved = 0
                    for src in delete_paths:
                        try:
                            src.unlink()
                            deleted += 1
                        except Exception:
                            pass
                    for src, tgt in move_pairs:
                        tgt_dir = DATA_ROOT / tgt
                        tgt_dir.mkdir(parents=True, exist_ok=True)
                        stem, ext = src.stem, src.suffix
                        dst = tgt_dir / src.name
                        n = 1
                        while dst.exists():
                            dst = tgt_dir / f"{stem}_dup{n}{ext}"
                            n += 1
                        try:
                            shutil.move(str(src), str(dst))
                            moved += 1
                        except Exception:
                            pass
                    st.success(f"Applied — deleted {deleted}, moved {moved}.")
                    st.session_state.review_actions = {}
                    _predict_path.clear()
                    st.rerun()

# ===========================================================================
# Tab 3: Train
# ===========================================================================

with tab_train:
    st.markdown("### Fine-tune the classifier")
    st.caption(
        "Trains a ResNet-18 (pretrained on ImageNet) on the labeled folders "
        "under `data/labeled/embedder/<CLASS>/`. Trained weights save to "
        "`models/connector_classifier/` and are picked up by the `/predict` "
        "endpoint that the demo and the AR app use."
    )

    total_labeled = sum(counts.values())
    if total_labeled == 0:
        st.warning("No labeled images yet. Upload + label some videos first.")

    col1, col2, col3, col4 = st.columns(4)
    epochs = col1.number_input("Epochs", min_value=1, max_value=50, value=8, key="td_epochs")
    batch_size = col2.number_input("Batch size", min_value=2, max_value=64, value=16, key="td_bs")
    learning_rate = col3.number_input(
        "Learning rate", min_value=1e-5, max_value=1e-2, value=3e-4,
        format="%.5f", step=1e-4, key="td_lr",
    )
    val_fraction = col4.slider(
        "Val split", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="td_val",
    )

    model_dir = Path(st.text_input(
        "Model output dir", value=str(DEFAULT_MODEL_DIR), key="td_modeldir",
    ))

    if st.button("Train", type="primary", disabled=total_labeled == 0, key="td_train"):
        config = TrainConfig(
            data_dir=DATA_ROOT,
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
            original_print = print
            history_lines: list[str] = []

            def _capture_print(*args, **kwargs):
                line = " ".join(str(a) for a in args)
                history_lines.append(line)
                status.code("\n".join(history_lines))
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
            last = metrics["history"][-1]
            st.write(
                f"- train_acc: {last['train_acc']:.3f}  |  val_acc: {last['val_acc']:.3f}\n"
                f"- train_loss: {last['train_loss']:.3f}  |  val_loss: {last['val_loss']:.3f}"
            )
            # Bust prediction cache so the Review tab uses the fresh model.
            _predict_path.clear()
            _load_classifier.clear()
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.divider()
    st.markdown("### Test on a sample image")

    if not (model_dir / "weights.pt").exists():
        st.info(f"Train a model first (no weights at `{model_dir}/weights.pt`).")
    else:
        sample = st.file_uploader(
            "Upload an image to classify", type=["jpg", "jpeg", "png", "webp"],
            key="td_sample",
        )
        if sample is not None:
            nparr = np.frombuffer(sample.getvalue(), np.uint8)
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
