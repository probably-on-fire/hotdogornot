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
TEST_HOLDOUT_ROOT = REPO / "data" / "test_holdout"
VIDEOS_ROOT = REPO / "data" / "videos"
DEFAULT_MODEL_DIR = REPO / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]
LABEL_CHOICES = ["(skip)"] + CANONICAL_CLASSES


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

    saved_videos = sorted(
        [p for p in VIDEOS_ROOT.iterdir() if p.is_file() and p.suffix.lower() in
         (".mp4", ".mov", ".avi", ".mkv", ".webm")]
    ) if VIDEOS_ROOT.is_dir() else []

    if saved_videos:
        source_mode = st.radio(
            "Source",
            options=["Pick a saved video", "Upload a new video"],
            index=0, horizontal=True, key="td_source_mode",
        )
    else:
        source_mode = "Upload a new video"

    chosen_video_path: Path | None = None
    uploaded_bytes: bytes | None = None
    uploaded_name: str = "clip.mp4"

    if source_mode == "Pick a saved video":
        names = [p.name for p in saved_videos]
        sel_name = st.selectbox(
            "Saved videos",
            options=names,
            help=f"Stored in `{VIDEOS_ROOT}` on the server.",
            key="td_saved_video",
        )
        chosen_video_path = next(p for p in saved_videos if p.name == sel_name)
    else:
        uploaded = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="td_video",
        )
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            uploaded_name = uploaded.name

    col1, col2 = st.columns(2)
    fps = col1.number_input(
        "Extraction fps", min_value=0.5, max_value=30.0, value=5.0, step=0.5,
        help="Frames sampled per second of video. Higher = more crops, more clicks.",
        key="td_fps",
    )
    max_crops = col2.number_input(
        "Max connectors per frame", min_value=1, max_value=10, value=5,
        help="Cap on detections per frame.",
        key="td_maxcrops",
    )

    col3, col4 = st.columns(2)
    sensitivity = col3.select_slider(
        "Detector sensitivity",
        options=["Low (3.0σ)", "Normal (2.0σ)", "High (1.5σ)", "Aggressive (1.0σ)"],
        value="Normal (2.0σ)",
        help="Edge-density threshold. Lower = more crops, more false positives.",
        key="td_sens",
    )
    circularity = col4.select_slider(
        "Circularity filter",
        options=["Off", "Loose (0.3)", "Medium (0.5)", "Strict (0.7)"],
        value="Medium (0.5)",
        help="RF connector mating faces are circular; wood-grain artifacts aren't. "
             "Strict cuts most desk false positives; Loose lets through angled views.",
        key="td_circ",
    )
    sens_map = {
        "Low (3.0σ)": 3.0, "Normal (2.0σ)": 2.0,
        "High (1.5σ)": 1.5, "Aggressive (1.0σ)": 1.0,
    }
    circ_map = {
        "Off": 0.0, "Loose (0.3)": 0.3, "Medium (0.5)": 0.5, "Strict (0.7)": 0.7,
    }
    edge_threshold_std = sens_map[sensitivity]
    min_circularity = circ_map[circularity]

    if "lbl_crops" not in st.session_state:
        st.session_state.lbl_crops = []
        st.session_state.lbl_labels = {}

    have_video = chosen_video_path is not None or uploaded_bytes is not None
    if have_video and st.button("Extract & detect", type="primary", key="td_extract"):
        with tempfile.TemporaryDirectory(prefix="rfcai_label_") as tmp:
            tmp_path = Path(tmp)
            if chosen_video_path is not None:
                video_path = chosen_video_path
            else:
                suffix = Path(uploaded_name).suffix or ".mp4"
                video_path = tmp_path / f"clip{suffix}"
                video_path.write_bytes(uploaded_bytes)

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
                    results = detect_connector_crops(
                        bgr,
                        max_crops=int(max_crops),
                        edge_threshold_std=edge_threshold_std,
                        min_circularity=min_circularity,
                    )
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
        "Filter the labeled set down to a subset, see the classifier's "
        "prediction next to each image's on-disk class, and bulk-correct: "
        "keep, delete, or move to a different class. Apply commits the changes."
    )

    # ---- Filters ---------------------------------------------------------

    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        selected_classes = st.multiselect(
            "Classes",
            options=CANONICAL_CLASSES,
            default=CANONICAL_CLASSES,
            format_func=lambda c: f"{c} ({counts.get(c, 0)})",
            key="td_review_classes",
            help="Which class folders to walk. Default is all eight.",
        )
        sa, sb = st.columns(2)
        if sa.button("Select all", use_container_width=True, key="td_review_all"):
            st.session_state.td_review_classes = list(CANONICAL_CLASSES)
            st.rerun()
        if sb.button("Clear", use_container_width=True, key="td_review_none"):
            st.session_state.td_review_classes = []
            st.rerun()
    with fcol2:
        only_disagree = st.checkbox(
            "Only disagreements", value=False, key="td_review_disagree",
            help="Show only images where the classifier predicts a different class than the folder.",
        )
        use_classifier = st.checkbox(
            "Use classifier",
            value=(DEFAULT_MODEL_DIR / "weights.pt").exists(),
            key="td_review_useclf",
            help="Toggle off to skip classifier scoring (faster, but no disagreement signal).",
        )
    with fcol3:
        conf_lo, conf_hi = st.slider(
            "Confidence range", min_value=0, max_value=100, value=(0, 100),
            step=5, key="td_review_confband",
            help="Narrow to e.g. 0–60% to find low-confidence cases worth a human look.",
        )

    scol1, scol2 = st.columns([1, 1])
    sort_mode = scol1.selectbox(
        "Sort by",
        options=[
            "disagreements first",
            "lowest classifier confidence",
            "class then filename",
        ],
        index=0,
        key="td_review_sort",
    )
    page_size = scol2.number_input(
        "Per page", min_value=8, max_value=128, value=32, step=8, key="td_review_pagesize",
    )

    if not selected_classes:
        st.info("Pick at least one class above.")
    else:
        # ---- Build records across all selected classes ------------------

        clf_id = str(DEFAULT_MODEL_DIR) if use_classifier else None
        all_records = []
        total_in_scope = sum(counts.get(c, 0) for c in selected_classes)
        with st.spinner(f"Scoring {total_in_scope} images across {len(selected_classes)} classes..."):
            for cls in selected_classes:
                cls_dir = DATA_ROOT / cls
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(p for p in cls_dir.iterdir() if p.is_file()):
                    if clf_id is not None:
                        pred, conf = _predict_path(clf_id, str(img_path))
                    else:
                        pred, conf = "(no model)", 0.0
                    disagree = pred != cls and pred not in ("(no model)", "(unreadable)")
                    all_records.append({
                        "path": img_path,
                        "name": img_path.name,
                        "cls": cls,
                        "pred": pred,
                        "conf": conf,
                        "disagree": disagree,
                    })

        # ---- Apply filters ----------------------------------------------

        records = all_records
        if only_disagree:
            records = [r for r in records if r["disagree"]]
        if use_classifier and (conf_lo > 0 or conf_hi < 100):
            lo, hi = conf_lo / 100.0, conf_hi / 100.0
            records = [r for r in records if lo <= r["conf"] <= hi]

        if sort_mode == "disagreements first":
            records.sort(key=lambda r: (not r["disagree"], r["conf"]))
        elif sort_mode == "lowest classifier confidence":
            records.sort(key=lambda r: r["conf"])
        else:
            records.sort(key=lambda r: (r["cls"], r["name"]))

        n_total = len(all_records)
        n_visible = len(records)
        n_disagree = sum(1 for r in all_records if r["disagree"])
        st.markdown(
            f"**{n_visible}** of {n_total} images match — "
            f"{n_disagree} disagree with classifier across the selected classes."
        )

        if n_visible == 0:
            st.info("No images match the current filters.")
        else:
            # Pagination
            total_pages = max(1, (n_visible + page_size - 1) // page_size)
            page = st.number_input(
                "Page", min_value=1, max_value=total_pages, value=1, key="td_review_page",
            )
            page_start = (page - 1) * page_size
            page_end = min(page_start + page_size, n_visible)
            visible = records[page_start:page_end]
            st.caption(f"Showing {page_start + 1}–{page_end} of {n_visible}")

            # Grid: each tile has Delete and Flip-gender buttons that act
            # immediately. No staging, no Apply step.
            per_row = 4
            for row_start in range(0, len(visible), per_row):
                cols = st.columns(per_row)
                for col, rec in zip(cols, visible[row_start:row_start + per_row]):
                    path_str = str(rec["path"])
                    with col:
                        try:
                            st.image(path_str, use_container_width=True)
                        except Exception as e:
                            st.error(str(e))
                            continue

                        st.markdown(f"`{rec['name']}` · in :blue[**{rec['cls']}**]")
                        if rec["pred"] not in ("(no model)", "(unreadable)"):
                            badge = "✗" if rec["disagree"] else "✓"
                            color = "red" if rec["disagree"] else "green"
                            st.markdown(
                                f":{color}[{badge} classifier: **{rec['pred']}** {rec['conf']:.0%}]"
                            )

                        bc1, bc2 = st.columns(2)
                        if bc1.button(
                            "✗ Delete", key=f"td_del_{path_str}",
                            type="primary", use_container_width=True,
                        ):
                            try:
                                rec["path"].unlink()
                            except Exception as e:
                                st.error(f"Couldn't delete: {e}")
                            _predict_path.clear()
                            st.rerun()

                        family, gender = rec["cls"].rsplit("-", 1)
                        new_cls = f"{family}-{'F' if gender == 'M' else 'M'}"
                        flip_label = f"⇄ → {new_cls.rsplit('-', 1)[1]}"
                        if bc2.button(
                            flip_label, key=f"td_flip_{path_str}",
                            type="primary", use_container_width=True,
                            disabled=new_cls not in CANONICAL_CLASSES,
                            help=f"Move to {new_cls}",
                        ):
                            tgt_dir = DATA_ROOT / new_cls
                            tgt_dir.mkdir(parents=True, exist_ok=True)
                            stem, ext = rec["path"].stem, rec["path"].suffix
                            dst = tgt_dir / rec["path"].name
                            n = 1
                            while dst.exists():
                                dst = tgt_dir / f"{stem}_dup{n}{ext}"
                                n += 1
                            try:
                                shutil.move(str(rec["path"]), str(dst))
                            except Exception as e:
                                st.error(f"Couldn't flip: {e}")
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
    st.markdown("### Evaluate on held-out test set")
    st.caption(
        "Runs the current classifier against every image in "
        "`data/test_holdout/<CLASS>/` and reports accuracy. The held-out "
        "set is hand-verified ground truth — never trained on — so this "
        "is the real measure of classifier quality. Random would be "
        "~12.5% full-class, ~25% family, ~50% gender."
    )

    test_classes_present = sorted(
        d.name for d in TEST_HOLDOUT_ROOT.iterdir()
        if d.is_dir() and d.name in CANONICAL_CLASSES
    ) if TEST_HOLDOUT_ROOT.is_dir() else []
    test_count = sum(
        sum(1 for p in (TEST_HOLDOUT_ROOT / c).iterdir() if p.is_file())
        for c in test_classes_present
    )

    if not (model_dir / "weights.pt").exists():
        st.info(f"Train a model first (no weights at `{model_dir}/weights.pt`).")
    elif test_count == 0:
        st.info(
            f"No images in `{TEST_HOLDOUT_ROOT}`. Drop a few hand-verified "
            "images per class there to enable held-out evaluation."
        )
    else:
        show_misclassified = st.checkbox(
            "Show misclassified images", value=True, key="td_eval_show_miss",
        )

        if st.button("Run evaluation", type="primary", key="td_eval_run"):
            classifier = ConnectorClassifier.load(model_dir)
            results = []  # (truth_class, pred_class, confidence, img_path)
            for cls in test_classes_present:
                cls_dir = TEST_HOLDOUT_ROOT / cls
                for img_path in sorted(p for p in cls_dir.iterdir() if p.is_file()):
                    bgr = cv2.imread(str(img_path))
                    if bgr is None: continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    p = classifier.predict(rgb)
                    results.append((cls, p.class_name, float(p.confidence), img_path))

            if not results:
                st.error("Couldn't read any held-out images.")
            else:
                # Top-line accuracy. Class string is "<family>-<gender>".
                full_correct = sum(1 for t, p, _, _ in results if t == p)
                family_correct = sum(
                    1 for t, p, _, _ in results
                    if t.rsplit("-", 1)[0] == p.rsplit("-", 1)[0]
                )
                gender_correct = sum(
                    1 for t, p, _, _ in results
                    if t.rsplit("-", 1)[1] == p.rsplit("-", 1)[1]
                )
                n = len(results)

                m1, m2, m3 = st.columns(3)
                m1.metric("Full class", f"{full_correct}/{n}", f"{full_correct/n:.0%}")
                m2.metric("Family (mm)", f"{family_correct}/{n}", f"{family_correct/n:.0%}")
                m3.metric("Gender (M/F)", f"{gender_correct}/{n}", f"{gender_correct/n:.0%}")

                # Confusion matrix as a markdown table. Rows = truth,
                # cols = prediction. Only includes classes that appear in
                # the held-out set as truth (usually all 8, but be tolerant).
                truth_classes = sorted(set(t for t, _, _, _ in results))
                pred_classes = CANONICAL_CLASSES
                st.markdown("**Confusion matrix** (rows = truth, cols = prediction)")
                header = "| truth ↓ / pred → | " + " | ".join(pred_classes) + " | total |"
                sep = "|" + "---|" * (len(pred_classes) + 2)
                lines = [header, sep]
                for t in truth_classes:
                    row = [t]
                    truth_total = 0
                    for p in pred_classes:
                        cnt = sum(1 for tt, pp, _, _ in results if tt == t and pp == p)
                        truth_total += cnt
                        if cnt == 0:
                            row.append(".")
                        elif t == p:
                            row.append(f"**{cnt}**")
                        else:
                            row.append(str(cnt))
                    row.append(str(truth_total))
                    lines.append("| " + " | ".join(row) + " |")
                st.markdown("\n".join(lines))

                if show_misclassified:
                    misses = [(t, p, c, ip) for t, p, c, ip in results if t != p]
                    if not misses:
                        st.success("No misclassifications — every held-out image classified correctly.")
                    else:
                        st.markdown(f"**Misclassified** ({len(misses)}/{n}):")
                        per_row = 4
                        for row_start in range(0, len(misses), per_row):
                            cols = st.columns(per_row)
                            for col, (t, p, c, ip) in zip(cols, misses[row_start:row_start + per_row]):
                                with col:
                                    st.image(str(ip), use_container_width=True)
                                    st.markdown(
                                        f"`{ip.name}`  \n"
                                        f"truth: :blue[**{t}**]  \n"
                                        f"pred:  :red[**{p}**] {c:.0%}"
                                    )

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
