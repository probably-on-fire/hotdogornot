"""
Streamlit page for browsing and managing the training/eval data on disk.

Lets you:
  - Pick a data source (labeled, synthetic_faces, synthetic, field_test)
  - See per-class image counts
  - Browse a class's images as a thumbnail grid
  - Delete bad images (one click per image)
  - Run the measurement pipeline against any class to see live accuracy
  - Open the source folder in Explorer for bulk operations
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from rfconnectorai.measurement.class_predictor import predict_class


REPO_TRAINING = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_TRAINING / "data"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _list_data_sources() -> list[Path]:
    """Find subdirectories under data/ that look like training-data roots."""
    sources = []
    if not DATA_ROOT.exists():
        return sources
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        # A "data source" has class subdirectories.
        has_class_dirs = any(
            (entry / c).is_dir() for c in CANONICAL_CLASSES
        )
        if has_class_dirs:
            sources.append(entry)
        # Special case: data/labeled/ may have an embedder/ subdir
        elif entry.name == "labeled":
            for sub in entry.iterdir():
                if sub.is_dir() and any((sub / c).is_dir() for c in CANONICAL_CLASSES):
                    sources.append(sub)
    return sources


def _list_class_files(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        [p for p in class_dir.iterdir()
         if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )


def _open_in_explorer(path: Path) -> None:
    """Open a directory in Windows Explorer."""
    if path.exists():
        try:
            subprocess.run(["explorer", str(path)], check=False)
        except Exception:
            pass


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Manage training data", layout="wide")
st.title("Training data")
st.caption(
    "Browse, prune, and run the measurement pipeline against the on-disk training "
    "and eval sets. Files live under `training/data/`."
)

sources = _list_data_sources()
if not sources:
    st.warning(
        f"No data sources found under `{DATA_ROOT}`. Drop class-named folders "
        f"(e.g. SMA-M, SMA-F, ...) into one of: `data/labeled/embedder/`, "
        f"`data/synthetic_faces/`, `data/synthetic/`, `data/field_test/`."
    )
    st.stop()

with st.sidebar:
    src_label_to_path = {str(s.relative_to(REPO_TRAINING)): s for s in sources}
    chosen_label = st.selectbox("Data source", list(src_label_to_path.keys()))
    src_root = src_label_to_path[chosen_label]
    st.caption(f"Path: `{src_root}`")
    if st.button("Open folder in Explorer"):
        _open_in_explorer(src_root)

    st.divider()
    st.markdown("### Per-class counts")
    counts = {}
    for cls in CANONICAL_CLASSES:
        class_dir = src_root / cls
        n = len(_list_class_files(class_dir))
        counts[cls] = n
        st.write(f"- **{cls}**: {n}")
    st.write(f"_Total: {sum(counts.values())} images_")

    st.divider()
    st.markdown("### Bulk delete")
    confirm_phrase = st.text_input(
        f"Type `WIPE` to enable wiping `{src_root.name}`",
        key=f"wipe_confirm_{src_root.name}",
    )
    if confirm_phrase == "WIPE":
        if st.button(f"Delete ALL images under {src_root.name}", type="primary"):
            removed = 0
            for cls in CANONICAL_CLASSES:
                class_dir = src_root / cls
                for f in _list_class_files(class_dir):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass
            st.success(f"Removed {removed} images from {src_root.name}")
            st.rerun()

st.markdown(f"### Source: `{src_root.relative_to(REPO_TRAINING)}`")

class_tabs = st.tabs(CANONICAL_CLASSES)

for i, cls in enumerate(CANONICAL_CLASSES):
    with class_tabs[i]:
        class_dir = src_root / cls
        files = _list_class_files(class_dir)

        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
        col_a.metric(f"{cls}", f"{len(files)} images")
        if col_b.button("Open folder", key=f"open_{cls}"):
            _open_in_explorer(class_dir)
        run_eval = col_c.button("Run pipeline", key=f"eval_{cls}")
        delete_all = col_d.button(
            "Delete all", key=f"delall_{cls}", help="Remove every image in this class folder"
        )

        if delete_all and files:
            removed = 0
            for f in files:
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass
            st.success(f"Removed {removed} images from {cls}")
            st.rerun()

        if run_eval and files:
            with st.spinner(f"Running pipeline on {len(files)} images..."):
                correct = 0
                unknown_files = []
                wrong = []
                for f in files:
                    img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pred = predict_class(img_rgb)
                    if pred.class_name == cls:
                        correct += 1
                    elif pred.class_name == "Unknown":
                        unknown_files.append(f.name)
                    else:
                        wrong.append((f.name, pred.class_name))
                total = len(files)
                acc = (correct / total * 100) if total else 0
                st.metric("Accuracy", f"{acc:.1f}%", help=f"{correct}/{total} correct")
                col_x, col_y = st.columns(2)
                col_x.write(f"**Unknown**: {len(unknown_files)}")
                col_y.write(f"**Wrong**: {len(wrong)}")
                if wrong:
                    with st.expander(f"{len(wrong)} wrong predictions"):
                        for fn, pred_cls in wrong[:50]:
                            st.write(f"- `{fn}` -> `{pred_cls}`")
                        if st.button(
                            f"Delete all {len(wrong)} wrong-prediction files",
                            key=f"del_wrong_{cls}",
                        ):
                            removed = 0
                            for fn, _ in wrong:
                                try:
                                    (class_dir / fn).unlink()
                                    removed += 1
                                except Exception:
                                    pass
                            st.success(f"Removed {removed} files")
                            st.rerun()
                if unknown_files:
                    with st.expander(f"{len(unknown_files)} Unknown predictions"):
                        for fn in unknown_files[:50]:
                            st.write(f"- `{fn}`")
                        if st.button(
                            f"Delete all {len(unknown_files)} Unknown files",
                            key=f"del_unk_{cls}",
                        ):
                            removed = 0
                            for fn in unknown_files:
                                try:
                                    (class_dir / fn).unlink()
                                    removed += 1
                                except Exception:
                                    pass
                            st.success(f"Removed {removed} files")
                            st.rerun()

        if not files:
            st.info(f"No images in `{class_dir}` yet.")
            continue

        # Thumbnail grid with delete buttons.
        per_row = 5
        for row_start in range(0, len(files), per_row):
            row = files[row_start:row_start + per_row]
            cols = st.columns(per_row)
            for col, f in zip(cols, row):
                with col:
                    try:
                        thumb = Image.open(f)
                        thumb.thumbnail((200, 200))
                        st.image(thumb, caption=f.name, use_container_width=True)
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                        continue
                    if st.button("Delete", key=f"del_{cls}_{f.name}"):
                        try:
                            f.unlink()
                            st.success(f"Deleted {f.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
