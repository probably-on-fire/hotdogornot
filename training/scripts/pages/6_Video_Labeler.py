"""
Video Labeler — extract connector crops from a video, label each one,
save to the labeled training dataset.

Workflow:
  1. Upload a video (or pick one from a directory)
  2. Set extraction fps (default 2 — gives ~10 frames per 5s clip)
  3. Click "Extract & detect" → runs blob detection, gets ~1-3 crops per frame
  4. Per crop: pick a class from the dropdown (or skip)
  5. Click "Save labeled crops" → writes them to data/labeled/embedder/<CLASS>/

This replaces the back-and-forth chat process where you'd have to describe
"connector A is at lower-left in 2_4mm.MOV". The labeler shows each detected
connector zoomed in, you tap a class, done.

State is stashed in st.session_state so re-running scripts during labeling
doesn't lose progress.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st

from rfconnectorai.data_fetch.connector_crops import detect_connector_crops


REPO_TRAINING = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_TRAINING / "data" / "labeled" / "embedder"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]
CHOICES = ["(skip)"] + CANONICAL_CLASSES


def _extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Path]:
    """Use the bundled ffmpeg to extract frames at the target fps."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out_pat = str(out_dir / "frame_%04d.jpg")
    subprocess.run(
        [ff, "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "4", out_pat],
        capture_output=True, check=False,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def _next_video_idx(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    indices = []
    for p in class_dir.glob("video_*.jpg"):
        stem = p.stem
        if stem.startswith("video_"):
            tail = stem[len("video_"):]
            if tail.isdigit():
                indices.append(int(tail))
    return max(indices) + 1 if indices else 0


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Video Labeler", layout="wide")
st.title("Video Labeler")
st.caption(
    "Drop a connector video here, the labeler auto-detects each connector "
    "in each frame, you pick the class for each crop, and the labeled crops "
    "land in `data/labeled/embedder/<CLASS>/` ready for training."
)

# Step 1: upload video
uploaded = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    key="lbl_video",
)

col1, col2 = st.columns(2)
fps = col1.number_input(
    "Extraction fps", min_value=0.5, max_value=15.0, value=2.0, step=0.5,
    help="Higher fps = more frames = more data, but more clicks. 2 fps is a good default.",
)
max_crops = col2.number_input(
    "Max connectors per frame", min_value=1, max_value=6, value=3,
    help="If a frame has more than this many bright blobs, only the largest ones are kept.",
)

# Hold extracted crops in session_state so the user can label them
# across reruns without losing progress.
if "lbl_crops" not in st.session_state:
    st.session_state.lbl_crops = []   # list of (frame_idx, crop_idx, jpeg_bytes)
    st.session_state.lbl_labels = {}  # (frame_idx, crop_idx) -> class string or "(skip)"

if uploaded is not None and st.button("Extract & detect", type="primary"):
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

        # Run blob detection per frame and collect crops in memory.
        # We encode them to JPEG bytes so they survive in session_state.
        with st.spinner("Detecting connectors…"):
            crops_in_session = []
            for frame_idx, fp in enumerate(frames):
                bgr = cv2.imread(str(fp))
                if bgr is None: continue
                results = detect_connector_crops(bgr, max_crops=int(max_crops))
                for crop_idx, r in enumerate(results):
                    ok, buf = cv2.imencode(".jpg", r.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok: continue
                    crops_in_session.append((frame_idx, crop_idx, buf.tobytes()))
        st.session_state.lbl_crops = crops_in_session
        st.session_state.lbl_labels = {}
        st.success(f"Detected {len(crops_in_session)} connector crops across {len(frames)} frames")

# ---- Labeling UI ---------------------------------------------------------

crops = st.session_state.lbl_crops
if crops:
    st.divider()
    st.markdown(f"### Label crops ({len(crops)} total)")
    st.caption(
        "Pick a class for each crop. Use **(skip)** for false positives or "
        "things that aren't connectors. Skipped crops won't be saved."
    )

    # Show all crops in a grid, 4 per row
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
                    options=CHOICES,
                    index=CHOICES.index(current) if current in CHOICES else 0,
                    key=f"select_{frame_idx}_{crop_idx}",
                    label_visibility="collapsed",
                )
                st.session_state.lbl_labels[key] = choice

    st.divider()
    st.markdown("### Save")

    # Tally
    counts = {}
    for k, v in st.session_state.lbl_labels.items():
        counts[v] = counts.get(v, 0) + 1
    summary_parts = [f"**{v}** {k}" for k, v in sorted(counts.items())]
    st.write(", ".join(summary_parts) if summary_parts else "(no choices yet)")

    if st.button("Save labeled crops", type="primary", disabled=len(crops) == 0):
        saved = 0
        for (frame_idx, crop_idx, jpeg_bytes) in crops:
            cls = st.session_state.lbl_labels.get((frame_idx, crop_idx), "(skip)")
            if cls == "(skip)" or cls not in CANONICAL_CLASSES:
                continue
            cls_dir = DATA_ROOT / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            idx = _next_video_idx(cls_dir)
            out = cls_dir / f"video_{idx:04d}.jpg"
            out.write_bytes(jpeg_bytes)
            saved += 1
        if saved == 0:
            st.warning("No crops marked with a class — nothing saved.")
        else:
            st.success(f"Saved {saved} crops into `data/labeled/embedder/`")
            # Clear session so the user can label another video without
            # double-saving.
            st.session_state.lbl_crops = []
            st.session_state.lbl_labels = {}
            st.balloons()

# ---- Footer / counts ----------------------------------------------------

st.divider()
st.markdown("### Current dataset")
total = 0
for cls in CANONICAL_CLASSES:
    d = DATA_ROOT / cls
    n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
    total += n
    st.write(f"- **{cls}**: {n}")
st.write(f"_Total: {total} images_")
