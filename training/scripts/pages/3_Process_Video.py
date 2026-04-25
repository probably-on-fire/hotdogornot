"""
Streamlit page for ingesting capture videos into the labeled dataset.

Workflow:
  1. Upload a video (or point to a path on disk)
  2. Pick the connector class
  3. Choose frames-per-second to extract (default 2)
  4. Hit "Extract" — frames land in data/labeled/embedder/<CLASS>/ as
     video_NNNN.jpg, accumulating across multiple uploads.
  5. (Optional) Run the multi-frame averaged predictor on the freshly
     extracted frames to get a single high-confidence class prediction
     with per-class vote breakdown.

Once frames are saved they show up in the existing Manage Data page just
like Bing/DDG/Google fetched images, with the same browse / prune / "Run
pipeline" buttons.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from rfconnectorai.data_fetch.video_frames import extract_frames
from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ensemble_averager import average_ensemble
from rfconnectorai.measurement.frame_averager import average_predictions


REPO_TRAINING = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_TRAINING / "data" / "labeled" / "embedder"
DEFAULT_MODEL_DIR = REPO_TRAINING / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]


def _count_video_frames(cls: str) -> int:
    d = DATA_ROOT / cls
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.stem.startswith("video_"))


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Process capture video", layout="wide")
st.title("Process capture video")
st.caption(
    "Drop a connector capture video here. Frames extract into "
    "`training/data/labeled/embedder/<CLASS>/` as `video_NNNN.jpg`, ready for "
    "the multi-frame averager and the Manage Data eval."
)

with st.sidebar:
    st.markdown("### Existing video frames per class")
    for cls in CANONICAL_CLASSES:
        st.write(f"- **{cls}**: {_count_video_frames(cls)} video frames")

col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded = st.file_uploader(
        "Upload a video (mp4 / mov / avi / mkv)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )
    target_class = st.selectbox(
        "Connector class shown in this video",
        CANONICAL_CLASSES,
        index=0,
    )
    fps_target = st.number_input(
        "Frames per second to extract",
        min_value=0.5, max_value=15.0, value=2.0, step=0.5,
        help="2 fps is plenty for a 30s clip (60 frames). Bump higher for very short clips.",
    )
    max_frames = st.number_input(
        "Cap total frames (0 = no cap)",
        min_value=0, max_value=2000, value=0, step=10,
    )

with col_right:
    st.markdown("**Tips**")
    st.markdown(
        "- Hold the camera roughly perpendicular to the connector face\n"
        "- Place the printed 25mm ArUco marker in the same shot for absolute scale\n"
        "- 20–60 frames per class is a good starting target\n"
        "- The averager will MAD-filter outlier frames automatically"
    )

if uploaded is not None and st.button("Extract frames", type="primary"):
    target_dir = DATA_ROOT / target_class
    # Persist uploaded video to a temp file so OpenCV can open it.
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner(f"Extracting frames at {fps_target} fps…"):
            saved = extract_frames(
                video_path=tmp_path,
                out_dir=target_dir,
                fps_target=float(fps_target),
                max_frames=int(max_frames) if max_frames > 0 else None,
            )
        st.success(
            f"Extracted {len(saved)} frames into `data/labeled/embedder/{target_class}/`"
        )
        # Show a strip of thumbnails so user can sanity-check what came out.
        if saved:
            st.markdown("**Sample frames** (first 6)")
            cols = st.columns(min(6, len(saved)))
            for col, p in zip(cols, saved[:6]):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img, caption=p.name, use_container_width=True)

        # Stash the saved-paths in session_state so the user can run the
        # averager against them in the next block.
        st.session_state["last_extracted"] = [str(p) for p in saved]
        st.session_state["last_class"] = target_class
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

st.divider()
st.markdown("### Run multi-frame averaged prediction")

last_extracted = st.session_state.get("last_extracted", [])
last_class = st.session_state.get("last_class", None)

if not last_extracted:
    st.info(
        "Extract a video above first, or use the Manage Data page to run the "
        "single-frame pipeline against any class folder."
    )
else:
    st.write(
        f"Ready to average over {len(last_extracted)} frames most recently "
        f"extracted (labeled `{last_class}`)."
    )
    require_aruco = st.checkbox(
        "Require ArUco marker in every frame",
        value=False,
        help=(
            "Strict accuracy mode — disambiguates 2.92mm vs 3.5mm vs 2.4mm. "
            "Frames without a visible marker are dropped. Leave off for a "
            "first eyeball pass on video that doesn't have a marker yet."
        ),
    )

    classifier_available = (DEFAULT_MODEL_DIR / "weights.pt").exists()
    use_ensemble = st.checkbox(
        "Use ensemble (measurement + classifier)",
        value=classifier_available,
        disabled=not classifier_available,
        help=(
            "Uses the trained ResNet-18 classifier alongside the measurement "
            "pipeline and averages classifier softmax across frames. Requires "
            "a trained model at models/connector_classifier/."
        ),
    )

    if st.button("Run averaged prediction"):
        with st.spinner(f"Predicting across {len(last_extracted)} frames…"):
            frames = []
            for fp in last_extracted:
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if use_ensemble and classifier_available:
                predictor = EnsemblePredictor.load(DEFAULT_MODEL_DIR)
                ensemble = average_ensemble(
                    frames, predictor, require_aruco=require_aruco,
                )
                # Normalize the two result types so the rendering code below
                # works for either path.
                result = type(
                    "AvgShim", (),
                    {
                        "class_name": ensemble.class_name,
                        "confidence": ensemble.confidence,
                        "n_frames_total": ensemble.n_frames_total,
                        "n_frames_used": ensemble.n_frames_used,
                        "aperture_mm": ensemble.aperture_mm,
                        "aperture_mm_stddev": ensemble.aperture_mm_stddev,
                        "hex_flat_to_flat_mm": ensemble.hex_flat_to_flat_mm,
                        "pixels_per_mm": ensemble.pixels_per_mm,
                        "family": None,
                        "gender": None,
                        "per_class_votes": ensemble.per_class_votes,
                        "reason": ensemble.reason,
                        "_classifier_probabilities": ensemble.classifier_probabilities,
                        "_per_frame_agreement": ensemble.per_frame_agreement,
                    },
                )
            else:
                result = average_predictions(frames, require_aruco=require_aruco)

        if result.class_name == "Unknown":
            st.error(f"**Result: Unknown** — {result.reason}")
        else:
            agree = "match" if result.class_name == last_class else "mismatch"
            st.success(
                f"**{result.class_name}** ({agree} with label `{last_class}`) — "
                f"confidence {result.confidence:.0%}, "
                f"used {result.n_frames_used}/{result.n_frames_total} frames"
            )

        rows = []
        if result.aperture_mm is not None:
            stddev_str = (
                f" ± {result.aperture_mm_stddev:.3f}"
                if result.aperture_mm_stddev is not None else ""
            )
            rows.append(("Aperture (averaged)", f"{result.aperture_mm:.3f}{stddev_str} mm"))
        if result.hex_flat_to_flat_mm is not None:
            rows.append(("Hex flat-to-flat (averaged)", f"{result.hex_flat_to_flat_mm:.3f} mm"))
        if result.pixels_per_mm is not None:
            rows.append(("Pixels per mm", f"{result.pixels_per_mm:.1f}"))
        if result.family is not None:
            rows.append(("Family (vote)", result.family))
        if result.gender is not None:
            rows.append(("Gender (vote)", result.gender))
        for k, v in rows:
            st.write(f"**{k}:** {v}")

        if result.per_class_votes:
            st.markdown("**Per-class vote breakdown** (hard votes from each frame)")
            for cls_name, n in sorted(
                result.per_class_votes.items(), key=lambda kv: -kv[1]
            ):
                st.write(f"- {cls_name}: {n}")

        clf_probs = getattr(result, "_classifier_probabilities", None) or {}
        if clf_probs:
            st.markdown("**Averaged classifier softmax** (soft votes)")
            for cls_name, prob in sorted(clf_probs.items(), key=lambda kv: -kv[1]):
                st.write(f"- {cls_name}: {prob:.3f}")

        agreement = getattr(result, "_per_frame_agreement", None) or {}
        if agreement:
            st.markdown("**Per-frame agreement signal**")
            for kind, n in sorted(agreement.items(), key=lambda kv: -kv[1]):
                st.write(f"- {kind}: {n}")
