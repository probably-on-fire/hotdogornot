"""
Review Labels — quickly verify training-data accuracy.

Shows every image in `data/labeled/embedder/<CLASS>/` with the current
classifier's prediction for that image alongside its on-disk label. Lets
you bulk-fix mislabeled samples in three actions:

  - Keep    (no change)
  - Move to a different class (corrects mislabels)
  - Delete  (drops false positives — non-connector blobs the auto-detector
            picked up)

Sorting prioritizes images where the classifier disagrees with the label
(those are the most likely to need attention). After making changes, tap
"Apply" to commit the move/deletes; you can then go retrain on the
cleaned-up set.

Designed to make 30-second passes through a class folder feasible — the
critical alternative to a labeler is "I'll just hand-label everything"
which doesn't scale past ~100 images.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from rfconnectorai.classifier.predict import ConnectorClassifier


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data" / "labeled" / "embedder"
DEFAULT_MODEL_DIR = REPO / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]
ACTION_CHOICES = ["Keep", "Delete (false positive)"] + [f"Move to {c}" for c in CANONICAL_CLASSES]


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
    """Cached prediction (key on the model dir mtime + path)."""
    clf = _load_classifier(_clf_id) if _clf_id else None
    if clf is None: return ("(no model)", 0.0)
    bgr = cv2.imread(img_path_str)
    if bgr is None: return ("(unreadable)", 0.0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    p = clf.predict(rgb)
    return (p.class_name, float(p.confidence))


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Review Labels", layout="wide")
st.title("Review training labels")
st.caption(
    "Walk through each image in a class folder. The classifier's prediction "
    "shows alongside the on-disk label — disagreements bubble to the top. "
    "Pick an action per image, then tap Apply to commit moves/deletes."
)

# ---- Sidebar: pick a class to review -------------------------------------

with st.sidebar:
    st.markdown("### Class")
    counts = {}
    for cls in CANONICAL_CLASSES:
        d = DATA_ROOT / cls
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        counts[cls] = n
    cls_options = [f"{c} ({counts.get(c, 0)})" for c in CANONICAL_CLASSES]
    sel = st.radio("Class to review", options=cls_options, index=0)
    target_class = sel.split(" (")[0]

    st.divider()
    st.markdown("### Display")
    sort_mode = st.selectbox(
        "Sort by",
        options=["disagreements first", "lowest classifier confidence", "filename"],
        index=0,
    )
    page_size = st.number_input("Per page", min_value=8, max_value=64, value=24, step=8)
    use_classifier = st.checkbox(
        "Use classifier predictions",
        value=(DEFAULT_MODEL_DIR / "weights.pt").exists(),
    )

# ---- Main area: list + actions ------------------------------------------

class_dir = DATA_ROOT / target_class
if not class_dir.is_dir() or counts.get(target_class, 0) == 0:
    st.info(f"`{target_class}` has no images yet.")
    st.stop()

clf_id = str(DEFAULT_MODEL_DIR) if use_classifier else None

# Build the working list: filename + prediction + confidence + agreement.
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

# Sort
if sort_mode == "disagreements first":
    records.sort(key=lambda r: (not r["disagree"], r["conf"]))
elif sort_mode == "lowest classifier confidence":
    records.sort(key=lambda r: r["conf"])
else:
    records.sort(key=lambda r: r["name"])

# Pagination
total_pages = max(1, (len(records) + page_size - 1) // page_size)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
page_start = (page - 1) * page_size
page_end = min(page_start + page_size, len(records))
visible = records[page_start:page_end]

n_disagree = sum(1 for r in records if r["disagree"])
st.markdown(
    f"### `{target_class}` — {len(records)} images, "
    f"**{n_disagree}** disagree with classifier "
    f"(showing {page_start + 1}–{page_end})"
)

# Actions accumulate in session_state so the user can flip across pages
# without losing prior choices.
if "review_actions" not in st.session_state:
    st.session_state.review_actions = {}

per_row = 4
for row_start in range(0, len(visible), per_row):
    cols = st.columns(per_row)
    for col, rec in zip(cols, visible[row_start:row_start + per_row]):
        with col:
            try:
                # Streamlit can take the file path directly
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
                key=f"act_{rec['path']}",
                label_visibility="collapsed",
            )
            st.session_state.review_actions[str(rec["path"])] = choice

st.divider()

# ---- Apply actions -------------------------------------------------------

# Tally pending changes
delete_paths = []
move_pairs = []  # (src_path, target_class)
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
        if tgt != target_class:   # don't move into self
            move_pairs.append((src, tgt))

n_pending = len(delete_paths) + len(move_pairs)
if n_pending == 0:
    st.caption("No pending changes.")
else:
    st.markdown(f"**{n_pending} pending changes** "
                f"(delete: {len(delete_paths)}, move: {len(move_pairs)})")
    if st.button("Apply", type="primary"):
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
            # Find a fresh filename to avoid clobbering existing samples in target class.
            stem = src.stem
            ext = src.suffix
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
        # Bust prediction cache so re-rendered list is fresh.
        _predict_path.clear()
        st.rerun()

# ---- Footer / dataset summary -------------------------------------------

with st.expander("All-class summary"):
    total = 0
    for cls in CANONICAL_CLASSES:
        n = counts.get(cls, 0)
        total += n
        st.write(f"- **{cls}**: {n}")
    st.write(f"_Total: {total}_")
