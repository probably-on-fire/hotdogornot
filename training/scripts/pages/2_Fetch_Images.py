"""
Streamlit page for fetching connector images from Bing image search.

Replaces the standalone labeler.py — same functionality, lives on the same
port as the demo + data manager. Fetched images land in
training/data/labeled/embedder/<CLASS>/ where the data manager and the
measurement-pipeline eval script can pick them up.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from icrawler.builtin import BingImageCrawler


REPO_TRAINING = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_TRAINING / "data" / "labeled" / "embedder"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

# Default Bing queries per class. The 3.5mm precision RF connector is easily
# confused with audio jacks; queries bias toward "microwave"/"precision".
DEFAULT_QUERIES = {
    "SMA-M":   "SMA male connector RF coaxial plug",
    "SMA-F":   "SMA female connector RF coaxial jack",
    "3.5mm-M": "3.5mm male precision microwave RF connector",
    "3.5mm-F": "3.5mm female precision microwave RF connector",
    "2.92mm-M": "2.92mm K connector male RF microwave",
    "2.92mm-F": "2.92mm K connector female RF microwave",
    "2.4mm-M":  "2.4mm male microwave RF connector precision",
    "2.4mm-F":  "2.4mm female microwave RF connector precision",
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _count(cls: str) -> int:
    d = DATA_ROOT / cls
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def _fetch_one(cls: str, query: str, n: int) -> int:
    target_dir = DATA_ROOT / cls
    target_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": str(target_dir)},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        log_level=30,
    )
    before = _count(cls)
    try:
        crawler.crawl(keyword=query, max_num=n, file_idx_offset="auto")
    except Exception as e:
        st.error(f"{cls}: {e}")
        return 0
    return _count(cls) - before


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fetch training data", layout="wide")
st.title("Fetch training data")
st.caption(
    "Pull candidate connector images from Bing. Images save to "
    "`training/data/labeled/embedder/<CLASS>/` — the data manager page lets you "
    "browse, prune, and run the measurement pipeline against them."
)

with st.sidebar:
    st.markdown("### Per-class counts")
    counts = {cls: _count(cls) for cls in CANONICAL_CLASSES}
    for cls in CANONICAL_CLASSES:
        st.write(f"- **{cls}**: {counts[cls]}")
    st.write(f"_Total: {sum(counts.values())} images_")

per_class_n = st.number_input(
    "Images per class to fetch",
    min_value=5, max_value=100, value=30, step=5,
)

if st.button("Fetch All (across all 8 classes)", type="primary"):
    progress = st.progress(0.0, text="Starting…")
    for i, (cls, query) in enumerate(DEFAULT_QUERIES.items()):
        progress.progress(i / len(DEFAULT_QUERIES), text=f"Fetching {cls} ({query})…")
        added = _fetch_one(cls, query, int(per_class_n))
        st.write(f"  • {cls}: +{added} images")
    progress.progress(1.0, text="Done")
    st.success(f"Fetched ~{int(per_class_n)} images per class")
    st.rerun()

st.divider()
st.markdown("### Per-class fetch")

for cls in CANONICAL_CLASSES:
    with st.container():
        col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
        col1.metric(cls, counts[cls])
        query = col2.text_input(
            f"Query for {cls}",
            value=DEFAULT_QUERIES[cls],
            key=f"q_{cls}",
            label_visibility="collapsed",
        )
        n = col3.number_input(
            f"N for {cls}",
            min_value=5, max_value=100, value=30, step=5,
            key=f"n_{cls}",
            label_visibility="collapsed",
        )
        if col4.button("Fetch", key=f"fetch_{cls}"):
            with st.spinner(f"Fetching {cls}…"):
                added = _fetch_one(cls, query, int(n))
            st.success(f"{cls}: +{added} images")
            st.rerun()
