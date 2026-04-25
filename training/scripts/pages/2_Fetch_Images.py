"""
Streamlit page for fetching connector images from web image search.

Three source backends:
  - Google CSE (best, requires GOOGLE_API_KEY + GOOGLE_CSE_ID — see
    rfconnectorai/data_fetch/google_cse.py for one-time setup)
  - DuckDuckGo (no key needed; mirrors much of Google's index)
  - Bing via icrawler (legacy fallback; downloader silently 403s on most
    modern image hosts so usually returns 0)

Fetched images land in training/data/labeled/embedder/<CLASS>/ where the data
manager and the measurement-pipeline eval script can pick them up.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from icrawler.builtin import BingImageCrawler

from rfconnectorai.data_fetch.ddg_images import fetch_images as ddg_fetch_images
from rfconnectorai.data_fetch.google_cse import (
    MissingCredentialsError,
    fetch_images as gcse_fetch_images,
)


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
# Queries tuned to mirror what a human types into Google for a vendor product
# photo. Short, parts-style phrasing ("SMA 2.4mm Male") tends to surface clean
# white-background catalog shots; long descriptive queries pull in too much
# noise. Each query stays editable in the UI per-class.
DEFAULT_QUERIES = {
    "SMA-M":    "SMA male connector",
    "SMA-F":    "SMA female connector",
    "3.5mm-M":  "3.5mm male connector RF",
    "3.5mm-F":  "3.5mm female connector RF",
    "2.92mm-M": "2.92mm male connector",
    "2.92mm-F": "2.92mm female connector",
    "2.4mm-M":  "2.4mm male connector",
    "2.4mm-F":  "2.4mm female connector",
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _count(cls: str) -> int:
    d = DATA_ROOT / cls
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def _fetch_one_bing(cls: str, query: str, n: int) -> int:
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


def _fetch_one_ddg(cls: str, query: str, n: int) -> int:
    target_dir = DATA_ROOT / cls
    try:
        return ddg_fetch_images(query=query, target_dir=target_dir, n=n)
    except Exception as e:
        st.error(f"{cls}: {e}")
        return 0


def _fetch_one_gcse(cls: str, query: str, n: int) -> int:
    target_dir = DATA_ROOT / cls
    try:
        return gcse_fetch_images(query=query, target_dir=target_dir, n=n)
    except MissingCredentialsError as e:
        st.error(str(e))
        return 0
    except Exception as e:
        st.error(f"{cls}: {e}")
        return 0


def _fetch_one(cls: str, query: str, n: int, source: str) -> int:
    if source == "Google":
        return _fetch_one_gcse(cls, query, n)
    if source == "DuckDuckGo":
        return _fetch_one_ddg(cls, query, n)
    return _fetch_one_bing(cls, query, n)


# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fetch training data", layout="wide")
st.title("Fetch training data")
st.caption(
    "Pull candidate connector images from Bing. Images save to "
    "`training/data/labeled/embedder/<CLASS>/` — the data manager page lets you "
    "browse, prune, and run the measurement pipeline against them."
)

with st.sidebar:
    st.markdown("### Search source")
    source = st.radio(
        "Image search backend",
        options=["Google", "DuckDuckGo", "Bing"],
        index=0,
        help=(
            "Google needs GOOGLE_API_KEY + GOOGLE_CSE_ID in training/.env "
            "(see rfconnectorai/data_fetch/google_cse.py for setup, ~10 min). "
            "DuckDuckGo works with no setup. Bing is the legacy icrawler "
            "path; usually returns 0 because the downloader 403s."
        ),
    )
    if source == "Google":
        st.caption(
            "First time? See google_cse.py for the API + CSE setup steps."
        )

    st.divider()
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
        added = _fetch_one(cls, query, int(per_class_n), source)
        st.write(f"  • {cls}: +{added} images")
    progress.progress(1.0, text="Done")
    st.success(f"Fetched ~{int(per_class_n)} images per class via {source}")
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
            with st.spinner(f"Fetching {cls} via {source}…"):
                added = _fetch_one(cls, query, int(n), source)
            st.success(f"{cls}: +{added} images")
            st.rerun()
