"""
DuckDuckGo image-search fetcher.

icrawler's Google + Bing crawlers stopped working reliably (Bing's image
search HTML returns thumbnail URLs that 403 on download; Google's parser
returns nothing). DDG mirrors most of Google's index and lets us hit it
through a maintained Python package (`ddgs`) with no API key.

Public surface:

    fetch_images(query, target_dir, n=20, min_kb=4) -> int

The function returns how many new images actually landed on disk after
download retries and dedup.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _existing_hashes(target_dir: Path) -> set[str]:
    """Return md5 hashes of files already in target_dir for dedup."""
    seen: set[str] = set()
    if not target_dir.is_dir():
        return seen
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        try:
            seen.add(hashlib.md5(p.read_bytes()).hexdigest())
        except OSError:
            continue
    return seen


def _next_index(target_dir: Path) -> int:
    """Find the next free numeric index so we don't clobber existing files."""
    if not target_dir.is_dir():
        return 0
    max_idx = -1
    for p in target_dir.iterdir():
        stem = p.stem
        if stem.startswith("ddg_") and stem[4:].isdigit():
            max_idx = max(max_idx, int(stem[4:]))
    return max_idx + 1


def _try_download(url: str, timeout: float = 8.0) -> bytes | None:
    try:
        r = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if not (ctype.startswith("image/") or "octet-stream" in ctype):
            return None
        data = r.content
        if not data:
            return None
        return data
    except (requests.RequestException, ValueError):
        return None


def _validate_image(data: bytes, min_kb: int) -> tuple[Image.Image, str] | None:
    """Decode + validate. Returns (image, file_extension) or None."""
    if len(data) < min_kb * 1024:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        # Re-open after verify (verify closes the stream).
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if min(img.size) < 80:
        return None
    fmt = (img.format or "JPEG").lower()
    ext = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp"}.get(fmt, "jpg")
    return img, ext


def fetch_images(
    query: str,
    target_dir: Path,
    n: int = 20,
    min_kb: int = 4,
    *,
    safesearch: str = "off",
    region: str = "wt-wt",
) -> int:
    """
    Fetch up to `n` images for `query` from DuckDuckGo Images.

    - Skips files smaller than `min_kb` (filters thumbnails / placeholder svgs).
    - Skips files whose md5 already exists in `target_dir` (dedup).
    - Skips images smaller than 80 px on the short side.
    - Names new files `ddg_<index>.<ext>` continuing from the highest existing
      index, so calling fetch_images multiple times accumulates rather than
      overwriting.

    Imports `ddgs` lazily so the rest of the codebase doesn't take the
    dependency just by importing this module.
    """
    from ddgs import DDGS

    target_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_hashes(target_dir)
    index = _next_index(target_dir)

    # Ask DDG for ~3x as many results as we want — we'll lose some to
    # download failures, dedup, and tiny-image rejects.
    desired = max(n * 3, n + 10)
    saved = 0

    with DDGS() as ddg:
        try:
            results = ddg.images(query, max_results=desired,
                                 safesearch=safesearch, region=region)
        except Exception as e:
            raise RuntimeError(f"DDG query failed: {e}") from e

        for hit in results:
            if saved >= n:
                break
            url = hit.get("image") or hit.get("thumbnail")
            if not url:
                continue
            data = _try_download(url)
            if data is None:
                continue
            md5 = hashlib.md5(data).hexdigest()
            if md5 in existing:
                continue
            valid = _validate_image(data, min_kb=min_kb)
            if valid is None:
                continue
            img, ext = valid
            out = target_dir / f"ddg_{index:04d}.{ext}"
            try:
                # Re-encode through PIL so we strip any weird metadata
                # and end up with a normal RGB file.
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(out)
            except Exception:
                continue
            existing.add(md5)
            index += 1
            saved += 1
            # Be polite — small jitter between downloads.
            time.sleep(0.05)
    return saved
