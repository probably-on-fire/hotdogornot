"""
Google Custom Search JSON API image fetcher.

Reasons to use this over the DDG fetcher:
  - Stable, won't break when Google rotates their HTML
  - Lets us hit Google's actual image index instead of DDG's mirror
  - 100 free queries/day (=1000 free images/day at max 10 results/page)

Setup (one-time, ~10 min):
  1. Create a Custom Search Engine at https://programmablesearchengine.google.com
       - "What to search": Search the entire web
       - "Image search": ON
     Copy the "Search engine ID" — this is GOOGLE_CSE_ID (looks like
     "017576662512468239146:omuauf_lfve").
  2. Get an API key at https://console.cloud.google.com/apis/credentials
       - Create a project if needed
       - Enable the "Custom Search API" for that project
       - Create credentials → API key
     Copy the key — this is GOOGLE_API_KEY.
  3. Put both in `training/.env`:
        GOOGLE_CSE_ID=...
        GOOGLE_API_KEY=...
     or export them as environment variables.

Public surface:

    fetch_images(query, target_dir, n=20) -> int

Returns the number of new images saved after dedup. Raises
`MissingCredentialsError` with a setup hint if the env vars aren't found.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image


REPO_TRAINING = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_TRAINING / ".env"

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_MAX_RESULTS_PER_QUERY = 10  # Google CSE caps at 10 per request
GOOGLE_MAX_TOTAL_RESULTS = 100     # Google CSE caps at 100 across pagination


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class MissingCredentialsError(RuntimeError):
    """Raised when GOOGLE_API_KEY or GOOGLE_CSE_ID isn't configured."""


def _load_credentials() -> tuple[str, str]:
    """Look up Google API key + CSE ID from env or training/.env file."""
    # Lazy load .env so we don't require python-dotenv unless this is called.
    if ENV_FILE.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_FILE)
        except ImportError:
            pass
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    cse_id = os.environ.get("GOOGLE_CSE_ID", "").strip()
    missing = [n for n, v in (("GOOGLE_API_KEY", api_key),
                              ("GOOGLE_CSE_ID", cse_id)) if not v]
    if missing:
        raise MissingCredentialsError(
            f"Missing {', '.join(missing)}. Add them to {ENV_FILE} or your "
            f"environment. See setup steps in google_cse.py docstring."
        )
    return api_key, cse_id


def _existing_hashes(target_dir: Path) -> set[str]:
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
    if not target_dir.is_dir():
        return 0
    max_idx = -1
    for p in target_dir.iterdir():
        stem = p.stem
        if stem.startswith("gcse_") and stem[5:].isdigit():
            max_idx = max(max_idx, int(stem[5:]))
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
    if len(data) < min_kb * 1024:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if min(img.size) < 80:
        return None
    fmt = (img.format or "JPEG").lower()
    ext = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp"}.get(fmt, "jpg")
    return img, ext


def _query_page(api_key: str, cse_id: str, query: str, start: int) -> list[dict]:
    """Hit Google CSE for one page (up to 10 image results) starting at `start`."""
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "searchType": "image",
        "num": GOOGLE_MAX_RESULTS_PER_QUERY,
        "start": start,
        "safe": "off",
    }
    r = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=15)
    if r.status_code == 429:
        raise RuntimeError("Google CSE quota exceeded (429). Free tier is 100/day.")
    if r.status_code == 403:
        raise RuntimeError(
            "Google CSE returned 403 — check API key, that the Custom Search "
            "API is enabled in your Cloud project, and that the CSE has image "
            "search enabled."
        )
    if r.status_code != 200:
        raise RuntimeError(f"Google CSE returned {r.status_code}: {r.text[:200]}")
    return r.json().get("items", [])


def fetch_images(
    query: str,
    target_dir: Path,
    n: int = 20,
    min_kb: int = 4,
) -> int:
    """
    Fetch up to `n` images for `query` from Google Custom Search.

    Pages through Google CSE 10 at a time (its max page size) up to 100
    results (its hard cap). Each page costs 1 query against the daily quota.

    - Skips files smaller than `min_kb`
    - Skips images smaller than 80 px on the short side
    - Dedups against md5 of files already in `target_dir`
    - Names new files `gcse_<index>.<ext>` continuing from highest existing
      index so repeated calls accumulate.
    """
    api_key, cse_id = _load_credentials()
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_hashes(target_dir)
    index = _next_index(target_dir)

    saved = 0
    start = 1  # Google CSE uses 1-based indexing

    while saved < n and start <= GOOGLE_MAX_TOTAL_RESULTS:
        try:
            items = _query_page(api_key, cse_id, query, start)
        except Exception:
            # Surface the error to the caller (Streamlit page) so they can show it.
            raise
        if not items:
            break
        for item in items:
            if saved >= n:
                break
            url = item.get("link") or item.get("image", {}).get("thumbnailLink")
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
            out = target_dir / f"gcse_{index:04d}.{ext}"
            try:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(out)
            except Exception:
                continue
            existing.add(md5)
            index += 1
            saved += 1
            time.sleep(0.05)
        start += GOOGLE_MAX_RESULTS_PER_QUERY
    return saved
