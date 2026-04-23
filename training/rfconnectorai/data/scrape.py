from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


USER_AGENT = "rfconnectorai-research/0.1 (contact: chris@aired.com)"
REQUEST_DELAY_SECONDS = 1.0


@dataclass
class CatalogImage:
    url: str
    class_name: str
    filename: str


def sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe on common filesystems with underscores."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def save_catalog_image(img: CatalogImage, root: Path) -> Path:
    """
    Download a single image to root/<class_name>/<sanitized_filename>.

    Raises:
        RuntimeError: on non-200 HTTP response.
        ValueError:  on non-image content type.
    """
    class_dir = root / img.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(img.filename)
    out_path = class_dir / safe_name

    resp = requests.get(img.url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {img.url}")

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"Non-image Content-Type {content_type!r} for {img.url}")

    out_path.write_bytes(resp.content)
    time.sleep(REQUEST_DELAY_SECONDS)  # polite rate limiting
    return out_path


def scrape_urls(urls: list[CatalogImage], root: Path) -> list[Path]:
    """Download a list of catalog images. Errors on individual images are logged and skipped."""
    saved: list[Path] = []
    for img in urls:
        try:
            saved.append(save_catalog_image(img, root))
        except (RuntimeError, ValueError) as e:
            print(f"[scrape] skip {img.url}: {e}")
    return saved
