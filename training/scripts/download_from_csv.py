#!/usr/bin/env python3
"""
Download a curated set of connector images from a CSV.

CSV format (headers required):
    class_name,url
    SMA-M,https://media.digikey.com/Photos/Amphenol%20Connex/901-143.jpg
    SMA-F,https://example.com/another.jpg
    ...

Files are saved to:
    <out-dir>/<class_name>/<sanitized_basename_or_index>.<ext>

Existing files are skipped (idempotent re-runs).

Usage:
    python scripts/download_from_csv.py training/data/sample_images.csv
    python scripts/download_from_csv.py images.csv --out-dir training/data/labeled/embedder
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_DELAY_SECONDS = 0.5
TIMEOUT = 20

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def url_to_filename(url: str, fallback_index: int) -> str:
    """Pick a filename from the URL; fall back to indexed name if no good basename."""
    path = urlparse(url).path
    base = Path(path).name
    if base and any(base.lower().endswith(ext) for ext in VALID_EXTS):
        return sanitize(base)
    # Take the trailing path segment, append .jpg as a default.
    segment = Path(path).stem or f"image_{fallback_index:04d}"
    return sanitize(segment) + ".jpg"


def download_one(url: str, out_path: Path) -> str:
    """Download a single image. Returns one of: 'saved', 'skipped', 'failed:<reason>'."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
            timeout=TIMEOUT,
            stream=True,
        )
    except requests.RequestException as e:
        return f"failed:{type(e).__name__}"

    if resp.status_code != 200:
        return f"failed:HTTP {resp.status_code}"

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return f"failed:Content-Type {content_type!r}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    if out_path.stat().st_size == 0:
        out_path.unlink()
        return "failed:empty body"

    return "saved"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_file", type=Path, help="CSV with columns: class_name, url")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("training/data/labeled/embedder"),
        help="Root directory for downloaded images (default: %(default)s)",
    )
    ap.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip the polite per-request delay (use only for trusted CDNs)",
    )
    args = ap.parse_args()

    if not args.csv_file.is_file():
        print(f"error: CSV not found: {args.csv_file}", file=sys.stderr)
        return 2

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0

    with open(args.csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "class_name" not in reader.fieldnames or "url" not in reader.fieldnames:
            print("error: CSV must have headers 'class_name' and 'url'", file=sys.stderr)
            return 2

        for i, row in enumerate(reader):
            class_name = (row.get("class_name") or "").strip()
            url = (row.get("url") or "").strip()
            if not class_name or not url:
                continue
            if class_name.startswith("#") or url.startswith("#"):
                continue  # allow comment-style lines

            total += 1
            filename = url_to_filename(url, fallback_index=i)
            out_path = args.out_dir / class_name / filename

            outcome = download_one(url, out_path)
            counts[class_name][outcome.split(":", 1)[0]] += 1

            status_marker = {
                "saved": "OK",
                "skipped": "..",
                "failed": "XX",
            }.get(outcome.split(":", 1)[0], "??")

            print(f"  [{status_marker}] {class_name:10s} {out_path.name}  {outcome}")

            if outcome == "saved" and not args.no_delay:
                time.sleep(REQUEST_DELAY_SECONDS)

    # Per-class summary
    print()
    print("=" * 60)
    print(f"Done. {total} URLs processed.")
    print(f"{'class':<12} {'saved':>6} {'skipped':>8} {'failed':>7}")
    print("-" * 60)
    grand = {"saved": 0, "skipped": 0, "failed": 0}
    for cls in sorted(counts):
        c = counts[cls]
        print(f"{cls:<12} {c.get('saved', 0):>6} {c.get('skipped', 0):>8} {c.get('failed', 0):>7}")
        for k in grand:
            grand[k] += c.get(k, 0)
    print("-" * 60)
    print(f"{'TOTAL':<12} {grand['saved']:>6} {grand['skipped']:>8} {grand['failed']:>7}")

    return 0 if grand["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
