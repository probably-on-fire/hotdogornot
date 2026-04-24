#!/usr/bin/env python3
"""
Run the hex + aperture measurement pipeline on real connector images and
print per-image results. Use to validate the approach on field-like photos.

Usage:
    python scripts/measure_connector.py <image-or-dir> [--expected SMA-F]
    python scripts/measure_connector.py data/labeled/embedder/2.4mm-F
    python scripts/measure_connector.py data/labeled/embedder --summary
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from rfconnectorai.measurement.class_predictor import predict_class


CLASS_NAMES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]


def measure_one(path: Path, expected: str | None = None) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"path": str(path), "error": "cv2.imread returned None"}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predict_class(img_rgb)

    return {
        "path": str(path.relative_to(path.parent.parent.parent)) if len(path.parents) >= 3 else str(path),
        "expected": expected,
        "predicted": pred.class_name,
        "hex_mm": pred.hex_flat_to_flat_mm,
        "aperture_mm": pred.aperture_mm,
        "ppm": pred.pixels_per_mm,
        "family": pred.family,
        "gender": pred.gender,
        "dielectric_brightness": pred.dielectric_brightness,
        "center_brightness": pred.center_brightness,
        "reason": pred.reason,
        "correct": (expected is None) or (pred.class_name == expected),
    }


def run(target: Path, summary: bool, limit: int | None) -> int:
    results: list[dict] = []

    if target.is_file():
        results.append(measure_one(target))
    elif target.is_dir():
        # Two layouts:
        #   1. <target>/<CLASS>/*.jpg — class-per-subdir, treat subdir as expected label
        #   2. <target>/*.jpg        — flat, no expected label
        subdirs = [d for d in target.iterdir() if d.is_dir() and d.name in CLASS_NAMES]
        if subdirs:
            for class_dir in sorted(subdirs):
                imgs = sorted(
                    [p for p in class_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
                )
                if limit:
                    imgs = imgs[:limit]
                for img_path in imgs:
                    results.append(measure_one(img_path, expected=class_dir.name))
        else:
            imgs = sorted(
                [p for p in target.iterdir()
                 if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            )
            if limit:
                imgs = imgs[:limit]
            for img_path in imgs:
                results.append(measure_one(img_path))
    else:
        print(f"error: {target} is neither a file nor a directory", file=sys.stderr)
        return 2

    if not summary:
        for r in results:
            if "error" in r:
                print(f"  [ERR] {r['path']}: {r['error']}")
                continue
            marker = "OK" if r["correct"] else "XX"
            exp = f" (expected {r['expected']})" if r["expected"] else ""
            meas = ""
            if r["hex_mm"] is not None:
                meas += f" hex={r['hex_mm']:.2f}mm"
            if r["aperture_mm"] is not None:
                meas += f" ap={r['aperture_mm']:.2f}mm"
            if r["family"] is not None:
                meas += f" fam={r['family']}"
            if r["gender"] is not None:
                meas += f" gen={r['gender']}"
            reason = f"  [{r['reason']}]" if r["reason"] else ""
            print(f"  [{marker}] {r['path']:<55s} -> {r['predicted']}{meas}{exp}{reason}")

    # Per-class summary
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        if "error" in r:
            continue
        k = r["expected"] or "?"
        by_class[k]["total"] += 1
        if r["correct"]:
            by_class[k]["correct"] += 1
        if r["predicted"] == "Unknown":
            by_class[k]["unknown"] += 1

    print()
    print("=" * 70)
    print(f"{'class':<12} {'n':>4} {'correct':>8} {'unknown':>8}  {'%':>5}")
    print("-" * 70)
    total_total = 0
    total_correct = 0
    for cls in sorted(by_class):
        c = by_class[cls]
        total = c["total"]
        correct = c["correct"]
        unknown = c["unknown"]
        pct = (correct / total * 100) if total else 0
        print(f"{cls:<12} {total:>4} {correct:>8} {unknown:>8}  {pct:>5.1f}")
        total_total += total
        total_correct += correct
    print("-" * 70)
    pct = (total_correct / total_total * 100) if total_total else 0
    print(f"{'TOTAL':<12} {total_total:>4} {total_correct:>8}             {pct:>5.1f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--summary", action="store_true", help="Only show per-class summary")
    ap.add_argument("--limit", type=int, default=None, help="Max images per class")
    args = ap.parse_args()
    return run(args.target, args.summary, args.limit)


if __name__ == "__main__":
    sys.exit(main())
