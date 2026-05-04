"""
Salvage existing training crops by running Hough on them again to
extract tight sub-crops around the actual connector(s). Many "training
crops" in the existing dataset are 1080x1500 wide bbox shots where the
connector is a small fraction of the image — Hough run on those wide
crops finds the connector and produces a tight sub-crop that's
actually useful as training data.

Filters out:
  - Sub-crops that are >= max_size_frac * parent size (likely a
    re-detection of the same wide pattern, not a tight zoom)
  - Sub-crops with fg_fraction below `--min-fg`

Saves passing sub-crops to the same class folder as `tight_NNNNNN.jpg`,
indexed sequentially. The original wide crops are NOT removed unless
--remove-original-on-success is passed.

Usage:
    python scripts/zoom_recrop_training.py \\
        --data-dir data/labeled/embedder \\
        --min-fg 0.04 \\
        --max-size-frac 0.7
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

from rembg import new_session, remove

from rfconnectorai.data_fetch.connector_crops import (
    detect_connector_crops_hough,
)


VARIANT_SUFFIXES = (
    "_clean", "_mask", "_central", "_centralv2", "_z70", "_z50",
) + tuple(f"_bg{i}" for i in range(10))


def _is_variant(p: Path) -> bool:
    return any(p.stem.endswith(s) for s in VARIANT_SUFFIXES)


def _is_already_tight(p: Path) -> bool:
    return p.stem.startswith("tight_")


def _next_tight_idx(class_dir: Path) -> int:
    """Find the next available tight_NNNNNN index in this class folder."""
    nums = []
    for p in class_dir.glob("tight_*.jpg"):
        tail = p.stem[len("tight_"):]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums) + 1 if nums else 0


def fg_metrics(bgr, session) -> tuple[float, float]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=session)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return 0.0, 0.0
    a = rgba[:, :, 3]
    h, w = a.shape
    fg = a > 32
    fg_total = int(fg.sum())
    fg_frac = fg_total / float(a.size)
    cy0, cy1 = h // 4, h - h // 4
    cx0, cx1 = w // 4, w - w // 4
    inner = fg[cy0:cy1, cx0:cx1]
    inner_total = int(inner.sum())
    inner_d = inner_total / max(1, inner.size)
    outer_d = (fg_total - inner_total) / max(1, fg.size - inner.size)
    if outer_d > 1e-6:
        ratio = inner_d / outer_d
    else:
        ratio = 10.0 if inner_d > 0 else 0.0
    return fg_frac, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--min-fg", type=float, default=0.04,
                    help="Min foreground fraction to keep a sub-crop.")
    ap.add_argument("--max-size-frac", type=float, default=0.7,
                    help="Sub-crop dimension must be < this fraction "
                         "of parent's shortest side. Filters out Hough "
                         "re-detections that just match the parent.")
    ap.add_argument("--pad-frac", type=float, default=0.35)
    ap.add_argument("--max-crops-per-image", type=int, default=4)
    ap.add_argument("--accumulator-threshold", type=int, default=22,
                    help="Hough param2; lower = more circles fire.")
    ap.add_argument("--remove-original-on-success", action="store_true",
                    help="Move parent (and its rembg-derived variants) "
                         "to a sibling _replaced/ folder when at least "
                         "one tight sub-crop was extracted from it.")
    args = ap.parse_args()

    session = new_session()
    saved = Counter()
    parents_processed = 0
    parents_with_at_least_one_tight = 0

    for cls_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        if cls_dir.name.startswith("_"):
            continue
        bases = [p for p in cls_dir.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                 and not _is_variant(p)
                 and not _is_already_tight(p)]
        if not bases:
            continue
        idx = _next_tight_idx(cls_dir)
        cls_saved = 0
        cls_replaced = 0
        replaced_dir = args.data_dir.parent / "_replaced" / cls_dir.name

        for src in bases:
            parents_processed += 1
            bgr = cv2.imread(str(src))
            if bgr is None:
                continue
            ph, pw = bgr.shape[:2]
            short = min(ph, pw)

            results = detect_connector_crops_hough(
                bgr,
                pad_frac=args.pad_frac,
                max_crops=args.max_crops_per_image,
                accumulator_threshold=args.accumulator_threshold,
            )

            kept_any = False
            for r in results:
                ch, cw = r.crop.shape[:2]
                if min(ch, cw) >= short * args.max_size_frac:
                    continue   # too large, likely a re-detection
                fg_f, ratio = fg_metrics(r.crop, session)
                if fg_f < args.min_fg:
                    continue
                out_path = cls_dir / f"tight_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), r.crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                idx += 1
                cls_saved += 1
                kept_any = True

            if kept_any:
                parents_with_at_least_one_tight += 1
                if args.remove_original_on_success:
                    replaced_dir.mkdir(parents=True, exist_ok=True)
                    # Move parent
                    dst = replaced_dir / src.name
                    if src.exists():
                        src.rename(dst)
                    # Move sibling variants of the parent so they don't
                    # contaminate training as orphan variants.
                    base_stem = src.stem
                    for vsuffix in VARIANT_SUFFIXES:
                        v = cls_dir / f"{base_stem}{vsuffix}{src.suffix}"
                        if v.exists():
                            v.rename(replaced_dir / v.name)
                            cls_replaced += 1

        saved[cls_dir.name] = cls_saved
        print(f"  {cls_dir.name:<10} bases={len(bases)} "
              f"saved_tight={cls_saved} replaced_variants={cls_replaced}")

    print()
    print(f"parents processed: {parents_processed}")
    print(f"parents producing >=1 tight sub-crop: "
          f"{parents_with_at_least_one_tight}")
    print(f"total tight sub-crops saved: {sum(saved.values())}")


if __name__ == "__main__":
    main()
