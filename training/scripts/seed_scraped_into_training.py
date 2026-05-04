"""
Audit scraped (web-image) data with the rembg fg filter and copy
passing images into the labeled-embedder folder so they're picked up
by the next retrain. Use a stricter --min-fg than the training-data
audit because scraped product shots are usually well-framed; if rembg
can't see a salient object, the image is probably text/UI screenshot
or low-quality.

Saves passing images as scraped_NNNNNN.jpg in the corresponding
embedder/<class>/ folder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from rembg import new_session, remove


def fg_metrics(bgr, session):
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


def passes(fg_frac, ratio, min_fg, low_r, high_r, min_uniform_fg):
    if fg_frac < min_fg:
        return False
    return (fg_frac >= min_uniform_fg and ratio <= low_r) or ratio >= high_r


def _next_scraped_idx(class_dir):
    nums = []
    for p in class_dir.glob("scraped_*.jpg"):
        tail = p.stem[len("scraped_"):]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums) + 1 if nums else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scraped-root", type=Path, required=True,
                    help="e.g. data/archive/scraped/")
    ap.add_argument("--embedder-root", type=Path, required=True,
                    help="e.g. data/labeled/embedder/")
    ap.add_argument("--min-fg", type=float, default=0.10,
                    help="Stricter than training-data threshold "
                         "since scraped product shots are usually clean.")
    ap.add_argument("--min-uniform-fg", type=float, default=0.20)
    ap.add_argument("--low-ratio", type=float, default=2.0)
    ap.add_argument("--high-ratio", type=float, default=5.0)
    ap.add_argument("--apply", action="store_true",
                    help="Without this, just reports counts.")
    args = ap.parse_args()

    session = new_session()
    pass_counts: Counter = Counter()
    fail_counts: Counter = Counter()
    copied = 0

    for cls_dir in sorted(p for p in args.scraped_root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        target = args.embedder_root / cls
        target.mkdir(parents=True, exist_ok=True)
        idx = _next_scraped_idx(target)
        cls_pass = 0
        cls_fail = 0
        for src in sorted(cls_dir.iterdir()):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            bgr = cv2.imread(str(src))
            if bgr is None:
                cls_fail += 1
                continue
            fg, r = fg_metrics(bgr, session)
            if passes(fg, r, args.min_fg, args.low_ratio,
                      args.high_ratio, args.min_uniform_fg):
                cls_pass += 1
                if args.apply:
                    dst = target / f"scraped_{idx:06d}.jpg"
                    cv2.imwrite(str(dst), bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    idx += 1
                    copied += 1
            else:
                cls_fail += 1
        pass_counts[cls] = cls_pass
        fail_counts[cls] = cls_fail
        total = cls_pass + cls_fail
        rate = (100.0 * cls_pass / total) if total else 0
        print(f"  {cls:<10} pass={cls_pass}  fail={cls_fail}  "
              f"rate={rate:.1f}%")

    total_pass = sum(pass_counts.values())
    total_fail = sum(fail_counts.values())
    print()
    print(f"OVERALL: pass={total_pass}  fail={total_fail}")
    if args.apply:
        print(f"copied {copied} scraped images into {args.embedder_root}")


if __name__ == "__main__":
    main()
