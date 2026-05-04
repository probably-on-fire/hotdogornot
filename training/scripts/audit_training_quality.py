"""
Audit + clean training data using the same rembg fg filter the predict
service uses to reject background-only crops. Without this, Hough
hits on non-connector circles (knobs, lights, the wood grain pattern,
etc.) end up as labeled training data — silently poisoning the model.

The filter rule is the same one in predict_service._crop_passes_fg_filter:
keep crops that either fill the tight crop edge-to-edge (low center
ratio with high fg) OR sit as a small centered object (very high center
ratio). Reject the in-between zone where rembg latches onto wood-grain
patterns that aren't real objects.

Usage (run on the box as the rfcai user):
    python scripts/_audit_training_quality.py \\
        --data-dir data/labeled/embedder \\
        --quarantine-dir data/labeled/_quarantine_lowq \\
        [--sample N]    # only audit a random sample (faster on huge folders)
        [--apply]       # actually move; without this, just reports counts
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from rembg import new_session, remove
except ImportError:
    print("rembg not installed; run from the rfcai venv", file=sys.stderr)
    sys.exit(2)


MIN_FG_FRACTION = 0.05
MIN_UNIFORM_FG = 0.20
LOW_CENTER_RATIO = 2.0
HIGH_CENTER_RATIO = 5.0


def _passes_with_thresholds(f, r, min_fg, min_uniform_fg, low_r, high_r):
    if f < min_fg:
        return False
    return (f >= min_uniform_fg and r <= low_r) or r >= high_r


def crop_metrics(bgr: np.ndarray, session) -> tuple[float, float]:
    """Return (fg_fraction, center_density_ratio) for a BGR crop."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=session)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return 1.0, 1.0   # fail open
    alpha = rgba[:, :, 3]
    h, w = alpha.shape
    if h == 0 or w == 0:
        return 0.0, 0.0
    fg = alpha > 32
    fg_total = int(fg.sum())
    fg_frac = fg_total / float(alpha.size)
    cy0, cy1 = h // 4, h - h // 4
    cx0, cx1 = w // 4, w - w // 4
    inner = fg[cy0:cy1, cx0:cx1]
    inner_total = int(inner.sum())
    inner_area = max(1, inner.size)
    outer_area = max(1, fg.size - inner_area)
    inner_density = inner_total / inner_area
    outer_density = (fg_total - inner_total) / outer_area
    if outer_density > 1e-6:
        ratio = inner_density / outer_density
    else:
        ratio = 10.0 if inner_density > 0 else 0.0
    return fg_frac, ratio


def crop_passes(fg_frac: float, ratio: float,
                min_fg=MIN_FG_FRACTION,
                min_uniform_fg=MIN_UNIFORM_FG,
                low_r=LOW_CENTER_RATIO,
                high_r=HIGH_CENTER_RATIO) -> bool:
    return _passes_with_thresholds(
        fg_frac, ratio, min_fg, min_uniform_fg, low_r, high_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Labeled-data root, e.g. data/labeled/embedder/")
    ap.add_argument("--quarantine-dir", type=Path,
                    help="Where to move failed crops (with --apply). "
                         "Failed paths are mirrored under here.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Audit only N random files per class (0 = all). "
                         "Faster for spot-checks.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move failing crops to quarantine. "
                         "Without this, just reports counts.")
    ap.add_argument("--min-fg", type=float, default=MIN_FG_FRACTION,
                    help="Override the foreground-fraction floor. Lower "
                         "= more permissive. 0.05 was the inference-time "
                         "threshold and rejects ~94%% of training data; "
                         "0.02 is a reasonable training-time threshold "
                         "that still kills mostly-desk crops.")
    ap.add_argument("--min-uniform-fg", type=float, default=MIN_UNIFORM_FG,
                    help="Min fg for the 'uniform fill' keep path.")
    ap.add_argument("--low-ratio", type=float, default=LOW_CENTER_RATIO,
                    help="Max center ratio for 'uniform fill' keep.")
    ap.add_argument("--high-ratio", type=float, default=HIGH_CENTER_RATIO,
                    help="Min center ratio for 'small centered object' keep.")
    ap.add_argument("--include-mask-variants", action="store_true",
                    help="Also audit *_mask.jpg / *_clean.jpg variants. "
                         "By default these are skipped because they're "
                         "rembg-derived and the filter doesn't apply.")
    ap.add_argument("--bases-only", action="store_true",
                    help="Audit only the base agg_NNNN.jpg / video_NNNN.jpg "
                         "/ photo_NNNN.jpg files; skip all rembg variants "
                         "(_bg0-4, _central, _z70, etc.). Lets you measure "
                         "the underlying capture quality without the "
                         "synthetic-background composites confusing rembg.")
    args = ap.parse_args()

    if args.apply and args.quarantine_dir is None:
        sys.exit("--apply requires --quarantine-dir")

    session = new_session()
    rng = random.Random(0)

    VARIANT_SUFFIXES = (
        "_clean", "_mask", "_central", "_centralv2",
        "_z70", "_z50",
    ) + tuple(f"_bg{i}" for i in range(10))

    def base_stem_of(p: Path) -> str:
        """Map agg_0001_bg0 -> agg_0001 so we can group a base with its
        rembg-derived variants."""
        for s in VARIANT_SUFFIXES:
            if p.stem.endswith(s):
                return p.stem[: -len(s)]
        return p.stem

    overall_pass = 0
    overall_fail = 0
    by_class_pass: Counter = Counter()
    by_class_fail: Counter = Counter()
    moved = 0

    for cls_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        if cls_dir.name.startswith("_"):
            continue   # quarantine-style folders
        files = sorted(p for p in cls_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        bases = [
            p for p in files
            if not any(p.stem.endswith(s) for s in VARIANT_SUFFIXES)
        ]
        if args.sample and len(bases) > args.sample:
            bases = rng.sample(bases, args.sample)
        cls_pass = 0
        cls_fail = 0
        # Group all files by their base stem so we can quarantine a
        # failing base together with its rembg variants — variants of
        # a bad base (mostly desk with no connector) are equally junk.
        files_by_base: dict[str, list[Path]] = {}
        for f in files:
            files_by_base.setdefault(base_stem_of(f), []).append(f)
        for base in bases:
            bgr = cv2.imread(str(base))
            if bgr is None:
                continue
            fg_frac, ratio = crop_metrics(bgr, session)
            base_key = base.stem  # bases have no variant suffix
            sibling_files = files_by_base.get(base_key, [base])
            if crop_passes(fg_frac, ratio,
                           min_fg=args.min_fg,
                           min_uniform_fg=args.min_uniform_fg,
                           low_r=args.low_ratio,
                           high_r=args.high_ratio):
                cls_pass += 1
            else:
                cls_fail += 1
                if args.apply:
                    for fp in sibling_files:
                        rel = fp.relative_to(args.data_dir)
                        dst = args.quarantine_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if fp.exists():
                            shutil.move(str(fp), str(dst))
                            moved += 1
        overall_pass += cls_pass
        overall_fail += cls_fail
        by_class_pass[cls_dir.name] = cls_pass
        by_class_fail[cls_dir.name] = cls_fail
        total = cls_pass + cls_fail
        rate = (100.0 * cls_pass / total) if total else 0
        print(f"  {cls_dir.name:<10} bases pass={cls_pass}  fail={cls_fail}  "
              f"rate={rate:.1f}%")

    print()
    total = overall_pass + overall_fail
    rate = (100.0 * overall_pass / total) if total else 0
    print(f"OVERALL: pass={overall_pass}  fail={overall_fail}  "
          f"rate={rate:.1f}%")
    if args.apply:
        print(f"moved {moved} failing crops to {args.quarantine_dir}")


if __name__ == "__main__":
    main()
