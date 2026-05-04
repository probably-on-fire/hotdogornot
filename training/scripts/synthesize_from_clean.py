"""
Generate high-quality synthetic training data from existing video crops.

Pipeline per base image:
  1. rembg → silhouette + alpha
  2. find tight bbox of the silhouette (alpha > threshold)
  3. crop the silhouette + small pad (5-25% jittered)
  4. composite onto a random background:
     * white  (30%) — matches classify-on-cleaned inference domain
     * gray   (15%) — neutral
     * beige  (10%) — paper/fabric
     * wood-tone-noise (25%) — desk-like
     * sampled-patch-from-real-bg (20%) — taken from a non-connector
       region of a real `_bg*` variant
  5. random rotation ±25°, optional horizontal flip
  6. mild color jitter to simulate lighting variation
  7. resize to 224x224, save as synth_NNNNNN.jpg

Per-base output: --variants-per-base (default 4) synthetic images.
The connector silhouette occupies 50-85% of the output frame —
matches the scale of real held-out phone shots after Hough cropping.

Usage (run on the box as rfcai):
    python scripts/synthesize_from_clean.py \\
        --data-dir data/labeled/embedder \\
        --variants-per-base 4
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove


VARIANT_SUFFIXES = (
    "_clean", "_mask", "_central", "_centralv2", "_z70", "_z50",
) + tuple(f"_bg{i}" for i in range(10))


def _is_variant_or_synth(p: Path) -> bool:
    if p.stem.startswith(("synth_", "tight_", "scraped_")):
        return True
    return any(p.stem.endswith(s) for s in VARIANT_SUFFIXES)


def _next_synth_idx(class_dir: Path) -> int:
    nums = []
    for p in class_dir.glob("synth_*.jpg"):
        tail = p.stem[len("synth_"):]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums) + 1 if nums else 0


def _silhouette_bbox(alpha: np.ndarray, threshold: int = 40) -> tuple[int, int, int, int] | None:
    mask = alpha > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def _bg_white(size: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(235, 255)
    return np.full((size, size, 3), base, dtype=np.uint8)


def _bg_gray(size: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(80, 180)
    return np.full((size, size, 3), base, dtype=np.uint8)


def _bg_beige(size: int, rng: random.Random) -> np.ndarray:
    b = rng.randint(170, 200)
    g = min(255, b + rng.randint(5, 25))
    r = min(255, b + rng.randint(15, 40))
    img = np.full((size, size, 3), 0, dtype=np.uint8)
    img[:, :] = (b, g, r)   # BGR
    return img


def _bg_wood_noise(size: int, rng: random.Random) -> np.ndarray:
    # Procedural wood-tone noise: striated horizontal pattern with
    # mid-warm color. Tan/oak palette — keep blue lower than green
    # lower than red so we get warm browns, not pinks/purples.
    base_b = rng.randint(110, 155)
    base_g = base_b + rng.randint(15, 35)
    base_r = base_g + rng.randint(15, 35)
    base_b, base_g, base_r = min(base_b, 200), min(base_g, 215), min(base_r, 230)
    img = np.full((size, size, 3), [base_b, base_g, base_r], dtype=np.uint8)
    # Add horizontal noise streaks
    streak = (np.random.randn(size, 1) * rng.uniform(8, 18)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + streak[:, None], 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Sparse darker dots/grain marks
    n_marks = rng.randint(15, 50)
    for _ in range(n_marks):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        rad = rng.randint(1, 3)
        col = (max(0, base_b - rng.randint(30, 70)),
               max(0, base_g - rng.randint(30, 70)),
               max(0, base_r - rng.randint(30, 70)))
        cv2.circle(img, (x, y), rad, col, -1)
    return img


def _bg_real_patch(size: int, real_bg_files: list[Path], rng: random.Random) -> np.ndarray | None:
    """Sample a square patch from one of the existing _bg* variants —
    take a patch far from the center (where the connector lives) so we
    get mostly-background pixels."""
    if not real_bg_files:
        return None
    src = rng.choice(real_bg_files)
    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    # Pick a corner: TL, TR, BL, BR
    corner = rng.choice(["TL", "TR", "BL", "BR"])
    patch_size = min(h, w) // 3
    if corner == "TL":
        y0, x0 = 0, 0
    elif corner == "TR":
        y0, x0 = 0, w - patch_size
    elif corner == "BL":
        y0, x0 = h - patch_size, 0
    else:
        y0, x0 = h - patch_size, w - patch_size
    patch = img[y0:y0 + patch_size, x0:x0 + patch_size]
    if patch.shape[0] != patch.shape[1] or patch.shape[0] < 32:
        return None
    return cv2.resize(patch, (size, size))


def _composite_silhouette(silhouette_rgba: np.ndarray, bg: np.ndarray,
                          target_scale: float, rng: random.Random) -> np.ndarray:
    """Place the silhouette centered on the background, scaled so it
    occupies `target_scale` of the bg's shortest side. The silhouette's
    own bbox is tight, so the composite has the connector dominate."""
    bg_size = bg.shape[0]
    sh, sw = silhouette_rgba.shape[:2]
    target_side = int(bg_size * target_scale)
    # Keep aspect ratio; resize so the longer side hits target.
    if sw >= sh:
        new_w = target_side
        new_h = max(1, int(sh * target_side / sw))
    else:
        new_h = target_side
        new_w = max(1, int(sw * target_side / sh))
    sil = cv2.resize(silhouette_rgba, (new_w, new_h),
                     interpolation=cv2.INTER_AREA)
    cy = bg_size // 2 + rng.randint(-bg_size // 12, bg_size // 12)
    cx = bg_size // 2 + rng.randint(-bg_size // 12, bg_size // 12)
    y0 = max(0, cy - new_h // 2)
    x0 = max(0, cx - new_w // 2)
    y1 = min(bg_size, y0 + new_h)
    x1 = min(bg_size, x0 + new_w)
    sy0 = 0; sx0 = 0
    sy1 = sy0 + (y1 - y0); sx1 = sx0 + (x1 - x0)
    if sy1 <= sy0 or sx1 <= sx0:
        return bg
    rgb = sil[sy0:sy1, sx0:sx1, :3].astype(np.float32)
    alpha = sil[sy0:sy1, sx0:sx1, 3:4].astype(np.float32) / 255.0
    out = bg.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (rgb * alpha + region * (1.0 - alpha)).astype(np.uint8)
    return out


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    # Random rotation ±25°
    angle = rng.uniform(-25.0, 25.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Maybe horizontal flip (preserves M/F since we crop centered)
    if rng.random() < 0.5:
        rotated = cv2.flip(rotated, 1)
    # Mild color jitter — brightness ±15%, contrast ±15%
    bri = rng.uniform(-0.15, 0.15)
    con = rng.uniform(-0.15, 0.15)
    f = rotated.astype(np.float32) / 255.0
    f = (f - 0.5) * (1.0 + con) + 0.5 + bri
    rotated = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    return rotated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--variants-per-base", type=int, default=4)
    ap.add_argument("--target-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bases-per-class", type=int, default=0,
                    help="Cap base images per class for faster runs (0 = all).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    session = new_session()

    # Collect a library of real `_bg*` files for sampling background patches.
    real_bg_files: list[Path] = []
    for bg_class in args.data_dir.iterdir():
        if not bg_class.is_dir() or bg_class.name.startswith("_"):
            continue
        real_bg_files.extend(bg_class.glob("*_bg*.jpg"))
    print(f"real-bg corpus: {len(real_bg_files)} files")

    saved_total = 0

    for cls_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        if cls_dir.name.startswith("_"):
            continue
        bases = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not _is_variant_or_synth(p)
        ]
        if args.max_bases_per_class:
            rng.shuffle(bases)
            bases = bases[: args.max_bases_per_class]
        print(f"  {cls_dir.name:<10} bases={len(bases)}")
        idx = _next_synth_idx(cls_dir)

        for base_path in bases:
            bgr = cv2.imread(str(base_path))
            if bgr is None:
                continue
            try:
                rgba = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                              session=session)
            except Exception:
                continue
            if rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            alpha = rgba[:, :, 3]
            box = _silhouette_bbox(alpha)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            ph, pw = y1 - y0 + 1, x1 - x0 + 1
            if ph < 24 or pw < 24:
                continue
            # Pad by jittered amount.
            for _ in range(args.variants_per_base):
                pad_frac = rng.uniform(0.05, 0.25)
                pad_y = int(ph * pad_frac)
                pad_x = int(pw * pad_frac)
                ay0 = max(0, y0 - pad_y)
                ay1 = min(rgba.shape[0], y1 + pad_y + 1)
                ax0 = max(0, x0 - pad_x)
                ax1 = min(rgba.shape[1], x1 + pad_x + 1)
                tight_rgba = rgba[ay0:ay1, ax0:ax1]
                # Convert tight_rgba's RGB (rembg ordering) to BGR for
                # composition with the BGR background.
                tight_bgr = cv2.cvtColor(tight_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                tight_silhouette = np.dstack(
                    [tight_bgr, tight_rgba[:, :, 3:4]])

                # Pick background.
                pick = rng.random()
                if pick < 0.30:
                    bg = _bg_white(args.target_size, rng)
                elif pick < 0.45:
                    bg = _bg_gray(args.target_size, rng)
                elif pick < 0.55:
                    bg = _bg_beige(args.target_size, rng)
                elif pick < 0.80:
                    bg = _bg_wood_noise(args.target_size, rng)
                else:
                    bg = _bg_real_patch(args.target_size, real_bg_files, rng)
                    if bg is None:
                        bg = _bg_wood_noise(args.target_size, rng)

                target_scale = rng.uniform(0.55, 0.90)
                comp = _composite_silhouette(
                    tight_silhouette, bg, target_scale, rng)
                comp = _augment(comp, rng)
                out = cls_dir / f"synth_{idx:06d}.jpg"
                cv2.imwrite(str(out), comp,
                            [cv2.IMWRITE_JPEG_QUALITY,
                             rng.randint(70, 95)])
                idx += 1
                saved_total += 1

    print()
    print(f"saved {saved_total} synthetic images")


if __name__ == "__main__":
    main()
