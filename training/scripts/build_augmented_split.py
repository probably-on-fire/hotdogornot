"""Build a leak-free train/val split from the augmented dataset.

The existing data/labeled/embedder/<class>/agg_NNNN.jpg corpus has
crops from many source videos. If we shuffle randomly into
train/val, two crops from the same video will land on opposite
sides — a strong leak (same camera angle, same lighting, same
connector instance).

This script walks every video sidecar and treats the (video, class)
pair as the unit of splitting. The crops listed in each sidecar's
extracted_crops field all go to the same side. Holds back ~15% of
videos per class for val (rounded up to at least 1 video).

Photo (non-video) crops (photo_*, IMG_*, scraped_*) get a separate
random split — they're per-image, no leakage risk.

Outputs:
  data/snapshots/augmented_<date>/
    train/<class>/<file>.jpg ...
    val/<class>/<file>.jpg ...
    manifest.json   — per-file source attribution
  data/snapshots/augmented_<date>.tar.gz   (optional)

Usage:

  python -m scripts.build_augmented_split \
      --data-dir /opt/rfcai/repo/training/data/labeled/embedder \
      --videos-dir /opt/rfcai/repo/training/data/videos \
      --out /opt/rfcai/repo/training/data/snapshots/augmented_<date> \
      --val-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
    "1.85mm-M", "1.85mm-F",
]
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
NON_TRAINING_PREFIXES = ("synth_",)
DERIVED_SUFFIXES = ("_clean", "_mask", "_central", "_centralv2")


def _is_derived(stem: str) -> bool:
    for s in DERIVED_SUFFIXES:
        if stem.endswith(s):
            return True
    if "_bg" in stem and stem.rsplit("_bg", 1)[1].isdigit():
        return True
    if "_z" in stem and stem.rsplit("_z", 1)[1].isdigit():
        return True
    return False


def collect_video_crop_map(videos_dir: Path) -> dict[Path, str]:
    """Returns {abs_crop_path: source_video_filename} for every crop
    listed in any video sidecar. Crops not in any sidecar are
    treated as photo-mode (no leakage group).
    """
    mapping: dict[Path, str] = {}
    for sidecar in videos_dir.glob("*.crops.json"):
        try:
            d = json.loads(sidecar.read_text())
        except Exception:
            continue
        video_name = d.get("video_filename") or sidecar.stem
        for c in d.get("extracted_crops") or []:
            mapping[Path(c).resolve()] = video_name
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="data/labeled/embedder/")
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="data/videos/ (for sidecar lookup)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output snapshot dir")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="held-out fraction (target)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tar", action="store_true",
                    help="also emit out.tar.gz")
    ap.add_argument("--filter-manifest", type=Path, default=None,
                    help="filter_video_crops_*.json — only include crops "
                         "whose path is in its kept_paths (video-crops only; "
                         "photo/synthetic stay unfiltered)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    video_crops = collect_video_crop_map(args.videos_dir)
    print(f"sidecar-tracked crops: {len(video_crops)}")

    keep_video_crops: set[Path] | None = None
    if args.filter_manifest:
        fm = json.loads(args.filter_manifest.read_text())
        keep_video_crops = set()
        for paths in fm.get("kept_paths", {}).values():
            for p in paths:
                keep_video_crops.add(Path(p).resolve())
        print(f"filter manifest: {len(keep_video_crops)} kept video crops")

    args.out.mkdir(parents=True, exist_ok=True)
    train_root = args.out / "train"
    val_root = args.out / "val"
    train_root.mkdir(exist_ok=True)
    val_root.mkdir(exist_ok=True)

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": str(args.data_dir),
        "videos_dir": str(args.videos_dir),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "classes": [],
    }

    for cls in CANONICAL_CLASSES:
        cls_dir = args.data_dir / cls
        if not cls_dir.is_dir():
            continue
        # Group crops by source.
        videos: dict[str, list[Path]] = defaultdict(list)   # video_name -> crops
        photos: list[Path] = []
        for p in cls_dir.iterdir():
            if not p.is_file(): continue
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
            if p.stem.startswith(NON_TRAINING_PREFIXES): continue
            if _is_derived(p.stem): continue
            src_vid = video_crops.get(p.resolve())
            if src_vid:
                if keep_video_crops is not None and p.resolve() not in keep_video_crops:
                    continue
                videos[src_vid].append(p)
            else:
                photos.append(p)

        # Choose val videos (~val_frac of videos, at least 1 if any).
        vid_names = sorted(videos)
        rng.shuffle(vid_names)
        n_val_videos = max(1, round(len(vid_names) * args.val_frac)) if vid_names else 0
        val_videos = set(vid_names[:n_val_videos])

        # Photos: shuffle + slice.
        rng.shuffle(photos)
        n_val_photos = round(len(photos) * args.val_frac)
        val_photos = set(photos[:n_val_photos])

        (train_root / cls).mkdir(exist_ok=True)
        (val_root / cls).mkdir(exist_ok=True)

        train_count = 0; val_count = 0
        for v, crops in videos.items():
            dst_root = val_root if v in val_videos else train_root
            for c in crops:
                shutil.copy2(c, dst_root / cls / c.name)
                if v in val_videos: val_count += 1
                else: train_count += 1
        for p in photos:
            dst_root = val_root if p in val_photos else train_root
            shutil.copy2(p, dst_root / cls / p.name)
            if p in val_photos: val_count += 1
            else: train_count += 1

        cls_record = {
            "class": cls,
            "train": train_count,
            "val": val_count,
            "n_videos_total": len(vid_names),
            "n_videos_val": n_val_videos,
            "val_videos": sorted(val_videos),
            "n_photos_total": len(photos),
            "n_photos_val": n_val_photos,
        }
        manifest["classes"].append(cls_record)
        print(
            f"  {cls:<10} train={train_count:>5} val={val_count:>4} "
            f"(videos: {len(vid_names)} total, {n_val_videos} held; "
            f"photos: {len(photos)} total, {n_val_photos} held)"
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}/manifest.json")

    if args.tar:
        import tarfile
        tar_path = args.out.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=args.out.name)
        print(f"wrote {tar_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
