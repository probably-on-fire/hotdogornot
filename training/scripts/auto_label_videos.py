"""
Auto-label connector videos using a center-brightness heuristic.

For each input video, we extract frames at a target fps, run the
connector blob detector on each frame, and decide M vs F per detected
crop based on whether the central region is bright (copper pin → M)
or dark (recessed socket → F). Crops land in
data/labeled/embedder/<size>-<M|F>/video_NNNN.jpg.

This is a stopgap until the user labels via the Streamlit Video Labeler,
but the heuristic gets us off zero quickly and the eventual labeled
review pass can correct any mis-classifications.

Usage (programmatic):
    from scripts.auto_label_videos import process_video
    process_video(Path("clip.mov"), size_class="2.4mm",
                  forced_gender=None, fps=4.0)

`forced_gender` short-circuits the heuristic when you already know the
video is all-M or all-F (e.g. IMG_0279.MOV which the user confirmed).

Usage (CLI):
    python -m scripts.auto_label_videos \\
        --video path/to/2_4mm.MOV --size 2.4mm
    python -m scripts.auto_label_videos \\
        --video path/to/IMG_0279.MOV --size 2.4mm --gender M
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from rfconnectorai.data_fetch.connector_crops import detect_connector_crops


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "labeled" / "embedder"


def _next_idx(class_dir: Path) -> int:
    if not class_dir.is_dir(): return 0
    indices = [
        int(p.stem[len("video_"):])
        for p in class_dir.glob("video_*.jpg")
        if p.stem[len("video_"):].isdigit()
    ]
    return max(indices) + 1 if indices else 0


def _classify_gender(crop_bgr: np.ndarray, threshold: float = 80.0) -> str:
    """Returns 'M' if the crop's central region is bright, 'F' if dark.

    Logic: the crop is centered on the connector's mating face. The very
    center contains either a bright copper pin (M) or a dark recessed
    hole/ring around the socket (F). We sample a tiny patch (~15% of
    side) at the center and compare its mean luminance against a
    threshold tuned for typical phone shots on a wood bench.
    """
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return "M"   # too small to judge; default to M
    cx, cy = w // 2, h // 2
    r = max(2, int(min(w, h) * 0.075))
    region = crop_bgr[cy - r:cy + r, cx - r:cx + r]
    if region.size == 0:
        return "M"
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return "M" if gray.mean() > threshold else "F"


def process_video(
    video_path: Path,
    size_class: str,
    forced_gender: str | None = None,
    fps: float = 4.0,
    max_crops_per_frame: int = 3,
) -> dict[str, int]:
    """Extract → detect → classify → save crops. Returns per-class counts."""
    if size_class not in {"SMA", "3.5mm", "2.92mm", "2.4mm"}:
        raise ValueError(f"unknown size_class {size_class!r}")
    if forced_gender is not None and forced_gender not in {"M", "F"}:
        raise ValueError(f"forced_gender must be 'M' or 'F' or None")

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    counts = {"M": 0, "F": 0, "skipped": 0}
    with tempfile.TemporaryDirectory(prefix="rfcai_autolbl_") as tmp:
        tmp_dir = Path(tmp)
        out_pat = str(tmp_dir / "frame_%04d.jpg")
        # Extract frames at the target fps.
        subprocess.run(
            [ff, "-y", "-i", str(video_path),
             "-vf", f"fps={fps}", "-q:v", "4", out_pat],
            capture_output=True, check=False,
        )
        frames = sorted(tmp_dir.glob("frame_*.jpg"))
        if not frames:
            print(f"  WARN: no frames extracted from {video_path}")
            return counts

        for frame_path in frames:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                counts["skipped"] += 1
                continue
            crops = detect_connector_crops(bgr, max_crops=max_crops_per_frame)
            for crop_result in crops:
                gender = forced_gender if forced_gender else _classify_gender(crop_result.crop)
                cls = f"{size_class}-{gender}"
                cls_dir = DATA_ROOT / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                idx = _next_idx(cls_dir)
                cv2.imwrite(
                    str(cls_dir / f"video_{idx:04d}.jpg"),
                    crop_result.crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                counts[gender] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--size", type=str, required=True,
                    choices=["SMA", "3.5mm", "2.92mm", "2.4mm"])
    ap.add_argument("--gender", choices=["M", "F"], default=None,
                    help="Force all crops to a single gender (skips heuristic).")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-crops", type=int, default=3)
    args = ap.parse_args()

    counts = process_video(args.video, args.size, args.gender, args.fps, args.max_crops)
    print(f"  M: {counts['M']}, F: {counts['F']}, skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
