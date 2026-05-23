"""One-shot extraction over every video in data/videos/.

Reads each .mp4/.mov in the videos root, runs ffmpeg at the given
fps, runs the same Hough crop detector the live /upload-video
endpoint uses (pad_frac=0.35, accumulator_threshold=22), writes
agg_NNNN.jpg into data/labeled/embedder/<class>/, and patches the
sidecar manifest so /labeler/videos reflects the extraction state.

Skips any video whose sidecar already has a non-empty
extracted_crops list. Safe to re-run after partial failures.

Usage:

    sudo -u rfcai /opt/rfcai/training/.venv/bin/python -m \
        scripts.batch_extract_videos \
        --videos-dir /opt/rfcai/repo/training/data/videos \
        --data-dir /opt/rfcai/repo/training/data/labeled/embedder \
        --fps 2 --max-crops 5

Designed to be safe to nohup + tail the log; never touches the
training corpus for classes where the video has no sidecar
class info (i.e. legacy REC_*.mp4 files without a class label
get skipped with a warning rather than dumped into a guessed
folder).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

from rfconnectorai.data_fetch.connector_crops import (
    detect_connector_crops_hough,
)


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
CANONICAL_CLASSES = {
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
    "1.85mm-M", "1.85mm-F",
}


def _video_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".crops.json")


def _next_agg_index(out_dir: Path) -> int:
    existing: list[int] = []
    for p in out_dir.glob("agg_*.jpg"):
        tail = p.stem[len("agg_"):]
        if tail.isdigit():
            existing.append(int(tail))
    return max(existing) + 1 if existing else 0


def _infer_class_from_name(stem: str) -> str | None:
    """For legacy clips with iOS-style names — give up gracefully."""
    for cls in CANONICAL_CLASSES:
        if stem.startswith(cls + "_"):
            return cls
    return None


def _frame_sharpness(bgr) -> float:
    """Variance of the Laplacian — high = sharp, low = blurry. Standard
    cv2 trick. Computed on the raw frame (not the crop) because Hough
    on a blurry frame is unreliable too — we want to bail out before
    spending time on a bad crop."""
    import cv2 as _cv2
    gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
    return float(_cv2.Laplacian(gray, _cv2.CV_64F).var())


def process_one(
    video: Path, data_root: Path, fps: float, max_crops: int, ff: str,
    min_sharpness: float = 0.0,
) -> dict:
    """Returns {video, target_class, n_frames, n_crops, n_skipped_blurry, status, ...}."""
    sidecar = _video_sidecar_path(video)
    manifest: dict = {}
    if sidecar.exists():
        try:
            manifest = json.loads(sidecar.read_text())
        except Exception:
            pass

    # Skip if already extracted.
    existing_crops = manifest.get("extracted_crops") or []
    if existing_crops:
        return {
            "video": video.name,
            "target_class": manifest.get("target_class"),
            "status": "skipped-already-extracted",
            "n_crops": len(existing_crops),
        }

    # Need a target class. Use sidecar's target_class if present,
    # else infer from filename. Files with neither get skipped.
    target_cls = manifest.get("target_class") or _infer_class_from_name(video.stem)
    if not target_cls or target_cls not in CANONICAL_CLASSES:
        return {
            "video": video.name,
            "target_class": None,
            "status": "skipped-no-class",
        }

    out_dir = data_root / target_cls
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_agg_index(out_dir)
    saved_crops: list[str] = []
    n_frames = 0
    n_skipped_blurry = 0
    sharpness_samples: list[float] = []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="batch_extract_") as tmp:
        tmpp = Path(tmp)
        subprocess.run(
            [ff, "-y", "-i", str(video),
             "-vf", f"fps={fps}", "-q:v", "4",
             str(tmpp / "f_%04d.jpg")],
            capture_output=True, check=False,
        )
        frame_paths = sorted(tmpp.glob("f_*.jpg"))
        n_frames = len(frame_paths)
        for fp in frame_paths:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            # Reject blurry frames before paying for Hough + write.
            # Sharpness measured on the full frame; if the wide shot is
            # already blurry the cropped connector face will be worse.
            if min_sharpness > 0.0:
                sharp = _frame_sharpness(bgr)
                sharpness_samples.append(sharp)
                if sharp < min_sharpness:
                    n_skipped_blurry += 1
                    continue
            results = detect_connector_crops_hough(
                bgr, max_crops=int(max_crops),
                pad_frac=0.35, accumulator_threshold=22,
            )
            for r in results:
                out_path = out_dir / f"agg_{idx:04d}.jpg"
                cv2.imwrite(str(out_path), r.crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                idx += 1
                saved_crops.append(str(out_path))
    dt = time.perf_counter() - t0

    # Patch sidecar.
    if not manifest:
        # Build a minimal manifest for legacy clips that had no sidecar.
        manifest = {
            "video_filename": video.name,
            "target_class": target_cls,
            "uploaded_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(video.stat().st_mtime),
            ),
            "uploaded_by": None,
        }
    manifest["n_frames_extracted"] = n_frames
    manifest["n_frames_skipped_blurry"] = n_skipped_blurry
    if sharpness_samples:
        manifest["sharpness_min"] = round(min(sharpness_samples), 1)
        manifest["sharpness_max"] = round(max(sharpness_samples), 1)
        manifest["sharpness_mean"] = round(
            sum(sharpness_samples) / len(sharpness_samples), 1,
        )
    manifest["min_sharpness_threshold"] = min_sharpness
    manifest["extracted_crops"] = saved_crops
    manifest["processed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
    )
    manifest["extraction_state"] = "ok"
    manifest["batch_extracted"] = True
    sidecar.write_text(json.dumps(manifest, indent=2))

    return {
        "video": video.name,
        "target_class": target_cls,
        "status": "ok",
        "n_frames": n_frames,
        "n_skipped_blurry": n_skipped_blurry,
        "n_crops": len(saved_crops),
        "dt_sec": round(dt, 1),
        "sharpness_mean": (
            round(sum(sharpness_samples) / len(sharpness_samples), 1)
            if sharpness_samples else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path,
                    default=Path("/opt/rfcai/repo/training/data/videos"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/opt/rfcai/repo/training/data/labeled/embedder"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-crops", type=int, default=5)
    ap.add_argument(
        "--min-sharpness", type=float, default=0.0,
        help="Skip frames whose Laplacian-variance sharpness is below "
             "this. 0 = accept all. Typical thresholds: 60 (lenient), "
             "120 (strict), 200 (very strict). Most modern phone "
             "videos at 1080p sit in the 100-400 range when the shot "
             "is in focus.",
    )
    args = ap.parse_args()

    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"FATAL: ffmpeg unavailable: {e}", file=sys.stderr)
        return 1

    videos = sorted(
        p for p in args.videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    print(f"found {len(videos)} videos in {args.videos_dir}", flush=True)
    if not videos:
        return 0

    summary: list[dict] = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name} ...", flush=True)
        try:
            res = process_one(
                v, args.data_dir, args.fps, args.max_crops, ff,
                min_sharpness=args.min_sharpness,
            )
        except Exception as e:
            res = {"video": v.name, "status": "error", "error": str(e)}
        summary.append(res)
        cls = res.get("target_class") or "-"
        if res.get("status") == "ok":
            n_blurry = res.get("n_skipped_blurry", 0)
            blur_str = f" (skipped {n_blurry} blurry)" if n_blurry else ""
            print(
                f"  -> {cls:<10} frames={res['n_frames']} "
                f"crops={res['n_crops']}{blur_str} ({res['dt_sec']}s)",
                flush=True,
            )
        else:
            print(f"  -> status={res['status']} {res.get('error', '')}",
                  flush=True)

    # Print a per-class summary at the end.
    print("\n=== summary ===", flush=True)
    by_cls: dict[str, dict] = {}
    for r in summary:
        cls = r.get("target_class") or "-"
        d = by_cls.setdefault(cls, {"videos": 0, "frames": 0, "crops": 0})
        d["videos"] += 1
        d["frames"] += r.get("n_frames", 0)
        d["crops"] += r.get("n_crops", 0)
    for cls in sorted(by_cls):
        d = by_cls[cls]
        print(f"  {cls:<12} videos={d['videos']:>3} "
              f"frames={d['frames']:>5} crops={d['crops']:>6}", flush=True)
    print(f"\nTotal: {len(summary)} videos processed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
