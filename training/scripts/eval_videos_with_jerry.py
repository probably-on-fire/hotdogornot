"""Run Jerry's YOLO+EffNet pipeline over every frame of every video.

For each .mp4/.mov in --videos-dir:
  1. Pull target_class from the sidecar (or infer from filename).
  2. Extract frames at --fps via ffmpeg.
  3. Run JerryPipeline.run() per frame — that gives detector bbox +
     classifier top-1 + confidence.
  4. Aggregate: per-frame top-1, majority vote, mean confidence per class.

Output a markdown report at --out plus a JSON alongside. Shows
per-video and overall accuracy at both per-frame and per-video
(majority-vote) granularity. Tells us whether Jerry's pretrained
pipeline generalizes to OUR recording style.

Usage:

  python -m scripts.eval_videos_with_jerry \\
      --videos-dir /opt/rfcai/repo/training/data/videos \\
      --jerry-dir /home/rfcai/training/models/jerry \\
      --fps 2 \\
      --out reports/jerry_on_videos_<ts>.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from rfconnectorai.pipeline.jerry_pipeline import JerryPipeline


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
CANONICAL_CLASSES = {
    "SMA-M", "SMA-F", "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F", "2.4mm-M", "2.4mm-F",
    "1.85mm-M", "1.85mm-F",
}


def _sidecar_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".crops.json")


def _infer_class(stem: str) -> str | None:
    for cls in CANONICAL_CLASSES:
        if stem.startswith(cls + "_"):
            return cls
    return None


def _truth(video: Path) -> str | None:
    sc = _sidecar_path(video)
    if sc.exists():
        try:
            d = json.loads(sc.read_text())
            cls = d.get("target_class")
            if cls in CANONICAL_CLASSES:
                return cls
        except Exception:
            pass
    return _infer_class(video.stem)


def _sharpness(bgr) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_video(
    video: Path, pipe: JerryPipeline, fps: float, ff: str,
    min_sharpness: float = 0.0,
) -> dict:
    truth = _truth(video)
    if truth is None:
        return {"video": video.name, "skipped": True, "reason": "no class"}

    per_frame: list[dict] = []
    n_blurry = 0
    with tempfile.TemporaryDirectory(prefix="jerry_eval_") as tmp:
        tmpp = Path(tmp)
        subprocess.run(
            [ff, "-y", "-i", str(video),
             "-vf", f"fps={fps}", "-q:v", "4",
             str(tmpp / "f_%04d.jpg")],
            capture_output=True, check=False,
        )
        frames = sorted(tmpp.glob("f_*.jpg"))
        for fp in frames:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                per_frame.append({"frame": fp.name, "pred": None, "conf": 0.0})
                continue
            if min_sharpness > 0.0:
                s = _sharpness(bgr)
                if s < min_sharpness:
                    n_blurry += 1
                    per_frame.append({
                        "frame": fp.name, "pred": None, "conf": 0.0,
                        "skipped_blurry": True, "sharpness": s,
                    })
                    continue
            preds = pipe.run(bgr)
            if not preds:
                per_frame.append({"frame": fp.name, "pred": None, "conf": 0.0,
                                  "no_detection": True})
                continue
            top = preds[0]
            per_frame.append({
                "frame": fp.name,
                "pred": top["class_name"],
                "conf": float(top["confidence"]),
                "box_score": float(top.get("box_score", 0.0)),
            })

    n_total = len(per_frame)
    n_detected = sum(1 for r in per_frame if r.get("pred"))
    n_correct_frames = sum(1 for r in per_frame if r.get("pred") == truth)
    counts = Counter(r["pred"] for r in per_frame if r.get("pred"))
    majority_pred, majority_count = counts.most_common(1)[0] if counts else (None, 0)
    correct_video = (majority_pred == truth)
    mean_conf = (
        sum(r["conf"] for r in per_frame if r.get("pred") == majority_pred)
        / max(1, majority_count)
    )
    n_kept = n_total - n_blurry
    return {
        "video": video.name,
        "truth": truth,
        "n_frames": n_total,
        "n_blurry_skipped": n_blurry,
        "n_kept": n_kept,
        "n_detected": n_detected,
        "n_correct_frames": n_correct_frames,
        "frame_accuracy": n_correct_frames / n_kept if n_kept else 0.0,
        "majority_pred": majority_pred,
        "majority_share": majority_count / n_kept if n_kept else 0.0,
        "video_correct": correct_video,
        "mean_conf_majority": mean_conf,
        "prediction_dist": dict(counts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--jerry-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument(
        "--min-sharpness", type=float, default=0.0,
        help="Skip frames whose Laplacian-variance sharpness is below "
             "this (0 = no filter; ~60 lenient, ~120 strict).",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"ffmpeg missing: {e}", file=sys.stderr)
        return 1

    pipe = JerryPipeline(args.jerry_dir)
    print(f"loaded JerryPipeline: classes={pipe.class_names} "
          f"cls_size={pipe.cls_size}")

    videos = sorted(
        p for p in args.videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    print(f"found {len(videos)} videos")

    rows: list[dict] = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}", flush=True)
        t0 = time.perf_counter()
        try:
            r = process_video(v, pipe, args.fps, ff,
                              min_sharpness=args.min_sharpness)
        except Exception as e:
            r = {"video": v.name, "error": str(e)}
        dt = time.perf_counter() - t0
        if r.get("skipped"):
            print(f"  SKIP ({r.get('reason')})", flush=True)
            continue
        if r.get("error"):
            print(f"  ERROR {r['error']}", flush=True)
            rows.append(r)
            continue
        rows.append(r)
        print(
            f"  truth={r['truth']} maj={r['majority_pred']} "
            f"({r['majority_share']*100:.0f}%, conf {r['mean_conf_majority']:.2f}) "
            f"frame_acc={r['frame_accuracy']*100:.1f}%  "
            f"detect={r['n_detected']}/{r['n_frames']}  ({dt:.1f}s)",
            flush=True,
        )

    # Aggregate
    n_videos = len(rows)
    video_correct = sum(1 for r in rows if r.get("video_correct"))
    total_frames = sum(r.get("n_frames", 0) for r in rows)
    correct_frames = sum(r.get("n_correct_frames", 0) for r in rows)
    detected_frames = sum(r.get("n_detected", 0) for r in rows)

    md = [
        f"# Jerry's pipeline vs our videos\n",
        f"_{n_videos} videos, {total_frames} frames @ {args.fps} fps_\n",
        f"- **Video-level accuracy** (majority vote per clip): "
        f"**{video_correct}/{n_videos} = {video_correct/n_videos*100:.1f}%**",
        f"- **Frame-level accuracy** (every frame counted): "
        f"**{correct_frames}/{total_frames} = {correct_frames/total_frames*100:.1f}%**",
        f"- **Detection recall**: {detected_frames}/{total_frames} "
        f"= {detected_frames/total_frames*100:.1f}% (frames where YOLO found a box)",
        "",
        "## Per-video results",
        "",
        "| video | truth | majority_pred | majority% | frame_acc | conf | det/frames |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            md.append(f"| {r['video']} | ERR | | | | | |")
            continue
        ok = "✓" if r["video_correct"] else "✗"
        md.append(
            f"| {r['video']} | {r['truth']} | **{r['majority_pred']}** {ok} | "
            f"{r['majority_share']*100:.0f}% | "
            f"{r['frame_accuracy']*100:.0f}% | "
            f"{r['mean_conf_majority']:.2f} | "
            f"{r['n_detected']}/{r['n_frames']} |"
        )

    # Per-class video majority accuracy
    md.append("\n## Per-class video accuracy")
    md.append("")
    md.append("| class | videos | correct |")
    md.append("|---|---|---|")
    by_cls: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r.get("truth"):
            by_cls[r["truth"]].append(r.get("video_correct", False))
    for cls in sorted(by_cls):
        oks = by_cls[cls]
        md.append(
            f"| {cls} | {len(oks)} | "
            f"{sum(oks)}/{len(oks)} = {sum(oks)/len(oks)*100:.0f}% |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"results": rows, "fps": args.fps}, indent=2),
    )
    print(f"\nwrote {args.out}")
    print(
        f"OVERALL: video {video_correct}/{n_videos} = "
        f"{video_correct/n_videos*100:.1f}%, "
        f"frame {correct_frames}/{total_frames} = "
        f"{correct_frames/total_frames*100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
