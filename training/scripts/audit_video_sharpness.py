"""Measure per-frame sharpness on every video, both on the full
frame and on the connector crop YOLO finds. Tells us where to
draw a real blur threshold.

Output: training/reports/sharpness_audit_<ts>.md with per-video
percentiles and a recommended threshold."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

from rfconnectorai.pipeline.jerry_pipeline import JerryPipeline


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _sharpness(bgr) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path,
                    default=Path("/opt/rfcai/repo/training/data/videos"))
    ap.add_argument("--jerry-dir", type=Path,
                    default=Path("/home/rfcai/training/models/jerry"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"ffmpeg missing: {e}", file=sys.stderr)
        return 1

    pipe = JerryPipeline(args.jerry_dir)
    videos = sorted(
        p for p in args.videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    print(f"{len(videos)} videos")

    rows: list[dict] = []
    all_frame_sharp: list[float] = []
    all_crop_sharp: list[float] = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.name}", flush=True)
        frame_vals: list[float] = []
        crop_vals: list[float] = []
        n_detected = 0
        with tempfile.TemporaryDirectory(prefix="sharp_") as tmp:
            tmpp = Path(tmp)
            subprocess.run(
                [ff, "-y", "-i", str(v),
                 "-vf", f"fps={args.fps}", "-q:v", "4",
                 str(tmpp / "f_%04d.jpg")],
                capture_output=True, check=False,
            )
            for fp in sorted(tmpp.glob("f_*.jpg")):
                bgr = cv2.imread(str(fp))
                if bgr is None: continue
                # Full-frame sharpness
                fs = _sharpness(bgr)
                frame_vals.append(fs)
                # YOLO crop, then sharpness on crop
                boxes = pipe._detect(bgr)
                if boxes:
                    n_detected += 1
                    x1, y1, x2, y2, _ = boxes[0]
                    crop = bgr[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        crop_vals.append(_sharpness(crop))
        if not frame_vals: continue
        rec = {
            "video": v.name,
            "n_frames": len(frame_vals),
            "n_detected": n_detected,
            "frame_sharp_p25": round(statistics.quantiles(frame_vals, n=4)[0], 1),
            "frame_sharp_p50": round(statistics.median(frame_vals), 1),
            "frame_sharp_p75": round(statistics.quantiles(frame_vals, n=4)[2], 1),
            "frame_sharp_max": round(max(frame_vals), 1),
            "crop_sharp_p25": (
                round(statistics.quantiles(crop_vals, n=4)[0], 1)
                if len(crop_vals) >= 4 else None
            ),
            "crop_sharp_p50": (
                round(statistics.median(crop_vals), 1) if crop_vals else None
            ),
            "crop_sharp_p75": (
                round(statistics.quantiles(crop_vals, n=4)[2], 1)
                if len(crop_vals) >= 4 else None
            ),
            "crop_sharp_max": (
                round(max(crop_vals), 1) if crop_vals else None
            ),
        }
        rows.append(rec)
        all_frame_sharp.extend(frame_vals)
        all_crop_sharp.extend(crop_vals)
        print(f"  frame p25/50/75={rec['frame_sharp_p25']}/{rec['frame_sharp_p50']}/{rec['frame_sharp_p75']}  "
              f"crop p25/50/75={rec['crop_sharp_p25']}/{rec['crop_sharp_p50']}/{rec['crop_sharp_p75']}",
              flush=True)

    def pct(xs, p):
        if not xs: return None
        xs = sorted(xs)
        idx = int(len(xs) * p / 100)
        return round(xs[min(idx, len(xs)-1)], 1)

    md = [f"# Sharpness audit @ {args.fps} fps\n"]
    md.append(f"_{len(videos)} videos, {len(all_frame_sharp)} frames total_\n")
    md.append(
        f"**Full-frame Laplacian variance:** "
        f"p10={pct(all_frame_sharp,10)} p25={pct(all_frame_sharp,25)} "
        f"p50={pct(all_frame_sharp,50)} p75={pct(all_frame_sharp,75)} "
        f"p90={pct(all_frame_sharp,90)} max={pct(all_frame_sharp,100)}\n"
    )
    md.append(
        f"**Connector-crop Laplacian variance:** "
        f"p10={pct(all_crop_sharp,10)} p25={pct(all_crop_sharp,25)} "
        f"p50={pct(all_crop_sharp,50)} p75={pct(all_crop_sharp,75)} "
        f"p90={pct(all_crop_sharp,90)} max={pct(all_crop_sharp,100)}\n"
    )
    md.append("Recommended threshold = the p10 of the *crop* distribution — "
              "filters the bottom 10% blurriest connectors without nuking everything.\n")
    md.append("\n## Per-video distribution\n")
    md.append("| video | frames | det | frame p25/50/75 | crop p25/50/75 |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['video']} | {r['n_frames']} | {r['n_detected']} | "
            f"{r['frame_sharp_p25']}/{r['frame_sharp_p50']}/{r['frame_sharp_p75']} | "
            f"{r['crop_sharp_p25']}/{r['crop_sharp_p50']}/{r['crop_sharp_p75']} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"per_video": rows,
                    "frame_global": {"p10": pct(all_frame_sharp,10),
                                     "p50": pct(all_frame_sharp,50),
                                     "p90": pct(all_frame_sharp,90)},
                    "crop_global": {"p10": pct(all_crop_sharp,10),
                                    "p50": pct(all_crop_sharp,50),
                                    "p90": pct(all_crop_sharp,90)}},
                   indent=2)
    )
    print(f"\nwrote {args.out}")
    print(
        f"\nRECOMMENDED: --min-sharpness {pct(all_crop_sharp, 10)} "
        f"(p10 of crop distribution, on the actual data)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
