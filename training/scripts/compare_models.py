"""Eval N classifiers against the same holdout, emit a comparison
report.

Supports two eval backends:
  * HTTP — POST each image to a /predict URL (legacy ResNet-18 path
    and any future model that ships behind the existing FastAPI
    endpoint).
  * Local ONNX — for the YOLO11n + EfficientNetV2-S pipeline; uses
    rfconnectorai.pipeline.jerry_pipeline.JerryPipeline.

Each model is described by a YAML/JSON config; for the basic case
just pass them via --http URL and --onnx PATH on the command line.

Output: training/reports/compare_<date>.md with a side-by-side
table, per-class accuracy, and confusion-matrix delta.

Example:

    python -m scripts.compare_models \
        --holdout data/test_holdout \
        --http v18=https://aired.com/rfcai/predict \
        --onnx jerry_v0=/opt/rfcai/repo/training/models/jerry \
        --onnx jerry_v1=/opt/rfcai/repo/training/models/jerry_v1 \
        --out reports/compare_$(date -u +%Y%m%dT%H%M%S).md
"""
from __future__ import annotations

import argparse
import json
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
DEVICE_TOKEN_DEFAULT = (
    "66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1"
)


def _truth_from_path(p: Path) -> str:
    """`data/test_holdout/2.4mm-M/IMG_0270.jpeg` → `2.4mm-M`."""
    return p.parent.name


# ---------------------------------------------------------------- backends

class HttpBackend:
    """Hit a /predict URL — same shape as predict_service.py."""

    def __init__(self, name: str, url: str, device_token: str):
        self.name = name
        self.url = url
        self.device_token = device_token
        import requests
        self._sess = requests.Session()

    def predict(self, img_path: Path) -> dict:
        with open(img_path, "rb") as fh:
            t0 = time.perf_counter()
            r = self._sess.post(
                self.url,
                headers={"X-Device-Token": self.device_token},
                files={"image": (img_path.name, fh, "image/jpeg")},
                timeout=120,
            )
            dt_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        preds = body.get("predictions") or []
        if not preds:
            return {"class_name": None, "confidence": 0.0, "lat_ms": dt_ms}
        top = max(preds, key=lambda p: p.get("confidence", 0.0))
        return {
            "class_name": top.get("class_name"),
            "confidence": float(top.get("confidence", 0.0)),
            "lat_ms": dt_ms,
        }


class OnnxBackend:
    """Use the locally-instantiated JerryPipeline for inference."""

    def __init__(self, name: str, model_dir: Path):
        from rfconnectorai.pipeline.jerry_pipeline import JerryPipeline
        import cv2
        self._cv2 = cv2
        self.name = name
        self.pipe = JerryPipeline(Path(model_dir))

    def predict(self, img_path: Path) -> dict:
        bgr = self._cv2.imread(str(img_path))
        if bgr is None:
            return {"class_name": None, "confidence": 0.0, "lat_ms": 0.0}
        t0 = time.perf_counter()
        results = self.pipe.run(bgr)
        dt_ms = (time.perf_counter() - t0) * 1000
        if not results:
            return {"class_name": None, "confidence": 0.0, "lat_ms": dt_ms}
        top = results[0]   # already best-first
        return {
            "class_name": top["class_name"],
            "confidence": float(top["confidence"]),
            "lat_ms": dt_ms,
        }


# ---------------------------------------------------------------- harness

def evaluate(backend, images: list[Path]) -> dict:
    rows: list[dict] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lat_ms: list[float] = []
    for img in images:
        truth = _truth_from_path(img)
        try:
            pred = backend.predict(img)
        except Exception as e:
            rows.append({"file": img.name, "truth": truth, "error": str(e)})
            continue
        rows.append({
            "file": img.name, "truth": truth,
            "pred": pred["class_name"],
            "confidence": pred["confidence"],
            "lat_ms": pred["lat_ms"],
        })
        lat_ms.append(pred["lat_ms"])
        if pred["class_name"]:
            confusion[truth][pred["class_name"]] += 1
        else:
            confusion[truth]["(none)"] += 1
    n = len(rows)
    correct_full = sum(
        1 for r in rows
        if r.get("pred") and r["pred"] == r["truth"]
    )
    correct_family = 0
    correct_gender = 0
    for r in rows:
        if not r.get("pred"): continue
        if "-" in r["truth"] and "-" in r["pred"]:
            tf, tg = r["truth"].rsplit("-", 1)
            pf, pg = r["pred"].rsplit("-", 1)
            correct_family += int(tf == pf)
            correct_gender += int(tg == pg)
    return {
        "name": backend.name,
        "n": n,
        "full": correct_full / n if n else 0.0,
        "family": correct_family / n if n else 0.0,
        "gender": correct_gender / n if n else 0.0,
        "lat_mean": (sum(lat_ms) / len(lat_ms)) if lat_ms else 0.0,
        "lat_max": max(lat_ms) if lat_ms else 0.0,
        "rows": rows,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def render(results: list[dict], holdout_root: Path) -> str:
    lines: list[str] = []
    lines.append(f"# Model comparison — holdout `{holdout_root}`\n")
    lines.append(
        "| model | Full | Family | Gender | latency mean/max | n |\n"
        "|---|---|---|---|---|---|"
    )
    for r in results:
        lines.append(
            f"| **{r['name']}** | {r['full']*100:.1f}% | "
            f"{r['family']*100:.1f}% | {r['gender']*100:.1f}% | "
            f"{r['lat_mean']:.0f} / {r['lat_max']:.0f} ms | {r['n']} |"
        )

    # Per-class accuracy table.
    lines.append("\n## Per-class accuracy\n")
    header = ["class"] + [r["name"] for r in results]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cls in CANONICAL_CLASSES:
        row = [cls]
        for r in results:
            n_cls = sum(1 for rr in r["rows"] if rr["truth"] == cls)
            ok = sum(
                1 for rr in r["rows"]
                if rr["truth"] == cls and rr.get("pred") == cls
            )
            row.append(
                f"{ok}/{n_cls} = {ok/n_cls*100:.0f}%" if n_cls else "n/a"
            )
        lines.append("| " + " | ".join(row) + " |")

    # Confusion matrix per model.
    for r in results:
        lines.append(f"\n## Confusion matrix — {r['name']}\n")
        truths = sorted(r["confusion"].keys())
        preds = sorted({p for d in r["confusion"].values() for p in d})
        lines.append("| truth ↓  pred → | " + " | ".join(preds) + " |")
        lines.append("|" + "|".join(["---"] * (len(preds) + 1)) + "|")
        for t in truths:
            row = [t] + [str(r["confusion"][t].get(p, "")) for p in preds]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, required=True,
                    help="Holdout root: <cls>/<file>.jpg layout")
    ap.add_argument(
        "--http", action="append", default=[],
        help="name=URL — evaluate via /predict",
    )
    ap.add_argument(
        "--onnx", action="append", default=[],
        help="name=path-to-jerry-model-dir — evaluate via JerryPipeline",
    )
    ap.add_argument(
        "--device-token", default=DEVICE_TOKEN_DEFAULT,
        help="X-Device-Token for HTTP backends",
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="Markdown report destination")
    args = ap.parse_args()

    images: list[Path] = []
    for ext in (".jpg", ".jpeg", ".png"):
        images.extend(args.holdout.rglob(f"*{ext}"))
    images.sort()
    if not images:
        print("no holdout images found", file=sys.stderr)
        return 1
    print(f"evaluating {len(images)} images")

    backends: list = []
    for spec in args.http:
        name, _, url = spec.partition("=")
        backends.append(HttpBackend(name, url, args.device_token))
    for spec in args.onnx:
        name, _, mdir = spec.partition("=")
        backends.append(OnnxBackend(name, Path(mdir)))

    results: list[dict] = []
    for b in backends:
        print(f"=== {b.name} ===")
        r = evaluate(b, images)
        results.append(r)
        print(
            f"  Full={r['full']*100:.1f}%  Family={r['family']*100:.1f}%  "
            f"Gender={r['gender']*100:.1f}%  "
            f"lat={r['lat_mean']:.0f}/{r['lat_max']:.0f}ms"
        )

    md = render(results, args.holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps(
        {"holdout": str(args.holdout), "results": results}, indent=2,
    ))
    print(f"\nwrote {args.out} and {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
