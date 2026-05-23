"""Eval multiple classifier checkpoints against the same holdout.

Each checkpoint is a model dir with weights.pt + labels.json (the
format both our existing auto_retrain and Jerry's training scripts
produce). For each holdout image we run the SAME Hough crop pass
used by the production /predict service, feed the best crop to each
classifier, and tally accuracy + family/gender.

Usage:

  python -m scripts.eval_classifiers \\
      --holdout /opt/rfcai/repo/training/data/test_holdout \\
      --model models/connector_classifier=v18-prod \\
      --model models/aug_resnet18_<ts>=aug-resnet18 \\
      --model models/aug_effnetv2s_<ts>=aug-effnetv2s \\
      --out reports/compare_<ts>.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from rfconnectorai.data_fetch.connector_crops import detect_connector_crops


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
    "1.85mm-M", "1.85mm-F",
]


def _truth_from_path(p: Path) -> str:
    return p.parent.name


def build_model(num_classes: int, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if architecture == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"unknown architecture: {architecture}")


class Classifier:
    """Loads a trained model dir + runs inference on RGB crops."""

    def __init__(self, model_dir: Path, name: str, device: str = "cuda"):
        self.name = name
        self.model_dir = model_dir
        labels = json.loads((model_dir / "labels.json").read_text())
        self.classes: list[str] = labels["class_names"]
        self.architecture = labels.get("architecture", "resnet18")
        self.input_size = labels.get("input_size", 224)
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu",
        )
        self.model = build_model(len(self.classes), self.architecture)
        state = torch.load(
            model_dir / "weights.pt", map_location="cpu", weights_only=False,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict_rgb(self, rgb: np.ndarray) -> tuple[str, float]:
        pil = Image.fromarray(rgb)
        t = self.transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument(
        "--model", action="append", default=[],
        help="path=name — repeat for each checkpoint",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    images = []
    for ext in (".jpg", ".jpeg", ".png"):
        images.extend(args.holdout.rglob(f"*{ext}"))
    images.sort()
    if not images:
        print("no holdout images", file=sys.stderr)
        return 1
    print(f"holdout: {len(images)} images")

    classifiers: list[Classifier] = []
    for spec in args.model:
        path, _, name = spec.partition("=")
        if not name: name = Path(path).name
        try:
            c = Classifier(Path(path), name, device=args.device)
            print(f"loaded {name} ({c.architecture}) from {path}")
            classifiers.append(c)
        except Exception as e:
            print(f"failed to load {path}: {e}")
    if not classifiers:
        print("no classifiers loaded", file=sys.stderr)
        return 1

    # Pre-crop each holdout image once using the same Hough detector
    # production uses; share the crop across all classifiers.
    print("\n-- detecting crops in holdout --")
    crops: list[tuple[Path, np.ndarray | None, str]] = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            crops.append((img_path, None, _truth_from_path(img_path)))
            continue
        results = detect_connector_crops(bgr, max_crops=1)
        if not results:
            crops.append((img_path, None, _truth_from_path(img_path)))
            continue
        rgb = cv2.cvtColor(results[0].crop, cv2.COLOR_BGR2RGB)
        crops.append((img_path, rgb, _truth_from_path(img_path)))
    n_detected = sum(1 for _, c, _ in crops if c is not None)
    print(f"  {n_detected}/{len(crops)} crops detected")

    # Run each classifier over the shared crops.
    summary: list[dict] = []
    for cls_obj in classifiers:
        print(f"\n-- {cls_obj.name} --")
        rows = []
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        lat = []
        for img_path, rgb, truth in crops:
            if rgb is None:
                rows.append({"file": img_path.name, "truth": truth,
                             "pred": None, "conf": 0.0, "lat_ms": 0.0})
                confusion[truth]["(no-crop)"] += 1
                continue
            t0 = time.perf_counter()
            pred, conf = cls_obj.predict_rgb(rgb)
            lat.append((time.perf_counter() - t0) * 1000)
            rows.append({"file": img_path.name, "truth": truth,
                         "pred": pred, "conf": conf,
                         "lat_ms": lat[-1]})
            confusion[truth][pred] += 1
        n = len(rows)
        full = sum(1 for r in rows if r["pred"] == r["truth"])
        fam = 0
        gen = 0
        for r in rows:
            if not r["pred"]: continue
            if "-" not in r["truth"] or "-" not in r["pred"]: continue
            tf, tg = r["truth"].rsplit("-", 1)
            pf, pg = r["pred"].rsplit("-", 1)
            fam += int(tf == pf)
            gen += int(tg == pg)
        summary.append({
            "name": cls_obj.name,
            "architecture": cls_obj.architecture,
            "n": n, "n_detected": n_detected,
            "full": full / n if n else 0.0,
            "family": fam / n if n else 0.0,
            "gender": gen / n if n else 0.0,
            "lat_mean": sum(lat) / len(lat) if lat else 0.0,
            "rows": rows,
            "confusion": {k: dict(v) for k, v in confusion.items()},
        })
        print(f"  Full={full/n*100:.1f}%  Family={fam/n*100:.1f}%  "
              f"Gender={gen/n*100:.1f}%  "
              f"crops_used={n_detected}/{n}  lat_mean={summary[-1]['lat_mean']:.0f}ms")

    # Render markdown.
    md = [f"# Classifier comparison — holdout `{args.holdout}`\n"]
    md.append(f"_{len(images)} images, {n_detected} with detected crops_\n")
    md.append("| model | architecture | Full | Family | Gender | lat |")
    md.append("|---|---|---|---|---|---|")
    for s in summary:
        md.append(
            f"| **{s['name']}** | `{s['architecture']}` | "
            f"{s['full']*100:.1f}% | {s['family']*100:.1f}% | "
            f"{s['gender']*100:.1f}% | {s['lat_mean']:.0f}ms |"
        )
    md.append("\n## Per-class accuracy\n")
    header = ["class"] + [s["name"] for s in summary]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for cls in CANONICAL_CLASSES:
        row = [cls]
        for s in summary:
            n_cls = sum(1 for rr in s["rows"] if rr["truth"] == cls)
            ok = sum(1 for rr in s["rows"]
                     if rr["truth"] == cls and rr["pred"] == cls)
            row.append(
                f"{ok}/{n_cls} = {ok/n_cls*100:.0f}%" if n_cls else "n/a"
            )
        md.append("| " + " | ".join(row) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps(
        {"holdout": str(args.holdout), "summary": summary}, indent=2,
    ))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
