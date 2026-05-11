"""Reproducible holdout eval against a running predict service.

Drops a JSON + Markdown report in --out so we have a paper trail of
every model/config change. Catches the kind of silent regressions we
hit with the labels.json inversion bug — if production drifts from
the previous best, this script makes the drift visible at the next
deploy gate instead of weeks later.

Usage::

    # On the box, against the local service:
    python scripts/eval_holdout.py \
        --url http://127.0.0.1:8503/predict \
        --token "$RFCAI_DEVICE_TOKEN" \
        --holdout data/test_holdout \
        --out reports/holdout_$(date -u +%Y%m%dT%H%M%S)

    # Against production through the relay:
    python scripts/eval_holdout.py \
        --url https://aired.com/rfcai/predict \
        --token <device-token> \
        --holdout data/test_holdout \
        --out reports/holdout_prod
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def _truth_from_path(p: Path) -> tuple[str, str, str]:
    """`data/test_holdout/2.4mm-M/IMG_0270.jpeg` →
    ('2.4mm-M', '2.4mm', 'M')."""
    cls = p.parent.name
    if "-" in cls:
        fam, gen = cls.rsplit("-", 1)
    else:
        fam, gen = cls, ""
    return cls, fam, gen


def run(url: str, token: str, holdout: Path) -> dict:
    images = sorted(holdout.rglob("*.jpeg")) + sorted(holdout.rglob("*.jpg"))
    rows: list[dict] = []
    n_full = n_fam = n_gen = 0
    n_detected = 0
    latencies_ms: list[float] = []
    for img in images:
        truth, t_fam, t_gen = _truth_from_path(img)
        t0 = time.perf_counter()
        with open(img, "rb") as fh:
            r = requests.post(
                url,
                headers={"X-Device-Token": token} if token else {},
                files={"image": (img.name, fh, "image/jpeg")},
                timeout=60,
            )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        try:
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            rows.append({"file": img.name, "truth": truth, "error": str(e)})
            continue
        preds = body.get("predictions") or []
        if not preds:
            rows.append({
                "file": img.name, "truth": truth,
                "pred": None, "detected": False,
            })
            continue
        top = max(preds, key=lambda p: p.get("confidence", 0.0))
        pred_class = top.get("class_name", "")
        pred_fam = top.get("family") or (
            pred_class.rsplit("-", 1)[0] if "-" in pred_class else ""
        )
        pred_gen = top.get("gender") or (
            pred_class.rsplit("-", 1)[1] if "-" in pred_class else ""
        )
        n_detected += 1
        if pred_class == truth:
            n_full += 1
        if pred_fam == t_fam:
            n_fam += 1
        if pred_gen == t_gen:
            n_gen += 1
        rows.append({
            "file": img.name, "truth": truth,
            "pred": pred_class,
            "pred_family": pred_fam,
            "pred_gender": pred_gen,
            "confidence": top.get("confidence"),
            "family_confidence": top.get("family_confidence"),
            "gender_confidence": top.get("gender_confidence"),
            "detected": True,
            "crop_source": (top.get("_diag") or {}).get("crop_source"),
        })

    n = len(images)
    return {
        "n_images": n,
        "n_detected": n_detected,
        "full_correct": n_full,
        "family_correct": n_fam,
        "gender_correct": n_gen,
        "accuracy": {
            "full": n_full / n if n else 0.0,
            "family": n_fam / n if n else 0.0,
            "gender": n_gen / n if n else 0.0,
            "detect_recall": n_detected / n if n else 0.0,
        },
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0,
            "max": max(latencies_ms) if latencies_ms else 0,
        },
        "url": url,
        "holdout": str(holdout),
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    acc = report["accuracy"]
    lat = report["latency_ms"]
    lines = [
        "# Holdout eval",
        "",
        f"- url: `{report['url']}`",
        f"- holdout: `{report['holdout']}`",
        f"- images: {report['n_images']} (detected: {report['n_detected']})",
        f"- accuracy: Full {acc['full']*100:.1f}% · "
        f"Family {acc['family']*100:.1f}% · "
        f"Gender {acc['gender']*100:.1f}% · "
        f"Detect-recall {acc['detect_recall']*100:.1f}%",
        f"- latency mean/max: {lat['mean']:.0f} ms / {lat['max']:.0f} ms",
        "",
        "| truth | pred | family | gender | conf | crop |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        if "error" in r:
            lines.append(f"| {r['truth']} | ERR | | | | |")
            continue
        if not r["detected"]:
            lines.append(f"| {r['truth']} | (none) | | | | |")
            continue
        conf_s = f"{r['confidence']:.2f}" if r.get("confidence") is not None else ""
        lines.append(
            f"| {r['truth']} | {r['pred']} | {r['pred_family']} | "
            f"{r['pred_gender']} | {conf_s} | {r.get('crop_source') or ''} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Predict endpoint URL")
    p.add_argument("--token", default="", help="X-Device-Token value")
    p.add_argument("--holdout", type=Path, required=True,
                   help="Holdout root with <class>/<file>.jpg layout")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory; will receive report.json + report.md")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = run(args.url, args.token, args.holdout)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "report.md").write_text(render_markdown(report))
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
