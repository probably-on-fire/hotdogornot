"""YOLO connector-detector training entry point.

This script is configuration-validated, model-record-emitting, and runs in
the cloud. Local CPU-only environments can call ``--dry-run`` to verify
config correctness without invoking ``ultralytics``.

Usage::

    python -m rfconnectorai.detector.train_yolo \\
        --data datasets/rfconnectors/data.yaml \\
        --model yolo11n.pt \\
        --epochs 100 --imgsz 640 --batch 16 --device 0 \\
        --out reports/experiments/detector_2026_05_10 \\
        --artifact-out models/detector \\
        --dry-run

Heavy training is deliberately delegated to Kaggle/Colab. This module's
unit tests cover only configuration parsing and record emission.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfconnectorai.models.registry import make_model_record, write_record


SUPPORTED_MODELS = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "rtdetr-l.pt",
)

DEFAULT_OUT_NAME = "connector"


@dataclass(frozen=True)
class TrainerConfig:
    data: Path
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    out: Path
    artifact_out: Path
    dataset_lock: Path | None = None
    seq: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO connector detector")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml path")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help=f"Base weights. Supported: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device string. Use 'cpu' on machines without GPU.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Run output dir")
    parser.add_argument(
        "--artifact-out",
        type=Path,
        required=True,
        help="Where to copy best.pt after training",
    )
    parser.add_argument(
        "--dataset-lock",
        type=Path,
        default=None,
        help="Optional path to datasets/rfconnectors/dataset.lock.json so the "
        "model_record records exactly which dataset was used.",
    )
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and emit model_record metadata without training.",
    )
    return parser.parse_args(argv)


def validate_config(cfg: TrainerConfig) -> None:
    if not cfg.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {cfg.data}")
    if cfg.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"model {cfg.model!r} not in supported set {SUPPORTED_MODELS}"
        )
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive; got {cfg.epochs}")
    if cfg.imgsz <= 0:
        raise ValueError(f"imgsz must be positive; got {cfg.imgsz}")
    if cfg.batch <= 0:
        raise ValueError(f"batch must be positive; got {cfg.batch}")


def _read_dataset_lock(lock_path: Path | None) -> tuple[str, str]:
    """Return (dataset_id, taxonomy_sha256) from a dataset.lock.json file."""
    if lock_path is None or not lock_path.exists():
        return ("unknown_dataset", "unknown_taxonomy")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    return (
        str(payload.get("dataset_id", "unknown_dataset")),
        str(payload.get("taxonomy_sha256", "unknown_taxonomy")),
    )


def emit_run_metadata(cfg: TrainerConfig) -> dict[str, Any]:
    """Write the per-run metadata files into ``cfg.out``.

    Always written, including in dry-run, so the user can see exactly what
    a real run would record.
    """
    cfg.out.mkdir(parents=True, exist_ok=True)
    dataset_id, taxonomy_sha = _read_dataset_lock(cfg.dataset_lock)
    record = make_model_record(
        model_type="detector",
        architecture=cfg.model.replace(".pt", ""),
        trained_on=f"datasets/rfconnectors@{dataset_id}",
        taxonomy_version=taxonomy_sha,
        metrics_path=str(cfg.out / "metrics.json"),
        artifact_path=str(cfg.artifact_out / "best.pt"),
        seq=cfg.seq,
        extra={
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "device": cfg.device,
            **cfg.extra,
        },
    )
    write_record(record, cfg.out / "model_record.json")

    config_snapshot = {
        "data": str(cfg.data),
        "model": cfg.model,
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "artifact_out": str(cfg.artifact_out),
        "dataset_lock": str(cfg.dataset_lock) if cfg.dataset_lock else None,
        "seq": cfg.seq,
    }
    (cfg.out / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"record": record.to_dict(), "config": config_snapshot}


def run_training(cfg: TrainerConfig) -> Path:
    """Invoke ``ultralytics`` and copy the best weights to ``cfg.artifact_out``."""
    try:
        from ultralytics import YOLO  # imported lazily; not required for tests
    except ImportError as exc:  # pragma: no cover - cloud only
        raise RuntimeError(
            "ultralytics not installed; install in cloud env before training"
        ) from exc

    project = cfg.out / "ultralytics"
    model = YOLO(cfg.model)
    model.train(
        data=str(cfg.data.resolve()),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(project),
        name=DEFAULT_OUT_NAME,
        exist_ok=True,
        verbose=True,
    )
    best = project / DEFAULT_OUT_NAME / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training completed but best weights missing at {best}")
    cfg.artifact_out.mkdir(parents=True, exist_ok=True)
    dest = cfg.artifact_out / "best.pt"
    shutil.copy2(best, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = TrainerConfig(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        out=args.out,
        artifact_out=args.artifact_out,
        dataset_lock=args.dataset_lock,
        seq=args.seq,
    )
    validate_config(cfg)
    metadata = emit_run_metadata(cfg)

    if args.dry_run:
        print(json.dumps({"dry_run": True, **metadata}, indent=2, sort_keys=True))
        return 0

    artifact = run_training(cfg)
    print(f"detector best weights: {artifact}")
    print(f"run metadata: {cfg.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
