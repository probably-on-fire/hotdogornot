"""Multi-head classifier training entry point.

Cloud-only entry. The local CPU PC should only invoke ``--dry-run`` to
verify config and emit metadata without importing torchvision weights.

Usage::

    python -m rfconnectorai.classifier.train_multihead \\
        --dataset datasets/rfconnectors \\
        --backbone efficientnet_v2_s \\
        --epochs 80 \\
        --batch 64 \\
        --imgsz 384 \\
        --device 0 \\
        --out reports/experiments/multihead_2026_05_10 \\
        --artifact-out models/multihead_classifier
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfconnectorai.classifier.label_encoding import (
    HEAD_ORDER,
    HeadVocab,
    MISSING_INDEX,
    default_head_vocabs,
    encode_attributes,
    read_attributes_csv,
)
from rfconnectorai.classifier.model_multihead import SUPPORTED_BACKBONES
from rfconnectorai.models.registry import make_model_record, write_record


@dataclass(frozen=True)
class MultiHeadTrainerConfig:
    dataset: Path
    backbone: str
    epochs: int
    batch: int
    imgsz: int
    device: str
    out: Path
    artifact_out: Path
    seq: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-head connector classifier")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        default="efficientnet_v2_s",
        help=f"Supported: {', '.join(SUPPORTED_BACKBONES)}",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_config(cfg: MultiHeadTrainerConfig) -> None:
    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone {cfg.backbone!r} not in supported {SUPPORTED_BACKBONES}"
        )
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive; got {cfg.epochs}")
    if cfg.batch <= 0:
        raise ValueError(f"batch must be positive; got {cfg.batch}")
    if cfg.imgsz <= 0:
        raise ValueError(f"imgsz must be positive; got {cfg.imgsz}")


def head_sizes_for_dataset(dataset_dir: Path) -> tuple[dict[str, int], dict[str, HeadVocab]]:
    attributes_csv = dataset_dir / "attributes.csv"
    if not attributes_csv.exists():
        raise FileNotFoundError(
            f"attributes.csv not found at {attributes_csv}; run build_yolo_dataset first"
        )
    rows = read_attributes_csv(attributes_csv)
    families = sorted({row.get("family", "") for row in rows if row.get("family")})
    vocabs = default_head_vocabs(families)
    sizes = {head: vocabs[head].num_classes for head in HEAD_ORDER}
    return sizes, vocabs


def emit_run_metadata(cfg: MultiHeadTrainerConfig) -> dict[str, Any]:
    cfg.out.mkdir(parents=True, exist_ok=True)
    sizes, vocabs = head_sizes_for_dataset(cfg.dataset)

    lock_path = cfg.dataset / "dataset.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        dataset_id = str(lock.get("dataset_id", "unknown_dataset"))
        taxonomy_sha = str(lock.get("taxonomy_sha256", "unknown_taxonomy"))
    else:
        dataset_id = "unknown_dataset"
        taxonomy_sha = "unknown_taxonomy"

    record = make_model_record(
        model_type="multihead_classifier",
        architecture=cfg.backbone,
        trained_on=f"datasets/rfconnectors@{dataset_id}",
        taxonomy_version=taxonomy_sha,
        metrics_path=str(cfg.out / "metrics.json"),
        artifact_path=str(cfg.artifact_out / "best.pt"),
        seq=cfg.seq,
        extra={
            "epochs": cfg.epochs,
            "batch": cfg.batch,
            "imgsz": cfg.imgsz,
            "device": cfg.device,
            "head_sizes": sizes,
            **cfg.extra,
        },
    )
    write_record(record, cfg.out / "model_record.json")

    snapshot = {
        "dataset": str(cfg.dataset),
        "backbone": cfg.backbone,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "imgsz": cfg.imgsz,
        "device": cfg.device,
        "artifact_out": str(cfg.artifact_out),
        "seq": cfg.seq,
    }
    (cfg.out / "config.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    head_vocab_payload = {head: list(vocab.values) for head, vocab in vocabs.items()}
    (cfg.out / "head_vocabs.json").write_text(
        json.dumps(head_vocab_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"record": record.to_dict(), "config": snapshot, "head_sizes": sizes}


def masked_cross_entropy_step(
    logits_per_head: dict,
    targets_per_head: dict,
    *,
    missing_index: int = MISSING_INDEX,
):  # pragma: no cover - cloud only
    """Compute mean cross-entropy across heads, ignoring missing labels.

    Centralized so the train loop and any future fine-tuning script share one
    implementation. Imported lazily because torch is cloud-only.
    """
    import torch  # type: ignore[import-not-found]
    from torch.nn import functional as F  # type: ignore[import-not-found]

    losses = []
    for head_name, logits in logits_per_head.items():
        targets = targets_per_head[head_name]
        mask = targets != missing_index
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(logits[mask], targets[mask])
        losses.append(loss)
    if not losses:
        # Use the same dtype/device as a logits tensor.
        any_logits = next(iter(logits_per_head.values()))
        return any_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def run_training(cfg: MultiHeadTrainerConfig) -> Path:  # pragma: no cover - cloud only
    raise NotImplementedError(
        "run_training() is intentionally a stub on the local PC. The full "
        "training loop runs in Kaggle/Colab. Use --dry-run locally."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = MultiHeadTrainerConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        out=args.out,
        artifact_out=args.artifact_out,
        seq=args.seq,
    )
    validate_config(cfg)
    metadata = emit_run_metadata(cfg)

    if args.dry_run:
        print(json.dumps({"dry_run": True, **metadata}, indent=2, sort_keys=True))
        return 0

    run_training(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
