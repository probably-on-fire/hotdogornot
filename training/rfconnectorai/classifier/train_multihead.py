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


def _add_base_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory used to resolve attributes.csv source_image paths "
        "(matches --base-dir passed to build_yolo_dataset). Defaults to --dataset.",
    )


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
    base_dir: Path | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    num_workers: int = 2
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    _add_base_dir_arg(parser)
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
    """Train the multi-head classifier and emit per-sample predictions.

    Imports torch lazily so the local CPU PC can keep using --dry-run
    without torch installed. Outputs in ``cfg.out``:
      best.pt              torch state_dict of the best (highest mean
                           val accuracy across heads) checkpoint.
      metrics.json         per-epoch train/val metrics + per-head val acc.
      head_vocabs.json     vocabulary used at training time.
      predictions.jsonl    test-split predictions in the schema the
                           rfconnectorai.eval harness expects.
      model_record.json    populated registry entry (re-emitted with
                           best-epoch metrics path).
      config.json          snapshot of training config.
    """
    import csv
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from rfconnectorai.classifier.label_encoding import (
        HEAD_ORDER, MISSING_INDEX, default_head_vocabs, read_attributes_csv,
    )
    from rfconnectorai.classifier.model_multihead import build_multihead_classifier
    from rfconnectorai.classifier.multihead_dataset import (
        MultiHeadAttributeDataset,
        collate_multihead,
        make_eval_transforms,
        make_train_transforms,
    )

    # -------------------- dataset prep --------------------
    attributes_csv = cfg.dataset / "attributes.csv"
    rows = read_attributes_csv(attributes_csv)
    if not rows:
        raise RuntimeError(f"no rows in {attributes_csv}")
    families = sorted({row.get("family", "") for row in rows if row.get("family")})
    vocabs = default_head_vocabs(families)
    head_sizes = {head: vocabs[head].num_classes for head in HEAD_ORDER}

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = (row.get("split") or "train").strip()
        if split not in by_split:
            split = "train"
        by_split[split].append(row)
    if not by_split["train"]:
        raise RuntimeError("train split is empty in attributes.csv")
    if not by_split["val"]:
        # Move some training rows to val if the dataset builder did not
        # produce a val split (e.g. tiny datasets). Deterministic.
        n_val = max(1, len(by_split["train"]) // 5)
        by_split["val"] = by_split["train"][:n_val]
        by_split["train"] = by_split["train"][n_val:]

    base_dir = cfg.base_dir or cfg.dataset
    train_tf = make_train_transforms(cfg.imgsz)
    eval_tf = make_eval_transforms(cfg.imgsz)
    train_ds = MultiHeadAttributeDataset(
        rows=by_split["train"], image_root=base_dir, vocabs=vocabs, transform=train_tf,
    )
    val_ds = MultiHeadAttributeDataset(
        rows=by_split["val"], image_root=base_dir, vocabs=vocabs, transform=eval_tf,
    )
    test_ds = MultiHeadAttributeDataset(
        rows=by_split["test"], image_root=base_dir, vocabs=vocabs, transform=eval_tf,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate_multihead, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate_multihead, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate_multihead, pin_memory=True,
    ) if len(test_ds) else None

    # -------------------- model --------------------
    device_str = cfg.device
    if device_str == "0":
        device = torch.device("cuda:0")
    elif device_str.isdigit():
        device = torch.device(f"cuda:{device_str}")
    else:
        device = torch.device(device_str)
    model = build_multihead_classifier(cfg.backbone, head_sizes).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - 1), eta_min=1e-6,
    )
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    def _step(images, targets, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        images = images.to(device, non_blocking=True)
        logits = model(images)
        losses = []
        per_head_correct: dict[str, tuple[int, int]] = {}
        for head, vocab in vocabs.items():
            tgt = targets[head].to(device, non_blocking=True)
            mask = tgt != MISSING_INDEX
            n_valid = int(mask.sum().item())
            if n_valid == 0:
                per_head_correct[head] = (0, 0)
                continue
            head_logits = logits[head][mask]
            head_tgt = tgt[mask]
            losses.append(ce(head_logits, head_tgt))
            preds = head_logits.argmax(dim=1)
            correct = int((preds == head_tgt).sum().item())
            per_head_correct[head] = (correct, n_valid)
        if not losses:
            zero = images.new_tensor(0.0, requires_grad=True) if train else images.new_tensor(0.0)
            return zero, per_head_correct
        loss = torch.stack(losses).mean()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.detach(), per_head_correct

    def _epoch(loader, train: bool) -> dict:
        agg_loss = 0.0
        agg_n = 0
        per_head: dict[str, list[int]] = {h: [0, 0] for h in HEAD_ORDER}
        for images, targets, _ in loader:
            loss, head_stats = _step(images, targets, train=train)
            agg_loss += float(loss.item()) * images.size(0)
            agg_n += images.size(0)
            for h, (c, n) in head_stats.items():
                per_head[h][0] += c
                per_head[h][1] += n
        return {
            "loss": agg_loss / max(agg_n, 1),
            "per_head_acc": {
                h: (per_head[h][0] / per_head[h][1]) if per_head[h][1] else None
                for h in HEAD_ORDER
            },
            "n_samples": agg_n,
        }

    history: list[dict] = []
    best_score = -1.0
    best_path = cfg.out / "best.pt"
    cfg.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _epoch(train_loader, train=True)
        with torch.no_grad():
            val_metrics = _epoch(val_loader, train=False)
        scheduler.step()
        valid_accs = [v for v in val_metrics["per_head_acc"].values() if v is not None]
        mean_val_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "mean_val_acc": mean_val_acc,
        }
        history.append(record)
        print(
            f"epoch {epoch:>2}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.3f} "
            f"val_loss={val_metrics['loss']:.3f} "
            f"mean_val_acc={mean_val_acc:.3f}"
        )
        if mean_val_acc > best_score:
            best_score = mean_val_acc
            torch.save(
                {"state_dict": model.state_dict(), "head_sizes": head_sizes,
                 "vocabs": {h: list(v.values) for h, v in vocabs.items()},
                 "imgsz": cfg.imgsz, "backbone": cfg.backbone},
                best_path,
            )

    # -------------------- predictions on test split --------------------
    predictions_path = cfg.out / "predictions.jsonl"
    n_predictions = 0
    if test_loader is not None:
        # Reload best weights for predictions.
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad(), open(predictions_path, "w", encoding="utf-8") as f:
            for batch_idx, (images, targets, instance_ids) in enumerate(test_loader):
                logits = model(images.to(device))
                for i, instance_id in enumerate(instance_ids):
                    pred_row: dict[str, Any] = {
                        "instance_id": instance_id,
                        "split": "test",
                        "ground_truth": {},
                        "prediction": {},
                        "confidences": {},
                        "abstain": {},
                    }
                    for head, vocab in vocabs.items():
                        head_logits = logits[head][i]
                        probs = torch.softmax(head_logits, dim=0)
                        pred_idx = int(torch.argmax(probs).item())
                        confidence = float(probs[pred_idx].item())
                        pred_row["prediction"][head] = vocab.value_of(pred_idx)
                        pred_row["confidences"][head] = confidence
                        tgt_idx = int(targets[head][i].item())
                        if tgt_idx != MISSING_INDEX:
                            pred_row["ground_truth"][head] = vocab.value_of(tgt_idx)
                    f.write(json.dumps(pred_row, sort_keys=True) + "\n")
                    n_predictions += 1

    # -------------------- artifacts --------------------
    cfg.artifact_out.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        artifact_path = cfg.artifact_out / "best.pt"
        artifact_path.write_bytes(best_path.read_bytes())

    metrics_path = cfg.out / "metrics.json"
    metrics_path.write_text(
        json.dumps({"history": history, "best_mean_val_acc": best_score}, indent=2),
        encoding="utf-8",
    )

    print(f"best checkpoint: {best_path}")
    print(f"predictions:     {predictions_path} ({n_predictions} rows)")
    print(f"metrics:         {metrics_path}")
    return best_path


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
        base_dir=args.base_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
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
