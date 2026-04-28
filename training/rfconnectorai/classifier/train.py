"""
Train a small image classifier on the labeled connector folders.

Architecture: ResNet-18 pretrained on ImageNet, fine-tune the final FC layer
to N classes. Roughly the right size for 8 classes × 50–100 images per class
on CPU — trains in <10 minutes with reasonable transforms.

Usage:
    python -m rfconnectorai.classifier.train \\
        --data-dir data/labeled/embedder \\
        --out-dir models/connector_classifier \\
        --epochs 8

Outputs to `out_dir`:
    weights.pt        — torch state_dict of the fine-tuned model
    labels.json       — {"class_names": [...], "input_size": 224, ...}
    metrics.json      — train/val loss + accuracy per epoch

The predict module loads from this same directory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from rfconnectorai.classifier.dataset import (
    INPUT_SIZE,
    ConnectorFolderDataset,
    make_eval_transforms,
    make_train_transforms,
)


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    class_names: list[str]
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 3e-4
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(num_classes: int) -> nn.Module:
    """ResNet-18 with the final layer swapped for num_classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    return indices[n_val:], indices[:n_val]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One pass through `loader`. If optimizer is None, runs eval (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_ctx = torch.enable_grad if is_train else torch.no_grad
    with grad_ctx():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(dim=1)
            total_loss += float(loss.detach()) * inputs.size(0)
            total_correct += int((preds == targets).sum().item())
            total_samples += inputs.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def train(config: TrainConfig) -> dict:
    """Train the classifier per `config`. Returns the metrics dict written to disk."""
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ConnectorFolderDataset(
        root=config.data_dir,
        class_names=config.class_names,
        transform=make_train_transforms(),
    )
    eval_ds = ConnectorFolderDataset(
        root=config.data_dir,
        class_names=config.class_names,
        transform=make_eval_transforms(),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f"no labeled images found under {config.data_dir} for classes "
            f"{config.class_names}"
        )

    train_idx, val_idx = _split_indices(len(train_ds), config.val_fraction, config.seed)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=config.batch_size, shuffle=False, num_workers=0,
    )

    model = _build_model(num_classes=len(config.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    history: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        history.append(EpochMetrics(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
        ))
        print(
            f"epoch {epoch:>2}/{config.epochs}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
        )

    weights_path = config.out_dir / "weights.pt"
    labels_path = config.out_dir / "labels.json"
    metrics_path = config.out_dir / "metrics.json"

    torch.save(model.state_dict(), weights_path)
    labels_path.write_text(json.dumps({
        "class_names": config.class_names,
        "input_size": INPUT_SIZE,
        "architecture": "resnet18",
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
        "class_counts": ConnectorFolderDataset(
            config.data_dir, config.class_names
        ).class_counts(),
    }, indent=2))
    metrics_blob = {"history": [asdict(m) for m in history]}
    metrics_path.write_text(json.dumps(metrics_blob, indent=2))

    # Export ONNX alongside the .pt — the AR app loads ONNX via Sentis.
    from rfconnectorai.classifier.export_onnx import export_to_onnx
    onnx_path = config.out_dir / "weights.onnx"
    try:
        export_to_onnx(config.out_dir, onnx_path)
    except Exception as e:
        # Don't fail the whole train if ONNX export breaks; the .pt is still
        # usable from Python and the next retrain can retry the export.
        print(f"warning: ONNX export failed: {e}")

    # Snapshot versioned files + refresh the OTA manifest. Relay reads
    # manifest.json to advertise a new version to the app.
    from rfconnectorai.classifier.versioning import bump_version
    final_val_acc = history[-1].val_acc if history else None
    bump_version(
        model_dir=config.out_dir,
        weights_path=weights_path,
        val_acc=final_val_acc,
        n_train_samples=len(train_idx),
    )

    return metrics_blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=[
        "SMA-M", "SMA-F",
        "3.5mm-M", "3.5mm-F",
        "2.92mm-M", "2.92mm-F",
        "2.4mm-M", "2.4mm-F",
    ])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        class_names=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_fraction=args.val_fraction,
    )
    train(config)


if __name__ == "__main__":
    main()
