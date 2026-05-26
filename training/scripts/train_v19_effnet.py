"""Train EffNetV2-S at 384 input on the Sonnet-filtered snapshot.

Standalone trainer modeled after rfconnectorai/classifier/train.py but with:
- torchvision.models.efficientnet_v2_s (ImageNet pretrained)
- INPUT_SIZE=384 (matches Jerry's pretrained pipeline at /home/rfcai/training/models/jerry/)
- PIL.BILINEAR resampling (matches Jerry's preprocessing — cv2.INTER_LINEAR
  differs in sampling-center convention and costs ~14pts on fine-pitch faces)
- balance-to-smallest sampling (each class capped at the smallest count
  before training, eliminating SMA-F-dominated batch composition)

Reads from a flat per-class directory (e.g. the train/ subdir of a
build_augmented_split snapshot). Writes weights.pt + weights.onnx +
labels.json + manifest.json + metrics.json to --out-dir.

Usage on the box:

    sudo -u rfcai /opt/rfcai/training/.venv/bin/python -m \
        scripts.train_v19_effnet \
        --data-dir /opt/rfcai/repo/training/data/snapshots/augmented_2026-05-24_v19_sonnet/train \
        --out-dir /home/rfcai/training/models/classifier_v19_effnet \
        --epochs 12 --batch-size 8 --balance-to-smallest --seed 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
    "1.85mm-M", "1.85mm-F",
]
MIN_SAMPLES_PER_CLASS = 5
INPUT_SIZE = 384
VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

log = logging.getLogger("train_v19_effnet")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_train_transforms() -> transforms.Compose:
    """Augmentations matching train.py's aggressive setup, but at INPUT_SIZE=384.
    PIL.BILINEAR is the default for transforms.Resize since torchvision 0.13.
    """
    return transforms.Compose([
        transforms.Resize(int(INPUT_SIZE * 1.14), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.55, 1.0), ratio=(0.85, 1.18),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.05), ratio=(0.5, 2.0)),
    ])


def _make_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(INPUT_SIZE * 1.14), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FolderDataset(Dataset):
    """Flat per-class folder dataset. Mirrors ConnectorFolderDataset but
    inlined to avoid importing INPUT_SIZE=224 transforms from production code."""
    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            d = self.root / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def class_counts(self) -> dict[str, int]:
        out = {c: 0 for c in self.class_names}
        for _, y in self.samples:
            out[self.class_names[y]] += 1
        return out


def _build_model(num_classes: int) -> nn.Module:
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class _NormalizedClassifier(nn.Module):
    """Bake ImageNet normalization into the ONNX graph so the inference path
    feeds raw [0,1] NCHW and the model handles mean/std internally. Matches
    train.py's export pattern."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.base((x - self.mean) / self.std)


def _populated_classes(root: Path, candidates: list[str], min_samples: int) -> list[str]:
    out = []
    for c in candidates:
        d = root / c
        n = sum(1 for p in d.glob("*.[jpJP]*[gG]")) if d.is_dir() else 0
        if n >= min_samples:
            out.append(c)
    return out


def _balanced_subset_indices(dataset: FolderDataset, seed: int) -> list[int]:
    """Subsample each class to the smallest class's count."""
    by_class: dict[int, list[int]] = {}
    for i, (_, y) in enumerate(dataset.samples):
        by_class.setdefault(y, []).append(i)
    smallest = min(len(v) for v in by_class.values())
    rng = random.Random(seed)
    out = []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        out.extend(idxs[:smallest])
    return out


def train(cfg):
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    classes = _populated_classes(cfg.data_dir, CANONICAL_CLASSES, MIN_SAMPLES_PER_CLASS)
    log.info("training on %d classes: %s", len(classes), ", ".join(classes))

    train_dataset = FolderDataset(cfg.data_dir, classes, transform=_make_train_transforms())
    eval_dataset_all = FolderDataset(cfg.data_dir, classes, transform=_make_eval_transforms())
    log.info("total samples in data-dir: %d", len(train_dataset))

    # Carve val before any balancing (so val reflects natural class distribution)
    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(len(train_dataset))
    rng.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * cfg.val_fraction))
    val_idx = all_idx[:n_val].tolist()
    train_idx = all_idx[n_val:].tolist()
    train_idx_set = set(train_idx)

    if cfg.balance_to_smallest:
        balanced = _balanced_subset_indices(train_dataset, cfg.seed)
        train_idx = [i for i in balanced if i in train_idx_set]
        log.info("balance_to_smallest: %d -> %d training samples", len(all_idx) - n_val, len(train_idx))

    # WRS for any residual imbalance
    train_labels = [train_dataset.samples[i][1] for i in train_idx]
    cls_counts = np.bincount(train_labels, minlength=len(classes))
    cls_w = 1.0 / np.maximum(cls_counts, 1)
    sample_w = [cls_w[y] for y in train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), batch_size=cfg.batch_size,
        sampler=sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_dataset_all, val_idx), batch_size=cfg.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = _build_model(len(classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    weights_best_path = cfg.out_dir / "weights.0001.pt"
    weights_last_path = cfg.out_dir / "weights_last.pt"
    metrics_path = cfg.out_dir / "metrics.json"

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tl, ta, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            ta += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        train_loss, train_acc = tl / max(1, n), ta / max(1, n)

        model.eval()
        vl, va, nv = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                vl += loss.item() * x.size(0)
                va += (logits.argmax(1) == y).sum().item()
                nv += x.size(0)
        val_loss, val_acc = vl / max(1, nv), va / max(1, nv)
        sched.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        # Crash-safe per-epoch persistence. Jerry's train.py does the same so a
        # killed Kaggle session doesn't lose the whole run.
        metrics_path.write_text(json.dumps({"history": history}, indent=2))
        torch.save(model.state_dict(), weights_last_path)
        # Keep-best: weights.0001.pt tracks the best-val-acc epoch (Jerry's
        # convention — what export_onnx / eval / the app load).
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), weights_best_path)

        log.info(
            "epoch %2d/%d lr=%.6f train_loss=%.3f train_acc=%.3f val_loss=%.3f val_acc=%.3f%s",
            epoch, cfg.epochs, opt.param_groups[0]["lr"], train_loss, train_acc,
            val_loss, val_acc, "  <- best" if improved else "",
        )

    dt = time.time() - t0
    log.info("train complete in %.1fs; best epoch %d/%d val_acc=%.3f",
             dt, best_epoch, cfg.epochs, best_val_acc)

    # Reload the best-val-acc checkpoint for ONNX export (Jerry's convention).
    model.load_state_dict(torch.load(weights_best_path, map_location=device))
    # Mirror the canonical names train.py also writes.
    for name in ("weights.pt", "weights.latest.pt"):
        (cfg.out_dir / name).write_bytes(weights_best_path.read_bytes())

    # Export ONNX with baked-in normalization
    wrapped = _NormalizedClassifier(model.cpu().eval())
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    onnx_path = cfg.out_dir / "weights.0001.onnx"
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    for name in ("weights.onnx", "weights.latest.onnx"):
        (cfg.out_dir / name).write_bytes(onnx_path.read_bytes())

    # labels.json + manifest.json + metrics.json
    (cfg.out_dir / "labels.json").write_text(json.dumps({
        "class_names": classes,
        "input_size": INPUT_SIZE,
        "architecture": "efficientnet_v2_s",
    }, indent=2))
    # metrics.json already written per-epoch above; rewriting is fine
    (cfg.out_dir / "manifest.json").write_text(json.dumps({
        "version": 1,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weights_filename": "weights.0001.pt",
        "weights_onnx_filename": "weights.0001.onnx",
        "labels_filename": "labels.json",
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
        "val_acc": best_val_acc if history else 0.0,
        "best_epoch": best_epoch,
        "architecture": "efficientnet_v2_s",
        "input_size": INPUT_SIZE,
        "balance_to_smallest": cfg.balance_to_smallest,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
    }, indent=2))
    (cfg.out_dir / "version.json").write_text(json.dumps({
        "version": 1,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weights_filename": "weights.0001.pt",
        "weights_onnx_filename": "weights.0001.onnx",
        "n_train_samples": len(train_idx),
        "val_acc": best_val_acc if history else 0.0,
        "best_epoch": best_epoch,
    }, indent=2))
    log.info("wrote outputs to %s", cfg.out_dir)
    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    # Defaults match Jerry's v2 notebook cell 9 recipe exactly:
    #   --architecture efficientnet_v2_s --input-size 384
    #   --epochs 50 --batch-size 16 --lr 1e-4
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--balance-to-smallest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train(args)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
