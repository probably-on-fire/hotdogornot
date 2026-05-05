"""
Architecture variants:
  --variant mlp-head   : ResNet-18 + deeper classifier head
                         (512 -> 256 -> 64 -> 6 with Dropout + ReLU)
  --variant resnet50   : ResNet-50 backbone, single linear head
                         (2048 -> 6)

Both reuse the rest of the training pipeline (data, augmentation,
dHash-grouped split, balance-to-smallest, WRS, label smoothing,
cosine LR). Self-contained: trains, saves to sibling model dir,
runs the held-out benchmark in-script using the same rembg-clean
inference path the predict_service uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfconnectorai.classifier.dataset import (
    INPUT_SIZE, ConnectorFolderDataset,
    make_train_transforms, make_eval_transforms,
)
from rfconnectorai.classifier.train import _grouped_stratified_split


CANONICAL = ["3.5mm-M", "3.5mm-F", "2.92mm-M", "2.92mm-F",
             "2.4mm-M", "2.4mm-F"]


def build_model(variant: str, num_classes: int) -> nn.Module:
    if variant == "mlp-head":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = m.fc.in_features    # 512
        m.fc = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        return m
    if variant == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown variant: {variant}")


def _balance_to_smallest(samples, seed):
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx, (_p, label) in enumerate(samples):
        by_class.setdefault(label, []).append(idx)
    target = min(len(v) for v in by_class.values())
    keep = []
    for k, idxs in by_class.items():
        order = rng.permutation(len(idxs))[:target]
        keep.extend(idxs[i] for i in order)
    keep.sort()
    return keep, target


def _make_sampler(train_indices, samples, max_oversample_ratio=10.0):
    train_labels = [samples[i][1] for i in train_indices]
    counts = Counter(train_labels)
    majority = max(counts.values())
    max_w = max_oversample_ratio / float(majority)
    weights = [min(1.0/counts[l], max_w) for l in train_labels]
    return WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)


def train_run(args):
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.variant}] device={device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Drop classes with no data (parity with auto_retrain).
    classes_kept = []
    for c in CANONICAL:
        d = args.data_dir / c
        n = sum(1 for p in d.glob("*.[jJpP][pPnN]*[gG]")) if d.is_dir() else 0
        if n >= 5:
            classes_kept.append(c)
    print(f"[{args.variant}] classes kept: {classes_kept}")

    train_ds = ConnectorFolderDataset(
        root=args.data_dir, class_names=classes_kept,
        transform=make_train_transforms(),
    )
    eval_ds = ConnectorFolderDataset(
        root=args.data_dir, class_names=classes_kept,
        transform=make_eval_transforms(),
    )
    print(f"[{args.variant}] total samples: {len(train_ds)}")
    if args.balance_to_smallest:
        keep, target = _balance_to_smallest(train_ds.samples, args.seed)
        train_ds.samples = [train_ds.samples[i] for i in keep]
        eval_ds.samples = [eval_ds.samples[i] for i in keep]
        print(f"[{args.variant}] balanced -> {target}/class, total {len(train_ds)}")

    train_idx, val_idx = _grouped_stratified_split(
        [(str(p), l) for p, l in train_ds.samples], args.val_fraction, args.seed,
    )
    print(f"[{args.variant}] split: train={len(train_idx)} val={len(val_idx)}")

    sampler = _make_sampler(train_idx, train_ds.samples)
    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Subset(eval_ds, val_idx),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.variant, len(classes_kept)).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_corr, t_n = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = crit(logits, lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += float(loss.detach()) * imgs.size(0)
            t_corr += int((logits.argmax(1) == lbls).sum())
            t_n += imgs.size(0)
        sched.step()

        model.eval()
        v_loss, v_corr, v_n = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = crit(logits, lbls)
                v_loss += float(loss.detach()) * imgs.size(0)
                v_corr += int((logits.argmax(1) == lbls).sum())
                v_n += imgs.size(0)
        line = (f"epoch {epoch:>2}/{args.epochs}  "
                f"lr={opt.param_groups[0]['lr']:.5f}  "
                f"train_loss={t_loss/max(1,t_n):.3f} train_acc={t_corr/max(1,t_n):.3f}  "
                f"val_loss={v_loss/max(1,v_n):.3f} val_acc={v_corr/max(1,v_n):.3f}")
        print(line, flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": t_loss/max(1,t_n),
            "train_acc": t_corr/max(1,t_n),
            "val_loss": v_loss/max(1,v_n),
            "val_acc": v_corr/max(1,v_n),
        })

    weights_name = f"weights_{args.variant.replace('-','_')}.pt"
    labels_name = f"labels_{args.variant.replace('-','_')}.json"
    torch.save(model.state_dict(), args.out_dir / weights_name)
    (args.out_dir / labels_name).write_text(json.dumps({
        "class_names": classes_kept,
        "input_size": INPUT_SIZE,
        "architecture": args.variant,
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
    }, indent=2))
    return classes_kept


def _tta_variants(pil):
    w, h = pil.size
    crop = int(min(w, h) * 0.9)
    left = (w - crop) // 2; top = (h - crop) // 2
    return [pil,
            pil.transpose(Image.FLIP_LEFT_RIGHT),
            pil.rotate(10, resample=Image.BILINEAR),
            pil.rotate(-10, resample=Image.BILINEAR),
            pil.crop((left, top, left+crop, top+crop))]


def benchmark_holdout(model, holdout_dir, classes_kept, device):
    import cv2
    from rembg import new_session, remove
    from rfconnectorai.data_fetch.connector_crops import detect_connector_crops
    sess = new_session()
    transform = make_eval_transforms()

    full_ok = fam_ok = gen_ok = n = 0
    print()
    print("=== held-out (rembg-clean inference) ===")
    for cls_dir in sorted(p for p in holdout_dir.iterdir() if p.is_dir()):
        truth = cls_dir.name
        try: t_fam, t_gen = truth.rsplit("-", 1)
        except ValueError: continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            crops = detect_connector_crops(bgr, max_crops=4)
            best = None
            for c in crops:
                rgb = cv2.cvtColor(c.crop, cv2.COLOR_BGR2RGB)
                rgba = remove(rgb, session=sess)
                if rgba.ndim != 3 or rgba.shape[2] != 4: continue
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                white = np.full_like(rgba[:, :, :3], 255, dtype=np.float32)
                rgb_pixels = rgba[:, :, :3].astype(np.float32)
                comp = (rgb_pixels * alpha + white * (1.0 - alpha)).astype(np.uint8)
                pil = Image.fromarray(comp)
                xs = torch.stack([transform(v) for v in _tta_variants(pil)], 0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(xs), 1).mean(0).cpu().numpy()
                top = int(probs.argmax())
                if best is None or probs[top] > best[1]:
                    best = (classes_kept[top], float(probs[top]))
            if best is None:
                print(f"  truth={truth:<10}: NO_DETECT  ({img_path.name})")
                n += 1; continue
            pred, conf = best
            try: p_fam, p_gen = pred.rsplit("-", 1)
            except ValueError: p_fam, p_gen = pred, "?"
            full = pred == truth
            full_ok += int(full); fam_ok += int(p_fam == t_fam); gen_ok += int(p_gen == t_gen)
            n += 1
            mark = "✓" if full else " "
            print(f"  [{mark}] truth={truth:<10}: pred={pred:<10} conf={conf:.2f}  ({img_path.name})")

    if n == 0: return
    print()
    print(f"HOLDOUT  Full: {full_ok}/{n} ({100*full_ok/n:.1f}%)  "
          f"Family: {fam_ok}/{n} ({100*fam_ok/n:.1f}%)  "
          f"Gender: {gen_ok}/{n} ({100*gen_ok/n:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["mlp-head", "resnet50"])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--holdout-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-to-smallest", action="store_true")
    args = ap.parse_args()

    classes_kept = train_run(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.variant, len(classes_kept)).to(device).eval()
    weights_name = f"weights_{args.variant.replace('-','_')}.pt"
    model.load_state_dict(torch.load(args.out_dir / weights_name, map_location=device))
    benchmark_holdout(model, args.holdout_dir, classes_kept, device)


if __name__ == "__main__":
    main()
