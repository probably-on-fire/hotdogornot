"""
Two-head ResNet-18 — shared backbone, separate family + gender heads.

Hypothesis: a single 6-way softmax forces the model to entangle the
family and gender decisions, which lets it shortcut on either signal
to "explain" the other. A factorized two-head model trains the
backbone to produce features that *both* heads can read independently,
which should yield better discrimination on the family axis (where
v18's misses live).

This is different from `_exp_gender_v2.py` / the "two-classifier"
approach in the journey doc — that uses two SEPARATE backbones. Here
the backbone is shared; only the final linear layer is split.

Outputs to `--out-dir`:
    weights_two_head.pt    — full state_dict (backbone + both heads)
    labels_two_head.json   — family list, gender list, INPUT_SIZE
    metrics_two_head.json  — train/val per epoch + final held-out

Usage on the box:
    sudo -u rfcai .venv/bin/python scripts/exp_two_head_train.py \\
        --data-dir data/labeled/embedder \\
        --holdout-dir data/test_holdout \\
        --out-dir /home/rfcai/training/models/connector_classifier_2h \\
        --epochs 20 --balance-to-smallest --seed 0
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
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfconnectorai.classifier.dataset import (
    INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    make_train_transforms, make_eval_transforms,
)
from rfconnectorai.classifier.train import _grouped_stratified_split


FAMILIES = ["3.5mm", "2.92mm", "2.4mm", "SMA"]
GENDERS = ["M", "F"]
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}
GENDER_TO_IDX = {g: i for i, g in enumerate(GENDERS)}


def parse_class(class_name: str) -> tuple[int, int]:
    family, gender = class_name.rsplit("-", 1)
    return FAMILY_TO_IDX[family], GENDER_TO_IDX[gender]


class TwoHeadDataset(Dataset):
    """Folder-structured dataset; yields (image, family_idx, gender_idx)."""
    VALID = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: Path, class_names: list[str], transform=None,
                 min_samples_per_class: int = 5):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []
        self.classes_kept = []
        for cls in class_names:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            try:
                fam_i, gen_i = parse_class(cls)
            except (ValueError, KeyError):
                continue
            files = [p for p in cls_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in self.VALID]
            if len(files) < min_samples_per_class:
                continue
            self.classes_kept.append(cls)
            for p in files:
                self.samples.append((p, fam_i, gen_i))
        if not self.samples:
            raise RuntimeError(f"no usable training images under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fam, gen = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fam, gen

    # Compat shim for _grouped_stratified_split which expects samples
    # to be (path, label) pairs. We use the combined class-index as
    # the grouping label so the dHash split still groups within-class.
    @property
    def split_samples(self):
        return [(str(p), fam * len(GENDERS) + gen)
                for p, fam, gen in self.samples]


class TwoHeadResNet18(nn.Module):
    def __init__(self, n_families: int, n_genders: int):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.family_head = nn.Linear(n_features, n_families)
        self.gender_head = nn.Linear(n_features, n_genders)

    def forward(self, x):
        feats = self.backbone(x)
        return self.family_head(feats), self.gender_head(feats)


def _set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _make_sampler(train_indices, samples, max_oversample_ratio: float = 10.0):
    """Per-(family,gender) balanced sampling with cap."""
    train_keys = [(samples[i][1], samples[i][2]) for i in train_indices]
    counts = Counter(train_keys)
    majority = max(counts.values())
    max_w = max_oversample_ratio / float(majority)
    weights = [min(1.0 / counts[k], max_w) for k in train_keys]
    return WeightedRandomSampler(weights, num_samples=len(train_indices),
                                 replacement=True)


def _balance_to_smallest(samples, seed):
    """Subsample each (family,gender) class to the smallest count."""
    rng = np.random.default_rng(seed)
    by_class: dict[tuple[int, int], list[int]] = {}
    for idx, (_p, fam, gen) in enumerate(samples):
        by_class.setdefault((fam, gen), []).append(idx)
    target = min(len(v) for v in by_class.values())
    keep_indices: list[int] = []
    for k, idx_list in by_class.items():
        order = rng.permutation(len(idx_list))[:target]
        keep_indices.extend(idx_list[i] for i in order)
    keep_indices.sort()
    return keep_indices, target


def train_two_head(config) -> dict:
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[two_head] device={device}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TwoHeadDataset(
        root=config.data_dir,
        class_names=[f"{f}-{g}" for f in FAMILIES for g in GENDERS],
        transform=make_train_transforms(),
    )
    eval_ds = TwoHeadDataset(
        root=config.data_dir,
        class_names=[f"{f}-{g}" for f in FAMILIES for g in GENDERS],
        transform=make_eval_transforms(),
    )
    print(f"[two_head] classes kept: {train_ds.classes_kept}")
    print(f"[two_head] total samples: {len(train_ds)}")

    if config.balance_to_smallest:
        keep, target = _balance_to_smallest(train_ds.samples, config.seed)
        train_ds.samples = [train_ds.samples[i] for i in keep]
        eval_ds.samples = [eval_ds.samples[i] for i in keep]
        print(f"[two_head] balanced to {target}/class -> {len(train_ds)} samples")

    train_idx, val_idx = _grouped_stratified_split(
        train_ds.split_samples, config.val_fraction, config.seed,
    )
    print(f"[two_head] split: train={len(train_idx)} val={len(val_idx)}")

    sampler = _make_sampler(train_idx, train_ds.samples)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=config.batch_size, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=config.batch_size, shuffle=False, num_workers=0,
    )

    model = TwoHeadResNet18(len(FAMILIES), len(GENDERS)).to(device)
    criterion_fam = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs)

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        t_loss, t_fcorr, t_gcorr, t_n = 0.0, 0, 0, 0
        for imgs, fam_t, gen_t in train_loader:
            imgs = imgs.to(device); fam_t = fam_t.to(device); gen_t = gen_t.to(device)
            fam_logits, gen_logits = model(imgs)
            loss = criterion_fam(fam_logits, fam_t) + criterion_gen(gen_logits, gen_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += float(loss.detach()) * imgs.size(0)
            t_fcorr += int((fam_logits.argmax(1) == fam_t).sum())
            t_gcorr += int((gen_logits.argmax(1) == gen_t).sum())
            t_n += imgs.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_fcorr, v_gcorr, v_n = 0.0, 0, 0, 0
        with torch.no_grad():
            for imgs, fam_t, gen_t in val_loader:
                imgs = imgs.to(device); fam_t = fam_t.to(device); gen_t = gen_t.to(device)
                fam_logits, gen_logits = model(imgs)
                loss = criterion_fam(fam_logits, fam_t) + criterion_gen(gen_logits, gen_t)
                v_loss += float(loss.detach()) * imgs.size(0)
                v_fcorr += int((fam_logits.argmax(1) == fam_t).sum())
                v_gcorr += int((gen_logits.argmax(1) == gen_t).sum())
                v_n += imgs.size(0)
        line = (f"epoch {epoch:>2}/{config.epochs}  "
                f"lr={optimizer.param_groups[0]['lr']:.5f}  "
                f"train_loss={t_loss/max(1,t_n):.3f} "
                f"train_fam={t_fcorr/max(1,t_n):.3f} "
                f"train_gen={t_gcorr/max(1,t_n):.3f}  "
                f"val_loss={v_loss/max(1,v_n):.3f} "
                f"val_fam={v_fcorr/max(1,v_n):.3f} "
                f"val_gen={v_gcorr/max(1,v_n):.3f}")
        print(line)
        history.append({
            "epoch": epoch,
            "train_loss": t_loss/max(1,t_n),
            "train_fam_acc": t_fcorr/max(1,t_n),
            "train_gen_acc": t_gcorr/max(1,t_n),
            "val_loss": v_loss/max(1,v_n),
            "val_fam_acc": v_fcorr/max(1,v_n),
            "val_gen_acc": v_gcorr/max(1,v_n),
        })

    torch.save(model.state_dict(), config.out_dir / "weights_two_head.pt")
    (config.out_dir / "labels_two_head.json").write_text(json.dumps({
        "families": FAMILIES,
        "genders": GENDERS,
        "input_size": INPUT_SIZE,
        "architecture": "two_head_resnet18",
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
    }, indent=2))
    (config.out_dir / "metrics_two_head.json").write_text(
        json.dumps({"history": history}, indent=2))
    return {"history": history}


def _tta_variants(pil):
    w, h = pil.size
    crop = int(min(w, h) * 0.9)
    left = (w - crop) // 2; top = (h - crop) // 2
    return [
        pil,
        pil.transpose(Image.FLIP_LEFT_RIGHT),
        pil.rotate(10, resample=Image.BILINEAR),
        pil.rotate(-10, resample=Image.BILINEAR),
        pil.crop((left, top, left + crop, top + crop)),
    ]


def predict_two_head(model, image, device):
    """RGB ndarray or PIL -> (family_str, gender_str, combined_class, conf)."""
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image.convert("RGB")
    transform = make_eval_transforms()
    variants = _tta_variants(pil)
    xs = torch.stack([transform(v) for v in variants], dim=0).to(device)
    with torch.no_grad():
        fam_logits, gen_logits = model(xs)
        fam_probs = torch.softmax(fam_logits, dim=1).mean(dim=0).cpu().numpy()
        gen_probs = torch.softmax(gen_logits, dim=1).mean(dim=0).cpu().numpy()
    fi = int(fam_probs.argmax()); gi = int(gen_probs.argmax())
    return FAMILIES[fi], GENDERS[gi], f"{FAMILIES[fi]}-{GENDERS[gi]}", \
           float(fam_probs[fi] * gen_probs[gi])


def benchmark_holdout(model, holdout_dir: Path, device):
    """Eval against the test_holdout set, mirroring the production
    predict path: rembg → cleaned-on-white → classify."""
    import cv2
    from rembg import new_session, remove
    from rfconnectorai.data_fetch.connector_crops import detect_connector_crops
    sess = new_session()

    full_ok = fam_ok = gen_ok = 0
    n = 0
    print()
    print("=== held-out (rembg-clean inference) ===")
    for cls_dir in sorted(p for p in holdout_dir.iterdir() if p.is_dir()):
        truth = cls_dir.name
        try:
            t_fam, t_gen = truth.rsplit("-", 1)
        except ValueError:
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            crops = detect_connector_crops(bgr, max_crops=4)
            best = None
            for c in crops:
                rgb = cv2.cvtColor(c.crop, cv2.COLOR_BGR2RGB)
                rgba = remove(rgb, session=sess)
                if rgba.ndim != 3 or rgba.shape[2] != 4:
                    continue
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                white = np.full_like(rgba[:, :, :3], 255, dtype=np.float32)
                rgb_pixels = rgba[:, :, :3].astype(np.float32)
                comp = (rgb_pixels * alpha + white * (1.0 - alpha)).astype(np.uint8)
                fam, gen, combined, conf = predict_two_head(model, comp, device)
                if best is None or conf > best[3]:
                    best = (fam, gen, combined, conf)
            if best is None:
                print(f"  truth={truth:<10}: NO_DETECT  ({img_path.name})")
                n += 1
                continue
            fam, gen, combined, conf = best
            full = combined == truth
            fam_match = fam == t_fam
            gen_match = gen == t_gen
            full_ok += int(full); fam_ok += int(fam_match); gen_ok += int(gen_match)
            n += 1
            mark = "✓" if full else " "
            print(f"  [{mark}] truth={truth:<10}: pred={combined:<10} "
                  f"conf={conf:.2f}  ({img_path.name})")
    if n == 0:
        print("no held-out images found")
        return
    print()
    print(f"HOLDOUT  Full: {full_ok}/{n} ({100*full_ok/n:.1f}%)  "
          f"Family: {fam_ok}/{n} ({100*fam_ok/n:.1f}%)  "
          f"Gender: {gen_ok}/{n} ({100*gen_ok/n:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
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

    class C: pass
    cfg = C()
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.val_fraction = args.val_fraction
    cfg.seed = args.seed
    cfg.balance_to_smallest = args.balance_to_smallest

    train_two_head(cfg)

    # Reload weights and benchmark on held-out.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadResNet18(len(FAMILIES), len(GENDERS)).to(device).eval()
    model.load_state_dict(torch.load(
        cfg.out_dir / "weights_two_head.pt", map_location=device))
    benchmark_holdout(model, args.holdout_dir, device)


if __name__ == "__main__":
    main()
