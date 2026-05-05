"""
Prototypical Networks — metric-learning approach for low-shot
classification.

Standard supervised fine-tuning treats classification as drawing
boundaries in a 6-way softmax. With ~30-300 unique base images per
class, that's a lot of decision boundary to fit on very little
data. Prototypical Networks instead train the backbone to produce
a 128-d embedding where same-class images cluster tightly and
different-class images don't. At inference, you compute the test
image's embedding and pick the nearest class prototype (mean of
that class's training embeddings) by cosine distance.

This is the paradigm few-shot learning was built on. It typically
needs *less* data per class than supervised CE because it learns
similarity instead of class boundaries.

Training:
  1. Each batch: sample N support images per class + Q query images per class
  2. Compute embeddings for all
  3. Class prototype = mean of N support embeddings
  4. Loss = -log P(query belongs to its true class | distances to all prototypes)
  5. Backprop through the embedding network

Inference (after training):
  1. Compute "production prototypes" once: mean embedding across all training
     images of each class
  2. Test image -> embedding -> nearest prototype by cosine distance

Self-contained: trains, saves, runs the held-out benchmark using
the same rembg-clean inference path the predict_service uses.

Run on the box:
    sudo -u rfcai .venv/bin/python scripts/exp_proto_train.py \\
        --data-dir data/labeled/embedder \\
        --holdout-dir data/test_holdout \\
        --out-dir /home/rfcai/training/models/connector_classifier_proto \\
        --epochs 20 --balance-to-smallest --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfconnectorai.classifier.dataset import (
    INPUT_SIZE, ConnectorFolderDataset,
    make_train_transforms, make_eval_transforms,
)
from rfconnectorai.classifier.train import _grouped_stratified_split


CANONICAL = ["3.5mm-M", "3.5mm-F", "2.92mm-M", "2.92mm-F",
             "2.4mm-M", "2.4mm-F"]


class PrototypicalNet(nn.Module):
    """ResNet-18 backbone + small embedding head. Output is a 128-d
    L2-normalized embedding."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        n_features = backbone.fc.in_features    # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.embed(feats)
        return F.normalize(emb, dim=-1)


class EpisodicSampler:
    """Yields batches as (N+Q)*K episodes — N support + Q query per class
    per episode. Balanced N-way K-shot sampling for ProtoNet training."""

    def __init__(self, samples, n_classes, n_support, n_query,
                 episodes_per_epoch, seed=0):
        self.samples = samples
        self.n_support = n_support
        self.n_query = n_query
        self.episodes = episodes_per_epoch
        self.rng = random.Random(seed)
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for idx, (_p, lbl) in enumerate(samples):
            self.by_class[lbl].append(idx)
        self.classes = sorted(self.by_class.keys())
        if len(self.classes) < n_classes:
            n_classes = len(self.classes)
        self.n_classes = n_classes

    def __iter__(self):
        for _ in range(self.episodes):
            picked_classes = self.rng.sample(self.classes, self.n_classes)
            support_idx = []
            query_idx = []
            for c in picked_classes:
                pool = self.by_class[c]
                if len(pool) < self.n_support + self.n_query:
                    chosen = self.rng.choices(pool,
                                              k=self.n_support + self.n_query)
                else:
                    chosen = self.rng.sample(pool,
                                             self.n_support + self.n_query)
                support_idx.extend(chosen[:self.n_support])
                query_idx.extend(chosen[self.n_support:])
            yield support_idx, query_idx, picked_classes

    def __len__(self):
        return self.episodes


class IndexedSubset(Dataset):
    def __init__(self, parent_ds, indices):
        self.parent = parent_ds
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.parent[self.indices[i]]


def proto_loss_and_acc(model, support_imgs, support_labels,
                       query_imgs, query_labels, picked_classes, device):
    """One episode forward+loss. Returns (loss, n_correct, n_total)."""
    support_emb = model(support_imgs)   # (S, D)
    query_emb = model(query_imgs)       # (Q, D)
    # Compute prototype = mean embedding per class in this episode.
    cls_to_pos = {c: i for i, c in enumerate(picked_classes)}
    prototypes = torch.zeros(len(picked_classes), support_emb.size(1),
                             device=device)
    counts = torch.zeros(len(picked_classes), device=device)
    for emb, lbl in zip(support_emb, support_labels):
        pos = cls_to_pos[int(lbl)]
        prototypes[pos] += emb
        counts[pos] += 1
    prototypes = prototypes / counts.unsqueeze(1).clamp(min=1)
    prototypes = F.normalize(prototypes, dim=-1)
    # Cosine similarity between each query and each prototype, scaled
    # for sharper softmax.
    logits = 10.0 * (query_emb @ prototypes.t())   # (Q, n_classes)
    # Map true class to its position in picked_classes order.
    target = torch.tensor(
        [cls_to_pos[int(l)] for l in query_labels],
        device=device, dtype=torch.long,
    )
    loss = F.cross_entropy(logits, target)
    correct = int((logits.argmax(1) == target).sum())
    return loss, correct, len(target)


def _balance_to_smallest(samples, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for idx, (_p, l) in enumerate(samples):
        by_class[l].append(idx)
    target = min(len(v) for v in by_class.values())
    keep = []
    for k, idxs in by_class.items():
        order = rng.permutation(len(idxs))[:target]
        keep.extend(idxs[i] for i in order)
    keep.sort()
    return keep, target


def train_proto(args, classes_kept):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[proto] device={device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ConnectorFolderDataset(
        root=args.data_dir, class_names=classes_kept,
        transform=make_train_transforms(),
    )
    eval_ds = ConnectorFolderDataset(
        root=args.data_dir, class_names=classes_kept,
        transform=make_eval_transforms(),
    )
    print(f"[proto] total samples: {len(train_ds)}")
    if args.balance_to_smallest:
        keep, target = _balance_to_smallest(train_ds.samples, args.seed)
        train_ds.samples = [train_ds.samples[i] for i in keep]
        eval_ds.samples = [eval_ds.samples[i] for i in keep]
        print(f"[proto] balanced -> {target}/class, total {len(train_ds)}")

    train_idx, val_idx = _grouped_stratified_split(
        [(str(p), l) for p, l in train_ds.samples], args.val_fraction, args.seed,
    )
    print(f"[proto] split: train={len(train_idx)} val={len(val_idx)}")

    train_subset_samples = [train_ds.samples[i] for i in train_idx]
    val_subset_samples = [eval_ds.samples[i] for i in val_idx]

    n_classes_per_episode = min(len(classes_kept), args.way)
    train_sampler = EpisodicSampler(
        train_subset_samples, n_classes=n_classes_per_episode,
        n_support=args.n_support, n_query=args.n_query,
        episodes_per_epoch=args.episodes_per_epoch, seed=args.seed,
    )
    val_sampler = EpisodicSampler(
        val_subset_samples, n_classes=n_classes_per_episode,
        n_support=args.n_support, n_query=args.n_query,
        episodes_per_epoch=max(20, args.episodes_per_epoch // 4),
        seed=args.seed + 7,
    )

    model = PrototypicalNet(embed_dim=args.embed_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for support_idx, query_idx, picked in train_sampler:
            s_imgs = torch.stack([train_subset_samples[i][0] is not None
                                  and torch.tensor(0) or torch.tensor(0)
                                  for i in support_idx]).to(device)
            # Fetch via dataset to apply transforms.
            sup_items = [train_ds[train_idx[i]] for i in support_idx]
            que_items = [train_ds[train_idx[i]] for i in query_idx]
            s_imgs = torch.stack([x[0] for x in sup_items]).to(device)
            s_lbls = torch.tensor([x[1] for x in sup_items], device=device)
            q_imgs = torch.stack([x[0] for x in que_items]).to(device)
            q_lbls = torch.tensor([x[1] for x in que_items], device=device)
            loss, correct, total = proto_loss_and_acc(
                model, s_imgs, s_lbls, q_imgs, q_lbls, picked, device,
            )
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += float(loss.detach()) * total
            t_correct += correct
            t_total += total
        sched.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for support_idx, query_idx, picked in val_sampler:
                sup_items = [eval_ds[val_idx[i]] for i in support_idx]
                que_items = [eval_ds[val_idx[i]] for i in query_idx]
                s_imgs = torch.stack([x[0] for x in sup_items]).to(device)
                s_lbls = torch.tensor([x[1] for x in sup_items], device=device)
                q_imgs = torch.stack([x[0] for x in que_items]).to(device)
                q_lbls = torch.tensor([x[1] for x in que_items], device=device)
                loss, correct, total = proto_loss_and_acc(
                    model, s_imgs, s_lbls, q_imgs, q_lbls, picked, device,
                )
                v_loss += float(loss.detach()) * total
                v_correct += correct
                v_total += total

        line = (f"epoch {epoch:>2}/{args.epochs}  "
                f"lr={opt.param_groups[0]['lr']:.5f}  "
                f"train_loss={t_loss/max(1,t_total):.3f} "
                f"train_acc={t_correct/max(1,t_total):.3f}  "
                f"val_loss={v_loss/max(1,v_total):.3f} "
                f"val_acc={v_correct/max(1,v_total):.3f}")
        print(line, flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": t_loss/max(1,t_total),
            "train_acc": t_correct/max(1,t_total),
            "val_loss": v_loss/max(1,v_total),
            "val_acc": v_correct/max(1,v_total),
        })

    torch.save(model.state_dict(), args.out_dir / "weights_proto.pt")
    (args.out_dir / "labels_proto.json").write_text(json.dumps({
        "class_names": classes_kept,
        "input_size": INPUT_SIZE,
        "architecture": "prototypical_resnet18",
        "embed_dim": args.embed_dim,
    }, indent=2))
    (args.out_dir / "metrics_proto.json").write_text(
        json.dumps({"history": history}, indent=2))

    # Compute production prototypes: mean embedding of ALL train+val
    # (after split) images per class, using eval transforms (no aug).
    print("[proto] computing production prototypes...")
    model.eval()
    prototypes = {}
    with torch.no_grad():
        for cls_idx, cls_name in enumerate(classes_kept):
            cls_indices = [i for i in train_idx + val_idx
                           if eval_ds.samples[i][1] == cls_idx]
            if not cls_indices:
                continue
            embs = []
            # Batch embed
            for batch_start in range(0, len(cls_indices), 32):
                batch = cls_indices[batch_start:batch_start + 32]
                items = [eval_ds[i] for i in batch]
                imgs = torch.stack([x[0] for x in items]).to(device)
                emb = model(imgs).cpu().numpy()
                embs.append(emb)
            embs = np.concatenate(embs, axis=0)
            proto = embs.mean(axis=0)
            proto /= np.linalg.norm(proto) + 1e-8
            prototypes[cls_name] = proto.tolist()
    (args.out_dir / "prototypes.json").write_text(
        json.dumps(prototypes, indent=2))
    print(f"[proto] saved prototypes for {len(prototypes)} classes")
    return classes_kept


def benchmark_holdout(model, holdout_dir, classes_kept, prototypes, device):
    import cv2
    from rembg import new_session, remove
    from rfconnectorai.data_fetch.connector_crops import detect_connector_crops
    sess = new_session()
    transform = make_eval_transforms()

    # Stack prototypes into a tensor for matrix similarity.
    proto_names = list(prototypes.keys())
    proto_mat = torch.tensor(
        np.stack([np.array(prototypes[n]) for n in proto_names], axis=0),
        device=device, dtype=torch.float32,
    )

    def _tta_variants(pil):
        w, h = pil.size
        crop = int(min(w, h) * 0.9)
        left = (w - crop) // 2; top = (h - crop) // 2
        return [pil,
                pil.transpose(Image.FLIP_LEFT_RIGHT),
                pil.rotate(10, resample=Image.BILINEAR),
                pil.rotate(-10, resample=Image.BILINEAR),
                pil.crop((left, top, left+crop, top+crop))]

    full_ok = fam_ok = gen_ok = n = 0
    print()
    print("=== held-out (rembg-clean inference, prototypical) ===")
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
                    embs = model(xs)
                    avg = F.normalize(embs.mean(0, keepdim=True), dim=-1)
                    sims = (avg @ proto_mat.t()).squeeze(0).cpu().numpy()
                top = int(sims.argmax())
                pred = proto_names[top]
                conf = float(sims[top])
                if best is None or conf > best[1]:
                    best = (pred, conf)
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
            print(f"  [{mark}] truth={truth:<10}: pred={pred:<10} sim={conf:.3f}  ({img_path.name})")

    if n == 0: return
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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-to-smallest", action="store_true")
    ap.add_argument("--way", type=int, default=6,
                    help="N-way per episode")
    ap.add_argument("--n-support", type=int, default=4,
                    help="N support per class per episode")
    ap.add_argument("--n-query", type=int, default=8,
                    help="N query per class per episode")
    ap.add_argument("--episodes-per-epoch", type=int, default=200)
    ap.add_argument("--embed-dim", type=int, default=128)
    args = ap.parse_args()

    classes_kept = []
    for c in CANONICAL:
        d = args.data_dir / c
        n = sum(1 for p in d.glob("*.[jJpP][pPnN]*[gG]")) if d.is_dir() else 0
        if n >= 5:
            classes_kept.append(c)
    print(f"[proto] classes kept: {classes_kept}")

    train_proto(args, classes_kept)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrototypicalNet(embed_dim=args.embed_dim).to(device).eval()
    model.load_state_dict(torch.load(
        args.out_dir / "weights_proto.pt", map_location=device))
    prototypes = json.loads(
        (args.out_dir / "prototypes.json").read_text())
    benchmark_holdout(model, args.holdout_dir, classes_kept, prototypes, device)


if __name__ == "__main__":
    main()
