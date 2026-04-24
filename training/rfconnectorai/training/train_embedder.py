from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder, recommended_image_size
from rfconnectorai.training.arcface_loss import ArcFaceLoss
from rfconnectorai.training.triplet_loss import batch_hard_triplet_loss


class PKSampler(Sampler[list[int]]):
    """Yield batches containing P classes × K samples each (balanced for triplet mining)."""

    def __init__(self, labels: list[int], classes_per_batch: int, samples_per_class: int):
        self.labels = labels
        self.P = classes_per_batch
        self.K = samples_per_class
        self._by_class: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            self._by_class.setdefault(lab, []).append(idx)
        self._valid_classes = [c for c, idxs in self._by_class.items() if len(idxs) >= self.K]
        if len(self._valid_classes) < self.P:
            # Fall back to however many classes are available with enough samples.
            self.P = max(2, len(self._valid_classes))

    def __iter__(self):
        g = torch.Generator()
        classes = self._valid_classes.copy()
        idx = torch.randperm(len(classes), generator=g).tolist()
        picked = [classes[i] for i in idx[: self.P]]
        batch: list[int] = []
        for c in picked:
            pool = self._by_class[c]
            chosen = torch.randperm(len(pool), generator=g)[: self.K].tolist()
            batch.extend(pool[i] for i in chosen)
        yield batch

    def __len__(self) -> int:
        return 1


def train(
    data_root: Path,
    classes_yaml: Path,
    output_dir: Path,
    image_size: int,
    num_epochs: int,
    learning_rate: float,
    margin: float,
    classes_per_batch: int,
    samples_per_class: int,
    device: str,
    backbone: str,
    pretrained: bool,
    loss_name: str,
    augment: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = RGBDConnectorDataset(
        root=data_root,
        classes_yaml=classes_yaml,
        image_size=image_size,
        augment=augment,
    )
    labels = [sample[1] for sample in ds.samples]

    sampler = PKSampler(labels, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    model = RGBDEmbedder(embedding_dim=128, pretrained=pretrained, backbone=backbone).to(device)

    arcface = None
    if loss_name == "arcface":
        num_classes = len(load_classes(classes_yaml))
        arcface = ArcFaceLoss(embedding_dim=128, num_classes=num_classes, margin=0.5, scale=30.0).to(device)
        params = list(model.parameters()) + list(arcface.parameters())
    else:
        params = list(model.parameters())

    optim = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

    steps_per_epoch = max(1, len(ds) // (classes_per_batch * samples_per_class))

    for epoch in range(num_epochs):
        model.train()
        if arcface is not None:
            arcface.train()
        last_loss = float("nan")
        for _ in range(steps_per_epoch):
            batch = next(iter(loader))
            x, y = batch
            x, y = x.to(device), y.to(device)

            emb = model(x)
            if arcface is not None:
                loss = arcface(emb, y)
            else:
                loss = batch_hard_triplet_loss(emb, y, margin=margin)

            optim.zero_grad()
            loss.backward()
            optim.step()
            last_loss = loss.item()

        print(f"epoch {epoch + 1}/{num_epochs}  loss={last_loss:.4f}")

    ckpt = output_dir / "embedder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": 128,
            "backbone": backbone,
            "image_size": image_size,
            "loss": loss_name,
            "augment": augment,
        },
        ckpt,
    )
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--backbone",
        type=str,
        default="mobilevitv2_100",
        help="timm model name, e.g. mobilevitv2_100 or vit_small_patch14_dinov2.lvd142m",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution. If unset, uses recommended size for the backbone.",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--classes-per-batch", type=int, default=4)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--loss",
        type=str,
        default="arcface",
        choices=["arcface", "triplet"],
        help="Metric-learning loss. ArcFace generally outperforms triplet for fine-grained tasks.",
    )
    ap.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable training-time augmentation (rotation, flip, color jitter, blur). On by default.",
    )
    ap.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable augmentation (e.g. for ablation runs).",
    )
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading pretrained weights (forces random init; CI use only)",
    )
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 epochs at image_size=64 on tiny data. For CI only.",
    )
    args = ap.parse_args()

    if args.smoke_test:
        args.image_size = 64
        args.epochs = 2
        args.classes_per_batch = 2
        args.samples_per_class = 3

    if args.image_size is None:
        args.image_size = recommended_image_size(args.backbone)

    ckpt = train(
        data_root=args.data_root,
        classes_yaml=args.classes_yaml,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        margin=args.margin,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        device=args.device,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        loss_name=args.loss,
        augment=args.augment,
    )
    print(f"checkpoint written to {ckpt}")


if __name__ == "__main__":
    main()
