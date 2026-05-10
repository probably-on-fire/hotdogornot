"""Image dataset for the multi-head classifier.

Reads a row-per-instance ``attributes.csv`` produced by
``rfconnectorai.data.build_yolo_dataset`` and yields
``(image_tensor, target_dict, instance_id)`` tuples. Missing labels are
encoded as ``MISSING_INDEX`` (-1) and masked out of the loss in
:mod:`rfconnectorai.classifier.train_multihead`.

The dataset locates each image by joining ``base-dir`` with the manifest
``source_image`` field (the same convention used by build_yolo_dataset).
This keeps the train pipeline aligned with the manifest pipeline so a
re-run lands on the exact same image set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rfconnectorai.classifier.label_encoding import (
    HEAD_ORDER,
    HeadVocab,
    MISSING_INDEX,
    encode_row,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_train_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiHeadAttributeDataset(Dataset):
    """Read ``attributes.csv`` rows for one split and emit per-head targets.

    Args:
      rows: list of attributes.csv dicts already filtered to the requested split.
      image_root: base directory used to resolve ``source_image`` paths
        (typically the ``--base-dir`` passed to build_yolo_dataset; the
        attributes.csv stores ``source_image`` relative to this root).
      vocabs: mapping of head -> HeadVocab.
      transform: torchvision transform applied to each image tensor.
    """

    def __init__(
        self,
        *,
        rows: list[dict],
        image_root: Path,
        vocabs: Mapping[str, HeadVocab],
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.vocabs = vocabs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def _open_image(self, source_image: str) -> Image.Image:
        path = Path(source_image)
        if not path.is_absolute():
            path = self.image_root / source_image
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        row = self.rows[idx]
        image = self._open_image(row["source_image"])
        tensor = self.transform(image)
        encoded = encode_row(row, self.vocabs)
        targets = {
            head: torch.tensor(encoded.targets.get(head, MISSING_INDEX), dtype=torch.long)
            for head in HEAD_ORDER
        }
        return tensor, targets, str(row.get("instance_id", ""))


def collate_multihead(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], str]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    images = torch.stack([item[0] for item in batch])
    instance_ids = [item[2] for item in batch]
    targets: dict[str, torch.Tensor] = {}
    for head in HEAD_ORDER:
        targets[head] = torch.stack([item[1][head] for item in batch])
    return images, targets, instance_ids
