"""
Folder-structured image dataset for the connector classifier.

Expects a directory layout like:
    data_root/
      SMA-M/
        img_0001.jpg
        ...
      SMA-F/
        ...

Folder names become class labels. Subclassing torchvision.datasets.ImageFolder
gives us the right semantics with no extra code, but we centralize it here
so the train + predict modules agree on transforms, ignore non-image files,
and use a stable class-name → index mapping persisted alongside the weights.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_SIZE = 224


def make_train_transforms() -> transforms.Compose:
    """Aggressive augmentation: forces the model to identify the connector
    rather than relying on background / lighting cues that are constant
    across our training videos but absent from real-world held-out
    photos. Small RandomResizedCrop scales include "tight zoom on the
    connector" cases. Rotation, ColorJitter, blur, and RandomErasing
    all break shortcut cues."""
    return transforms.Compose([
        transforms.Resize(256),
        # Tight scale crops force the model to identify the central
        # connector instead of relying on background.
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        # NO vertical flip: connectors have a top/bottom in our shots and
        # flipping breaks pin-vs-hole appearance.
        transforms.RandomRotation(degrees=20),
        # Mild ColorJitter — anything stronger than ~0.3 brightness erases
        # the M-vs-F cue (bright pin vs dark hollow). Keep saturation
        # modest so material colors stay recognizable.
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Tiny RandomErasing patches: scale up to 5% only — small enough
        # that the central pin/hole gender cue almost always survives,
        # large enough to nudge the model away from background shortcuts.
        # Empirically: omitting it entirely or going larger both hurt
        # held-out across runs.
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.05), ratio=(0.5, 2.0)),
    ])


def make_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ConnectorFolderDataset(Dataset):
    """
    Walks `root/<class_name>/*.{jpg,png,...}` and yields (tensor, class_index)
    pairs. We don't use torchvision.ImageFolder directly because it raises on
    empty subdirs and includes non-image files; this custom class is more
    forgiving for in-progress data folders.
    """

    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[str, int]:
        counts = {n: 0 for n in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts
