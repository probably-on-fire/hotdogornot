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

# Bumped 224 -> 384 (2026-05-04). Phone-shot connector crops at full
# resolution often have the connector face at 200-400px diameter; the
# old 224 was throwing away most of that detail. ResNet-18's conv
# layers are size-agnostic (only the final FC depends), so raising
# INPUT_SIZE costs ~3x compute per forward pass but doesn't require
# arch changes. Expected to help Full/Family discrimination since the
# differentiating features (bore_id, dielectric appearance, threading)
# are a few pixels at 224 but tens of pixels at 384.
INPUT_SIZE = 384
RESIZE_BEFORE_CROP = int(INPUT_SIZE * 1.14)   # was 256 = 224 * 1.14


def make_train_transforms() -> transforms.Compose:
    """Aggressive augmentation: forces the model to identify the connector
    rather than relying on background / lighting cues that are constant
    across our training videos but absent from real-world held-out
    photos. Small RandomResizedCrop scales include "tight zoom on the
    connector" cases. Rotation, ColorJitter, blur, and RandomErasing
    all break shortcut cues.

    2026-05-04 update: bumped ColorJitter brightness 0.25->0.35 and
    contrast 0.25->0.35 (held-out phone shots vary much more in
    lighting than the training videos), and Gaussian blur p 0.2->0.4
    (held-out shots are sometimes slightly out of focus). Risk to the
    M-vs-F cue is minor at these ranges since the central pin/hole
    luminance contrast is much larger than 0.35 brightness span."""
    return transforms.Compose([
        transforms.Resize(RESIZE_BEFORE_CROP),
        # Tight scale crops force the model to identify the central
        # connector instead of relying on background.
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        # NO vertical flip: connectors have a top/bottom in our shots and
        # flipping breaks pin-vs-hole appearance.
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.4),
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
        transforms.Resize(RESIZE_BEFORE_CROP),
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
