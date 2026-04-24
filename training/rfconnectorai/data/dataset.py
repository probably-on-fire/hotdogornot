from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.depth_utils import synthesize_depth_from_mask


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBDConnectorDataset(Dataset):
    """
    Directory-per-class dataset that yields 4-channel RGBD tensors.

    Directory layout:
        root/
            SMA-M/*.png
            SMA-F/*.png
            ...

    For each image:
      - Load RGB, resize to (image_size, image_size)
      - Normalize RGB with ImageNet statistics
      - Build a foreground mask (simple: assume center half of image is the connector)
      - Synthesize a depth channel using depth_utils.synthesize_depth_from_mask
      - Stack into a 4xHxW tensor
    """

    def __init__(
        self,
        root: Path | str,
        classes_yaml: Path | str,
        image_size: int = 384,
        object_depth_m: float = 0.12,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.object_depth_m = object_depth_m
        self.augment = augment
        self.classes = load_classes(classes_yaml)
        self._name_to_id = {c.name: c.id for c in self.classes}

        # Accept jpg/jpeg/png/webp from the auto-fetcher and manual drops.
        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        self.samples: list[tuple[Path, int]] = []
        for c in self.classes:
            class_dir = self.root / c.name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory missing: {class_dir}"
                )
            for ext in valid_exts:
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, c.id))

        # Augmentation pipeline. Geometric and photometric transforms are
        # safe for fine-grained connector classification IF we deliberately
        # avoid scale jitter — physical size is the discriminative signal
        # for 2.4/2.92/3.5 mm and we don't want the model to learn invariance
        # to it.
        if augment:
            self._train_aug = transforms.Compose([
                transforms.RandomRotation(degrees=15, fill=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ])
        else:
            self._train_aug = None

        self._rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        # Resize before augmentation so transforms operate on a uniform canvas.
        img = transforms.Resize((self.image_size, self.image_size))(img)
        if self._train_aug is not None:
            img = self._train_aug(img)
        rgb = transforms.ToTensor()(img)
        rgb = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb)

        # Build a simple "center disc" foreground mask of the same size.
        # This is intentionally crude for proxy data; the real dataset uses
        # ground-truth masks from capture or rendering.
        h = w = self.image_size
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        radius = min(h, w) * 0.35
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

        depth_np = synthesize_depth_from_mask(
            mask=mask,
            focal_length_px=500.0,
            object_depth_m=self.object_depth_m,
            seed=idx,  # deterministic per-item
        )
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # (1, H, W)

        rgbd = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        return rgbd, label
