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
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.object_depth_m = object_depth_m
        self.classes = load_classes(classes_yaml)
        self._name_to_id = {c.name: c.id for c in self.classes}

        self.samples: list[tuple[Path, int]] = []
        for c in self.classes:
            class_dir = self.root / c.name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory missing: {class_dir}"
                )
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, c.id))

        self._rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # → [0, 1]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self._rgb_transform(img)  # (3, H, W)

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
