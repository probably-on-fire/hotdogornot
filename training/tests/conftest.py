from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """Create a tiny synthetic dataset on disk for testing."""
    root = tmp_path / "proxy"
    root.mkdir()

    # 2 classes × 3 images each, 64×64 RGB
    for class_name in ["SMA-M", "SMA-F"]:
        class_dir = root / class_name
        class_dir.mkdir()
        for i in range(3):
            rng = np.random.default_rng(hash(class_name + str(i)) % (2**32))
            arr = (rng.integers(0, 255, (64, 64, 3))).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i:03d}.png")

    return root


@pytest.fixture
def classes_yaml(tmp_path: Path) -> Path:
    """Minimal classes.yaml for tests."""
    content = """
classes:
  - {id: 0, name: "SMA-M", family: "sma", gender: "male", inner_pin_diameter_mm: 0.91, frequency_ghz_max: 18.0, impedance_ohms: 50, mating_torque_in_lb: 8.0}
  - {id: 1, name: "SMA-F", family: "sma", gender: "female", inner_pin_diameter_mm: 1.27, frequency_ghz_max: 18.0, impedance_ohms: 50, mating_torque_in_lb: 8.0}
"""
    p = tmp_path / "classes.yaml"
    p.write_text(content, encoding="utf-8")
    return p
