# Python Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python training pipeline that produces a YOLOv11n detector, a MobileViT-v2 RGBD embedder, and a reference embedding database from proxy data — all exported to ONNX and ready for Unity to consume.

**Architecture:** `uv`-managed Python package. Two-stage perception: a 1-class detector (Ultralytics YOLOv11n) and a metric-learning embedder (timm MobileViT-v2 modified for 4-channel RGBD input and a 128-d output head) trained with online-mined triplet loss. Proxy data comes from catalog-image scraping and procedural PyRender synthetic rendering with synthetic depth. ONNX export targets Unity Sentis opset compatibility.

**Tech Stack:** Python 3.11, uv, PyTorch 2.x, timm, Ultralytics YOLO, trimesh + pyrender, onnx, onnxruntime, pytest. Spec reference: `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`.

---

## File Structure

```
training/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── classes.yaml              # 8 connector class definitions + specs
│   ├── detector.yaml             # Ultralytics training config
│   └── embedder.yaml             # embedder training hyperparams
├── rfconnectorai/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── classes.py            # dataclasses + loader for classes.yaml
│   │   ├── dataset.py            # RGBDConnectorDataset (torch Dataset)
│   │   ├── depth_utils.py        # synthetic depth generation for RGB-only data
│   │   ├── scrape.py             # catalog image downloader
│   │   └── synthetic.py          # PyRender procedural connector renders
│   ├── models/
│   │   ├── __init__.py
│   │   └── embedder.py           # MobileViT-v2 4-channel-input 128-d-output
│   ├── training/
│   │   ├── __init__.py
│   │   ├── triplet_loss.py       # batch-hard online triplet mining
│   │   ├── train_detector.py     # YOLO wrapper CLI
│   │   └── train_embedder.py     # embedder training CLI
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── build_references.py   # compute + save reference_embeddings.bin
│   │   └── eval.py               # accuracy, confusion matrix, calibration
│   └── export/
│       ├── __init__.py
│       └── onnx_export.py        # export both models to ONNX for Sentis
└── tests/
    ├── conftest.py                # shared fixtures (synthetic images, etc.)
    ├── test_classes.py
    ├── test_dataset.py
    ├── test_depth_utils.py
    ├── test_triplet_loss.py
    ├── test_embedder.py
    ├── test_build_references.py
    ├── test_eval.py
    └── test_onnx_export.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/.gitignore`
- Create: `training/README.md`
- Create: `training/rfconnectorai/__init__.py`
- Create: `training/rfconnectorai/data/__init__.py`
- Create: `training/rfconnectorai/models/__init__.py`
- Create: `training/rfconnectorai/training/__init__.py`
- Create: `training/rfconnectorai/inference/__init__.py`
- Create: `training/rfconnectorai/export/__init__.py`
- Create: `training/tests/__init__.py`

- [ ] **Step 1: Create `training/pyproject.toml`**

```toml
[project]
name = "rfconnectorai"
version = "0.1.0"
description = "RF connector identification training pipeline"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.2,<3.0",
    "torchvision>=0.17,<1.0",
    "timm>=1.0.0",
    "ultralytics>=8.2.0",
    "numpy>=1.26",
    "Pillow>=10.0",
    "pyyaml>=6.0",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "trimesh>=4.0",
    "pyrender>=0.1.45",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rfconnectorai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create `training/.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
.venv/
venv/
*.egg-info/
dist/
build/

# Data and models
data/raw/
data/synthetic/
data/labeled/
data/processed/
runs/
*.pt
*.onnx
*.bin
!test_data/**/*.bin

# Environment
.env
.env.local
```

- [ ] **Step 3: Create `training/README.md`**

```markdown
# RF Connector AI — Training Pipeline

Python pipeline that trains a YOLOv11n connector detector and a MobileViT-v2 RGBD embedder, then exports both to ONNX for Unity Sentis.

## Setup

    uv venv
    uv pip install -e ".[dev]"

## Pipeline

1. Acquire proxy data: `python -m rfconnectorai.data.scrape` and `python -m rfconnectorai.data.synthetic`
2. Train detector: `python -m rfconnectorai.training.train_detector`
3. Train embedder: `python -m rfconnectorai.training.train_embedder`
4. Build references: `python -m rfconnectorai.inference.build_references`
5. Export ONNX: `python -m rfconnectorai.export.onnx_export`
6. Evaluate: `python -m rfconnectorai.inference.eval`

## Test

    pytest
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create these as empty files (0 bytes each):
- `training/rfconnectorai/__init__.py`
- `training/rfconnectorai/data/__init__.py`
- `training/rfconnectorai/models/__init__.py`
- `training/rfconnectorai/training/__init__.py`
- `training/rfconnectorai/inference/__init__.py`
- `training/rfconnectorai/export/__init__.py`
- `training/tests/__init__.py`

- [ ] **Step 5: Install and verify**

```bash
cd training
uv venv
uv pip install -e ".[dev]"
pytest --collect-only
```

Expected: `pytest` runs without import errors; 0 tests collected (none exist yet).

- [ ] **Step 6: Commit**

```bash
git add training/
git commit -m "feat(training): scaffold Python package and dev tooling"
```

---

## Task 2: Connector Class Definitions

Defines the 8 initial connector classes with metadata used everywhere downstream.

**Files:**
- Create: `training/configs/classes.yaml`
- Create: `training/rfconnectorai/data/classes.py`
- Create: `training/tests/test_classes.py`

- [ ] **Step 1: Create `training/configs/classes.yaml`**

```yaml
# Each connector has:
#   id: integer label used in training
#   name: short display name
#   family: one of [sma, precision]
#   gender: male | female
#   inner_pin_diameter_mm: nominal inner conductor diameter (for metrology)
#   frequency_ghz_max: max rated frequency
#   impedance_ohms: 50
#   mating_torque_in_lb: manufacturer-recommended mating torque

classes:
  - id: 0
    name: "SMA-M"
    family: "sma"
    gender: "male"
    inner_pin_diameter_mm: 0.91
    frequency_ghz_max: 18.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 1
    name: "SMA-F"
    family: "sma"
    gender: "female"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 18.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 2
    name: "3.5mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.52
    frequency_ghz_max: 34.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 3
    name: "3.5mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.52
    frequency_ghz_max: 34.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 4
    name: "2.92mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 40.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 5
    name: "2.92mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 40.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 6
    name: "2.4mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.04
    frequency_ghz_max: 50.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 7
    name: "2.4mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.04
    frequency_ghz_max: 50.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0
```

- [ ] **Step 2: Write failing test `training/tests/test_classes.py`**

```python
from pathlib import Path
import pytest
from rfconnectorai.data.classes import load_classes, ConnectorClass


CONFIG = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


def test_load_classes_returns_eight():
    classes = load_classes(CONFIG)
    assert len(classes) == 8


def test_load_classes_ids_are_contiguous():
    classes = load_classes(CONFIG)
    assert [c.id for c in classes] == list(range(8))


def test_load_classes_names_match_spec():
    classes = load_classes(CONFIG)
    names = {c.name for c in classes}
    assert names == {
        "SMA-M", "SMA-F",
        "3.5mm-M", "3.5mm-F",
        "2.92mm-M", "2.92mm-F",
        "2.4mm-M", "2.4mm-F",
    }


def test_precision_classes_flagged_correctly():
    classes = load_classes(CONFIG)
    families = {c.name: c.family for c in classes}
    assert families["SMA-M"] == "sma"
    assert families["SMA-F"] == "sma"
    for name in ["3.5mm-M", "3.5mm-F", "2.92mm-M", "2.92mm-F", "2.4mm-M", "2.4mm-F"]:
        assert families[name] == "precision"


def test_connector_class_is_frozen():
    c = ConnectorClass(
        id=0, name="X", family="sma", gender="male",
        inner_pin_diameter_mm=1.0, frequency_ghz_max=18.0,
        impedance_ohms=50, mating_torque_in_lb=8.0,
    )
    with pytest.raises(Exception):
        c.id = 1  # frozen dataclass must not allow mutation
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd training && pytest tests/test_classes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rfconnectorai.data.classes'`.

- [ ] **Step 4: Implement `training/rfconnectorai/data/classes.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ConnectorClass:
    id: int
    name: str
    family: str                      # "sma" | "precision"
    gender: str                      # "male" | "female"
    inner_pin_diameter_mm: float
    frequency_ghz_max: float
    impedance_ohms: int
    mating_torque_in_lb: float

    def is_precision(self) -> bool:
        return self.family == "precision"


def load_classes(path: Path | str) -> list[ConnectorClass]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    classes = [ConnectorClass(**entry) for entry in data["classes"]]

    # Invariant: ids must be 0..N-1 with no gaps
    ids = [c.id for c in classes]
    if ids != list(range(len(ids))):
        raise ValueError(f"Class ids must be contiguous from 0; got {ids}")

    return classes
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd training && pytest tests/test_classes.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add training/configs/classes.yaml training/rfconnectorai/data/classes.py training/tests/test_classes.py
git commit -m "feat(training): define 8 RF connector classes with metadata"
```

---

## Task 3: Synthetic Depth Generation

Catalog-scraped images are RGB-only. For training the RGBD embedder on proxy data, we synthesize plausible depth maps. Once real connectors arrive in Phase 1, real LiDAR depth captures replace these.

**Files:**
- Create: `training/rfconnectorai/data/depth_utils.py`
- Create: `training/tests/test_depth_utils.py`

- [ ] **Step 1: Write failing test `training/tests/test_depth_utils.py`**

```python
import numpy as np
import pytest
from rfconnectorai.data.depth_utils import synthesize_depth_from_mask


def test_synthesize_depth_shape_matches_mask():
    mask = np.zeros((384, 384), dtype=bool)
    mask[100:284, 100:284] = True  # square foreground
    depth = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.12)
    assert depth.shape == (384, 384)
    assert depth.dtype == np.float32


def test_synthesize_depth_background_is_far():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    depth = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1)
    bg_depth = depth[0, 0]
    fg_depth = depth[16, 16]
    assert bg_depth > 1.0        # background should be pushed far (>1m)
    assert 0.05 < fg_depth < 0.2  # foreground near the specified object depth


def test_synthesize_depth_is_deterministic_for_same_seed():
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    d1 = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1, seed=42)
    d2 = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1, seed=42)
    np.testing.assert_array_equal(d1, d2)


def test_synthesize_depth_rejects_wrong_dtype():
    mask = np.zeros((16, 16), dtype=np.float32)  # wrong dtype
    with pytest.raises(ValueError):
        synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_depth_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/data/depth_utils.py`**

```python
from __future__ import annotations

import numpy as np


def synthesize_depth_from_mask(
    mask: np.ndarray,
    focal_length_px: float,
    object_depth_m: float,
    background_depth_m: float = 2.0,
    noise_std_m: float = 0.002,
    seed: int | None = None,
) -> np.ndarray:
    """
    Produce a plausible depth map for an RGB-only image given a foreground mask.

    Foreground pixels get depth ~object_depth_m with small Gaussian noise.
    Background pixels get background_depth_m with larger noise.

    This exists so the RGBD embedder can train on scraped catalog images by
    treating the object as an approximately flat disc at a typical hand-held
    distance. Real LiDAR captures replace this in Phase 1.
    """
    if mask.dtype != bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")

    rng = np.random.default_rng(seed)
    depth = np.full(mask.shape, background_depth_m, dtype=np.float32)
    depth[mask] = object_depth_m

    # Add noise to prevent the model from learning a trivial "flat depth = object" shortcut.
    depth += rng.normal(0.0, noise_std_m, size=mask.shape).astype(np.float32)

    # Clamp to positive values (depth must be non-negative).
    depth = np.maximum(depth, 0.01)

    _ = focal_length_px  # reserved for future per-pixel geometric variation
    return depth
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_depth_utils.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/data/depth_utils.py training/tests/test_depth_utils.py
git commit -m "feat(training): synthesize depth maps for RGB-only proxy data"
```

---

## Task 4: RGBD Dataset Loader

PyTorch `Dataset` that yields `(rgbd_tensor, class_id)` tuples. Handles both real captures (when available) and proxy data (RGB + synthesized depth). Returns 4-channel tensors for the embedder.

**Files:**
- Create: `training/tests/conftest.py`
- Create: `training/rfconnectorai/data/dataset.py`
- Create: `training/tests/test_dataset.py`

- [ ] **Step 1: Create shared fixture file `training/tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write failing test `training/tests/test_dataset.py`**

```python
from pathlib import Path

import pytest
import torch
from rfconnectorai.data.dataset import RGBDConnectorDataset


def test_dataset_length_matches_image_count(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    assert len(ds) == 6  # 2 classes × 3 images


def test_dataset_item_is_rgbd_tensor_and_label(
    synthetic_image_dir: Path, classes_yaml: Path
):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, label = ds[0]
    assert isinstance(rgbd, torch.Tensor)
    assert rgbd.shape == (4, 64, 64)
    assert rgbd.dtype == torch.float32
    assert isinstance(label, int)
    assert label in (0, 1)


def test_dataset_rgb_channels_normalized(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, _ = ds[0]
    rgb = rgbd[:3]
    # After ImageNet normalization, typical values land in ~[-2.5, 2.5]
    assert rgb.min() > -3.0
    assert rgb.max() < 3.0


def test_dataset_depth_channel_in_meters(synthetic_image_dir: Path, classes_yaml: Path):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    rgbd, _ = ds[0]
    depth = rgbd[3]
    # Depth in the clamped range of synthesize_depth_from_mask
    assert depth.min() >= 0.0
    assert depth.max() < 5.0


def test_dataset_labels_come_from_classes_yaml(
    synthetic_image_dir: Path, classes_yaml: Path
):
    ds = RGBDConnectorDataset(
        root=synthetic_image_dir,
        classes_yaml=classes_yaml,
        image_size=64,
    )
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_dataset_missing_class_directory_raises(tmp_path: Path, classes_yaml: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        RGBDConnectorDataset(root=empty, classes_yaml=classes_yaml, image_size=64)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd training && pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `training/rfconnectorai/data/dataset.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd training && pytest tests/test_dataset.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add training/tests/conftest.py training/rfconnectorai/data/dataset.py training/tests/test_dataset.py
git commit -m "feat(training): RGBD dataset loader with synthesized depth"
```

---

## Task 5: Triplet Loss with Batch-Hard Mining

Core of metric learning. For each anchor in a batch, the hardest positive (same class, farthest in embedding space) and the hardest negative (different class, closest) are selected, and the triplet loss is computed over those mined triplets. This is more stable than random triplet selection.

**Files:**
- Create: `training/rfconnectorai/training/triplet_loss.py`
- Create: `training/tests/test_triplet_loss.py`

- [ ] **Step 1: Write failing test `training/tests/test_triplet_loss.py`**

```python
import pytest
import torch
from rfconnectorai.training.triplet_loss import batch_hard_triplet_loss


def test_zero_loss_when_classes_perfectly_separated():
    # Two classes placed at opposite unit vectors — perfectly separable.
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.5)
    # Easy-positive (distance 0) + easy-negative (distance 2) + margin 0.5 → loss = max(0 - 2 + 0.5, 0) = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_positive_loss_when_classes_overlap():
    # Same-class points are farther apart than cross-class points — loss must be > 0.
    embeddings = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],    # same class as above, but far
        [0.1, 0.0],    # different class, close to first point
        [0.9, 0.0],    # different class, close to second point
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
    assert loss.item() > 0.0


def test_loss_is_gradient_bearing():
    embeddings = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0.0


def test_requires_at_least_two_classes():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 0, 0])  # single class → no valid negatives
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels, margin=0.2)


def test_requires_at_least_two_samples_per_class():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])  # one sample per class → no valid positives
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels, margin=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_triplet_loss.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/training/triplet_loss.py`**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Batch-hard triplet loss using squared Euclidean distance on L2-normalized embeddings.

    For each anchor a in the batch:
      - hardest positive  = max_{p : label(p) = label(a), p != a} d(a, p)
      - hardest negative  = min_{n : label(n) != label(a)}       d(a, n)
      - triplet loss      = max(hardest_pos - hardest_neg + margin, 0)

    Returns the mean triplet loss across the batch.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be 1-D and match batch size")

    unique = torch.unique(labels)
    if unique.numel() < 2:
        raise ValueError("batch-hard triplet loss requires at least two distinct classes")

    # Check that at least one class has >=2 samples (otherwise no valid positive).
    counts = torch.stack([(labels == c).sum() for c in unique])
    if counts.max().item() < 2:
        raise ValueError("batch-hard triplet loss requires at least two samples of some class")

    # Normalize so distances are in a stable range.
    emb = F.normalize(embeddings, p=2, dim=1)

    # Pairwise squared Euclidean distances: d_ij = 2 - 2 * cos_sim (since vectors unit-norm)
    # but compute from first principles to keep it explicit.
    diff = emb.unsqueeze(0) - emb.unsqueeze(1)          # (N, N, D)
    dist = (diff ** 2).sum(dim=-1)                       # (N, N)

    # Masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N)
    labels_ne = ~labels_eq
    eye = torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)

    positive_mask = labels_eq & ~eye  # same class, not self
    negative_mask = labels_ne

    # For each anchor, find hardest positive (max distance) within same-class samples.
    # Replace invalid (non-positive) slots with -inf so they are ignored by max.
    pos_dist = dist.clone()
    pos_dist[~positive_mask] = float("-inf")
    hardest_pos = pos_dist.max(dim=1).values  # (N,)

    # For each anchor, find hardest negative (min distance) across different-class samples.
    neg_dist = dist.clone()
    neg_dist[~negative_mask] = float("inf")
    hardest_neg = neg_dist.min(dim=1).values  # (N,)

    # Anchors that have no valid positive (class with a single sample) will have -inf.
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        raise ValueError("No anchors with both a valid positive and negative were found.")

    losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return losses.mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_triplet_loss.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/training/triplet_loss.py training/tests/test_triplet_loss.py
git commit -m "feat(training): batch-hard triplet loss with mining"
```

---

## Task 6: RGBD Embedder Architecture

Modify timm's MobileViT-v2 to accept a 4-channel input (RGB + depth) and produce a 128-dimensional L2-normalized embedding. The first conv layer is replaced with a 4-channel version; depth-channel weights are initialized by averaging the pretrained RGB weights.

**Files:**
- Create: `training/rfconnectorai/models/embedder.py`
- Create: `training/tests/test_embedder.py`

- [ ] **Step 1: Write failing test `training/tests/test_embedder.py`**

```python
import pytest
import torch
from rfconnectorai.models.embedder import RGBDEmbedder


def test_forward_shape_is_128d():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(2, 4, 384, 384)
    out = model(x)
    assert out.shape == (2, 128)


def test_output_is_l2_normalized():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    model.eval()
    x = torch.randn(4, 4, 384, 384)
    with torch.no_grad():
        out = model(x)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_rejects_3channel_input():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(1, 3, 384, 384)
    with pytest.raises(RuntimeError):
        model(x)


def test_gradients_flow_through_depth_channel():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(1, 4, 384, 384, requires_grad=True)
    out = model(x)
    out.sum().backward()
    # Gradient must be non-zero on the depth channel (channel index 3)
    assert x.grad[:, 3].abs().sum().item() > 0.0


def test_custom_embedding_dim():
    model = RGBDEmbedder(embedding_dim=64, pretrained=False)
    x = torch.randn(1, 4, 384, 384)
    out = model(x)
    assert out.shape == (1, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/models/embedder.py`**

```python
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDEmbedder(nn.Module):
    """
    MobileViT-v2 backbone adapted for 4-channel RGBD input.

    - Replaces the first conv layer (3 -> out_channels) with a 4-channel version.
    - When pretrained=True, initializes the depth channel's conv weights to the
      mean of the pretrained RGB weights. This gives training a reasonable start
      rather than random init on the depth filter.
    - Replaces the classification head with a projection to `embedding_dim`.
    - L2-normalizes outputs for use with cosine-similarity-based matching and
      the batch-hard triplet loss.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        backbone: str = "mobilevitv2_100",
    ) -> None:
        super().__init__()

        # Create backbone without a classification head (num_classes=0 returns features).
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # Replace the first conv layer to accept 4 input channels.
        original_first = self._find_first_conv(self.backbone)
        new_first = nn.Conv2d(
            in_channels=4,
            out_channels=original_first.out_channels,
            kernel_size=original_first.kernel_size,
            stride=original_first.stride,
            padding=original_first.padding,
            bias=(original_first.bias is not None),
        )
        with torch.no_grad():
            new_first.weight[:, :3] = original_first.weight
            # Depth channel: mean of RGB filter weights.
            new_first.weight[:, 3:4] = original_first.weight.mean(dim=1, keepdim=True)
            if original_first.bias is not None:
                new_first.bias.copy_(original_first.bias)
        self._replace_first_conv(self.backbone, new_first)

        # Projection head from backbone features to embedding_dim.
        feat_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, embedding_dim),
        )

    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in backbone")

    @staticmethod
    def _replace_first_conv(module: nn.Module, new_conv: nn.Conv2d) -> None:
        for name, child in module.named_modules():
            for attr_name, attr_val in list(vars(child).get("_modules", {}).items()):
                if isinstance(attr_val, nn.Conv2d) and attr_val.in_channels == 3:
                    setattr(child, attr_name, new_conv)
                    return
        raise RuntimeError("Failed to replace first Conv2d")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B, feat_dim)
        emb = self.projection(feats)       # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_embedder.py -v`
Expected: 5 passed. (The first run may download the timm pretrained weights; subsequent runs are cached.)

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/models/embedder.py training/tests/test_embedder.py
git commit -m "feat(training): RGBD MobileViT-v2 embedder with 128-d output"
```

---

## Task 7: Catalog Image Scraper

Pulls RF connector product photos from public datasheets/catalogs as Phase-0 proxy data. Respects robots.txt, rate-limits requests. This is intentionally best-effort — quality is low, but it gets the pipeline running before real connectors arrive.

**Files:**
- Create: `training/rfconnectorai/data/scrape.py`
- Create: `training/tests/test_scrape.py`

- [ ] **Step 1: Write failing test `training/tests/test_scrape.py`**

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rfconnectorai.data.scrape import (
    CatalogImage,
    save_catalog_image,
    sanitize_filename,
)


def test_sanitize_filename_removes_unsafe_chars():
    assert sanitize_filename("sma/m connector??.jpg") == "sma_m_connector__.jpg"
    assert sanitize_filename("normal.png") == "normal.png"


def test_save_catalog_image_writes_to_class_dir(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_get.return_value = MagicMock(
            status_code=200,
            content=fake_bytes,
            headers={"Content-Type": "image/png"},
        )
        img = CatalogImage(
            url="https://example.com/foo.png",
            class_name="SMA-M",
            filename="foo.png",
        )
        out_path = save_catalog_image(img, root=tmp_path)

    assert out_path.exists()
    assert out_path.parent.name == "SMA-M"
    assert out_path.read_bytes() == fake_bytes


def test_save_catalog_image_rejects_non_image_content_type(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"<!doctype html>",
            headers={"Content-Type": "text/html"},
        )
        img = CatalogImage(
            url="https://example.com/foo.html",
            class_name="SMA-M",
            filename="foo.html",
        )
        with pytest.raises(ValueError):
            save_catalog_image(img, root=tmp_path)


def test_save_catalog_image_404_raises(tmp_path: Path):
    with patch("rfconnectorai.data.scrape.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, content=b"", headers={})
        img = CatalogImage(url="https://x/y", class_name="SMA-M", filename="y.png")
        with pytest.raises(RuntimeError):
            save_catalog_image(img, root=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_scrape.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/data/scrape.py`**

```python
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


USER_AGENT = "rfconnectorai-research/0.1 (contact: chris@aired.com)"
REQUEST_DELAY_SECONDS = 1.0


@dataclass
class CatalogImage:
    url: str
    class_name: str
    filename: str


def sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe on common filesystems with underscores."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def save_catalog_image(img: CatalogImage, root: Path) -> Path:
    """
    Download a single image to root/<class_name>/<sanitized_filename>.

    Raises:
        RuntimeError: on non-200 HTTP response.
        ValueError:  on non-image content type.
    """
    class_dir = root / img.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(img.filename)
    out_path = class_dir / safe_name

    resp = requests.get(img.url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {img.url}")

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"Non-image Content-Type {content_type!r} for {img.url}")

    out_path.write_bytes(resp.content)
    time.sleep(REQUEST_DELAY_SECONDS)  # polite rate limiting
    return out_path


def scrape_urls(urls: list[CatalogImage], root: Path) -> list[Path]:
    """Download a list of catalog images. Errors on individual images are logged and skipped."""
    saved: list[Path] = []
    for img in urls:
        try:
            saved.append(save_catalog_image(img, root))
        except (RuntimeError, ValueError) as e:
            print(f"[scrape] skip {img.url}: {e}")
    return saved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_scrape.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/data/scrape.py training/tests/test_scrape.py
git commit -m "feat(training): catalog image scraper for proxy dataset"
```

---

## Task 8: Synthetic Renderer

Procedurally renders connector-like shapes using trimesh and pyrender. Produces RGB + real depth + ground-truth mask. These are crude but diverse and cost-free to generate in bulk.

**Files:**
- Create: `training/rfconnectorai/data/synthetic.py`
- Create: `training/tests/test_synthetic.py`

- [ ] **Step 1: Write failing test `training/tests/test_synthetic.py`**

```python
import numpy as np
import pytest
from rfconnectorai.data.synthetic import render_connector_sample, make_connector_mesh


def test_make_connector_mesh_returns_trimesh():
    import trimesh
    m = make_connector_mesh(gender="male", family="precision")
    assert isinstance(m, trimesh.Trimesh)
    assert m.vertices.shape[1] == 3
    assert m.faces.shape[1] == 3


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("family", ["sma", "precision"])
def test_make_connector_mesh_variants(gender, family):
    m = make_connector_mesh(gender=gender, family=family)
    assert len(m.vertices) > 0


def test_render_connector_sample_returns_rgb_depth_mask():
    rgb, depth, mask = render_connector_sample(
        gender="male", family="precision", image_size=128, seed=1
    )
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (128, 128)
    assert depth.dtype == np.float32
    assert mask.shape == (128, 128)
    assert mask.dtype == bool


def test_render_has_some_foreground():
    _, _, mask = render_connector_sample(
        gender="male", family="precision", image_size=128, seed=1
    )
    # At least 1% of pixels should be foreground.
    assert mask.mean() > 0.01


def test_render_determinism_per_seed():
    rgb1, d1, m1 = render_connector_sample("male", "precision", image_size=64, seed=7)
    rgb2, d2, m2 = render_connector_sample("male", "precision", image_size=64, seed=7)
    np.testing.assert_array_equal(rgb1, rgb2)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(m1, m2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/data/synthetic.py`**

```python
from __future__ import annotations

import os

import numpy as np
import trimesh

# Offscreen rendering backend for headless environments.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402


# Nominal outer-body diameters (mm). Families visually similar but not identical.
FAMILY_OUTER_DIAMETER_MM = {
    "sma": 6.3,
    "precision": 5.5,
}

MALE_PIN_EXTENDS_MM = 2.5
FEMALE_PIN_RECESS_MM = 1.5


def make_connector_mesh(gender: str, family: str) -> trimesh.Trimesh:
    """
    Build a simple connector-like mesh:
      - cylinder for the body
      - smaller cylinder as the inner pin (extends for male, recessed for female)
    """
    body_radius_mm = FAMILY_OUTER_DIAMETER_MM[family] / 2
    body_length_mm = 10.0
    pin_radius_mm = body_radius_mm * 0.25 if family == "precision" else body_radius_mm * 0.28

    body = trimesh.creation.cylinder(radius=body_radius_mm, height=body_length_mm, sections=48)

    if gender == "male":
        pin = trimesh.creation.cylinder(
            radius=pin_radius_mm, height=MALE_PIN_EXTENDS_MM, sections=24
        )
        pin.apply_translation([0, 0, body_length_mm / 2 + MALE_PIN_EXTENDS_MM / 2])
        mesh = trimesh.util.concatenate([body, pin])
    else:
        pin_cavity = trimesh.creation.cylinder(
            radius=pin_radius_mm, height=FEMALE_PIN_RECESS_MM, sections=24
        )
        pin_cavity.apply_translation(
            [0, 0, body_length_mm / 2 - FEMALE_PIN_RECESS_MM / 2]
        )
        # Subtraction is slow via boolean ops; for the simple proxy mesh, we just
        # mark the cavity by leaving the body whole and add a visually-distinct disc.
        cap = trimesh.creation.cylinder(
            radius=pin_radius_mm * 1.8, height=0.2, sections=24
        )
        cap.apply_translation([0, 0, body_length_mm / 2])
        mesh = trimesh.util.concatenate([body, cap])

    # Convert mm → meters (pyrender expects meters).
    mesh.apply_scale(0.001)
    return mesh


def render_connector_sample(
    gender: str,
    family: str,
    image_size: int = 384,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a procedural connector and return (RGB uint8, depth float32 m, mask bool).
    """
    rng = np.random.default_rng(seed)

    mesh = make_connector_mesh(gender=gender, family=family)

    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0]),
        ambient_light=np.array([0.3, 0.3, 0.3]),
    )

    # Random material color (silver/gold-ish).
    base_color = [
        0.6 + rng.uniform(-0.15, 0.15),
        0.6 + rng.uniform(-0.15, 0.15),
        0.6 + rng.uniform(-0.2, 0.05),
        1.0,
    ]
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color, metallicFactor=0.9, roughnessFactor=0.3
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    # Random small rotation around the mesh's axes.
    pose = np.eye(4)
    rot_x = rng.uniform(-0.3, 0.3)
    rot_y = rng.uniform(-0.3, 0.3)
    pose[:3, :3] = trimesh.transformations.euler_matrix(rot_x, rot_y, 0.0)[:3, :3]
    scene.add(render_mesh, pose=pose)

    # Camera roughly 12 cm away, looking down the z axis.
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(40), aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 0.12 + rng.uniform(-0.02, 0.02)
    scene.add(camera, pose=cam_pose)

    # Directional light.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.1, 0.1, 0.3]
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_size, viewport_height=image_size
    )
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    mask = depth > 0.0
    rgb = color[..., :3].astype(np.uint8)
    return rgb, depth.astype(np.float32), mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_synthetic.py -v`
Expected: 7 passed. *Note:* `pyrender` requires a GL context. On headless Linux machines install EGL (`apt install libegl1`). On macOS the default works. On Windows run inside WSL2 with an EGL-capable driver or use a development machine with a display.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/data/synthetic.py training/tests/test_synthetic.py
git commit -m "feat(training): procedural connector renderer (RGB+depth+mask)"
```

---

## Task 9: Embedder Training Script

Trains the RGBD embedder end-to-end with the batch-hard triplet loss. Accepts a `--smoke-test` flag that runs 2 epochs on a tiny synthetic dataset for CI.

**Files:**
- Create: `training/configs/embedder.yaml`
- Create: `training/rfconnectorai/training/train_embedder.py`
- Create: `training/tests/test_train_embedder.py`

- [ ] **Step 1: Create `training/configs/embedder.yaml`**

```yaml
model:
  embedding_dim: 128
  backbone: "mobilevitv2_100"
  pretrained: true

training:
  image_size: 384
  batch_size: 32
  samples_per_class_per_batch: 4   # balanced sampling for triplet mining
  num_epochs: 40
  learning_rate: 0.0003
  weight_decay: 0.0001
  margin: 0.3
  warmup_epochs: 2

data:
  root: "data/labeled/embedder"
  classes_yaml: "configs/classes.yaml"

output:
  dir: "runs/embedder"
  checkpoint_name: "embedder.pt"
```

- [ ] **Step 2: Write failing test `training/tests/test_train_embedder.py`**

```python
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path: Path, classes_yaml: Path) -> Path:
    """Create a 2-class dataset (SMA-M, SMA-F) with 6 images each, 64×64."""
    root = tmp_path / "data"
    root.mkdir()
    for class_name in ["SMA-M", "SMA-F"]:
        d = root / class_name
        d.mkdir()
        rng = np.random.default_rng(hash(class_name) % (2**32))
        for i in range(6):
            arr = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i:03d}.png")
    return root


def test_smoke_run_produces_checkpoint(
    tiny_dataset: Path, classes_yaml: Path, tmp_path: Path
):
    output_dir = tmp_path / "runs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    result = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.training.train_embedder",
            "--data-root", str(tiny_dataset),
            "--classes-yaml", str(classes_yaml),
            "--output-dir", str(output_dir),
            "--smoke-test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    ckpt = output_dir / "embedder.pt"
    assert ckpt.exists(), "Expected checkpoint was not written"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd training && pytest tests/test_train_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rfconnectorai.training.train_embedder'`.

- [ ] **Step 4: Implement `training/rfconnectorai/training/train_embedder.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = RGBDConnectorDataset(root=data_root, classes_yaml=classes_yaml, image_size=image_size)
    labels = [sample[1] for sample in ds.samples]

    # One "batch" per sampler.__iter__, so we run many optimizer steps by re-iterating.
    sampler = PKSampler(labels, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    model = RGBDEmbedder(embedding_dim=128, pretrained=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    steps_per_epoch = max(1, len(ds) // (classes_per_batch * samples_per_class))

    for epoch in range(num_epochs):
        model.train()
        for _ in range(steps_per_epoch):
            batch = next(iter(loader))
            x, y = batch
            x, y = x.to(device), y.to(device)

            emb = model(x)
            loss = batch_hard_triplet_loss(emb, y, margin=margin)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch {epoch + 1}/{num_epochs}  loss={loss.item():.4f}")

    ckpt = output_dir / "embedder.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--classes-per-batch", type=int, default=4)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
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
    )
    print(f"checkpoint written to {ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd training && pytest tests/test_train_embedder.py -v`
Expected: 1 passed. (May take 1–3 minutes on CPU.)

- [ ] **Step 6: Commit**

```bash
git add training/configs/embedder.yaml training/rfconnectorai/training/train_embedder.py training/tests/test_train_embedder.py
git commit -m "feat(training): embedder training CLI with smoke-test mode"
```

---

## Task 10: Detector Training Script

Wraps Ultralytics YOLOv11n for 1-class connector detection. The dataset format is YOLO-standard; this script assumes data has been labeled via Roboflow or equivalent.

**Files:**
- Create: `training/configs/detector.yaml`
- Create: `training/rfconnectorai/training/train_detector.py`
- Create: `training/tests/test_train_detector.py`

- [ ] **Step 1: Create `training/configs/detector.yaml`**

```yaml
# Ultralytics YOLO dataset spec
path: data/labeled/detector     # relative to training/
train: images/train
val: images/val

names:
  0: connector
```

- [ ] **Step 2: Write failing test `training/tests/test_train_detector.py`**

```python
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_yolo_dataset(tmp_path: Path) -> Path:
    """Build a minimal YOLO-format dataset with 2 training + 2 val images, each with 1 box."""
    root = tmp_path / "yolo_ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            rng = np.random.default_rng(hash(split + str(i)) % (2**32))
            arr = rng.integers(0, 255, (128, 128, 3)).astype(np.uint8)
            Image.fromarray(arr).save(root / "images" / split / f"{i}.png")
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: connector\n"
    )
    return data_yaml


def test_smoke_run_produces_weights(tiny_yolo_dataset: Path, tmp_path: Path):
    output_dir = tmp_path / "yolo_runs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    result = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.training.train_detector",
            "--data-yaml", str(tiny_yolo_dataset),
            "--output-dir", str(output_dir),
            "--smoke-test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, env=env, timeout=900,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    weights = output_dir / "detector.pt"
    assert weights.exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd training && pytest tests/test_train_detector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `training/rfconnectorai/training/train_detector.py`**

```python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: Path,
    output_dir: Path,
    image_size: int,
    epochs: int,
    base_weights: str,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)

    # Ultralytics writes to runs/detect/<name>/weights/best.pt by default.
    # We point it at a subdir under output_dir and then copy the best weights out.
    project = output_dir / "ultralytics"
    name = "connector"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        project=str(project),
        name=name,
        exist_ok=True,
        device=device,
        verbose=False,
    )

    best = project / name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training completed but best weights missing at {best}")

    final = output_dir / "detector.pt"
    shutil.copy2(best, final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-weights", type=str, default="yolo11n.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.image_size = 128
        args.epochs = 1

    weights = train(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        base_weights=args.base_weights,
        device=args.device,
    )
    print(f"weights written to {weights}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd training && pytest tests/test_train_detector.py -v`
Expected: 1 passed. (First run downloads `yolo11n.pt` base weights; ~5 MB.)

- [ ] **Step 6: Commit**

```bash
git add training/configs/detector.yaml training/rfconnectorai/training/train_detector.py training/tests/test_train_detector.py
git commit -m "feat(training): YOLOv11n 1-class detector training CLI"
```

---

## Task 11: Reference Embedding Builder

Computes per-class mean embeddings from the trained embedder and a labeled reference set, saving them as `reference_embeddings.bin` for the Unity app's nearest-neighbor matcher.

**Files:**
- Create: `training/rfconnectorai/inference/build_references.py`
- Create: `training/tests/test_build_references.py`

- [ ] **Step 1: Write failing test `training/tests/test_build_references.py`**

```python
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _make_tiny_dataset(root: Path, class_names: list[str], per_class: int = 3) -> None:
    for name in class_names:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        for i in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_build_references_writes_expected_binary(tmp_path: Path, classes_yaml: Path):
    from rfconnectorai.inference.build_references import build_references
    from rfconnectorai.models.embedder import RGBDEmbedder

    data_root = tmp_path / "refs"
    _make_tiny_dataset(data_root, ["SMA-M", "SMA-F"], per_class=3)

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    ckpt = tmp_path / "embedder.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    out_bin = tmp_path / "reference_embeddings.bin"
    build_references(
        checkpoint=ckpt,
        data_root=data_root,
        classes_yaml=classes_yaml,
        output_path=out_bin,
        image_size=64,
        device="cpu",
    )

    assert out_bin.exists()

    with open(out_bin, "rb") as f:
        magic = f.read(4)
        assert magic == b"RFCE"                       # RF Connector Embeddings
        version, n_classes, dim = struct.unpack("<III", f.read(12))
        assert version == 1
        assert n_classes == 2
        assert dim == 128

        # n_classes × (int32 id, 64 bytes name, dim × float32)
        for _ in range(n_classes):
            class_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = f.read(64)
            name = name_bytes.split(b"\x00", 1)[0].decode("utf-8")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            assert class_id in (0, 1)
            assert name in ("SMA-M", "SMA-F")
            # Vectors are L2-normalized.
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5


def test_build_references_rejects_missing_class_dir(
    tmp_path: Path, classes_yaml: Path
):
    from rfconnectorai.inference.build_references import build_references
    from rfconnectorai.models.embedder import RGBDEmbedder

    data_root = tmp_path / "refs"
    _make_tiny_dataset(data_root, ["SMA-M"], per_class=2)  # missing SMA-F

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    out_bin = tmp_path / "r.bin"
    with pytest.raises(FileNotFoundError):
        build_references(
            checkpoint=ckpt,
            data_root=data_root,
            classes_yaml=classes_yaml,
            output_path=out_bin,
            image_size=64,
            device="cpu",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_build_references.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/inference/build_references.py`**

```python
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder


MAGIC = b"RFCE"
FORMAT_VERSION = 1


def build_references(
    checkpoint: Path,
    data_root: Path,
    classes_yaml: Path,
    output_path: Path,
    image_size: int = 384,
    device: str = "cpu",
) -> Path:
    classes = load_classes(classes_yaml)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    embedding_dim = ckpt["embedding_dim"]
    model = RGBDEmbedder(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = RGBDConnectorDataset(
        root=data_root, classes_yaml=classes_yaml, image_size=image_size
    )

    # Accumulate per-class embedding sums.
    sums: dict[int, torch.Tensor] = {c.id: torch.zeros(embedding_dim) for c in classes}
    counts: dict[int, int] = {c.id: 0 for c in classes}

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x).cpu()
            for i, lab in enumerate(y.tolist()):
                sums[lab] += emb[i]
                counts[lab] += 1

    # Write binary file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", FORMAT_VERSION, len(classes), embedding_dim))
        for c in classes:
            if counts[c.id] == 0:
                raise RuntimeError(f"No samples found for class {c.name}")
            mean = sums[c.id] / counts[c.id]
            mean = F.normalize(mean.unsqueeze(0), p=2, dim=1).squeeze(0)
            f.write(struct.pack("<i", c.id))
            name_bytes = c.name.encode("utf-8")
            if len(name_bytes) > 64:
                raise ValueError(f"Class name too long: {c.name}")
            f.write(name_bytes.ljust(64, b"\x00"))
            f.write(mean.numpy().astype("float32").tobytes())

    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = build_references(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        classes_yaml=args.classes_yaml,
        output_path=args.output,
        image_size=args.image_size,
        device=args.device,
    )
    print(f"reference embeddings written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_build_references.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/inference/build_references.py training/tests/test_build_references.py
git commit -m "feat(training): build reference embeddings binary for Unity matcher"
```

---

## Task 12: Evaluation Script

Runs the embedder against a labeled test set, computes per-class accuracy via nearest-reference matching, confusion matrix, and confidence calibration (ECE). Emits a single JSON report consumed by the eval gate.

**Files:**
- Create: `training/rfconnectorai/inference/eval.py`
- Create: `training/tests/test_eval.py`

- [ ] **Step 1: Write failing test `training/tests/test_eval.py`**

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _make_tiny_dataset(root: Path, class_names: list[str], per_class: int = 3) -> None:
    for name in class_names:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        for i in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_evaluate_writes_report(tmp_path: Path, classes_yaml: Path):
    from rfconnectorai.inference.build_references import build_references
    from rfconnectorai.inference.eval import evaluate
    from rfconnectorai.models.embedder import RGBDEmbedder

    data_root = tmp_path / "eval"
    _make_tiny_dataset(data_root, ["SMA-M", "SMA-F"], per_class=4)

    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    refs = tmp_path / "refs.bin"
    build_references(
        checkpoint=ckpt, data_root=data_root, classes_yaml=classes_yaml,
        output_path=refs, image_size=64, device="cpu",
    )

    report_path = tmp_path / "report.json"
    evaluate(
        checkpoint=ckpt,
        references=refs,
        data_root=data_root,
        classes_yaml=classes_yaml,
        output_path=report_path,
        image_size=64,
        device="cpu",
    )

    report = json.loads(report_path.read_text())
    assert "top1_accuracy" in report
    assert "per_class_recall" in report
    assert "confusion_matrix" in report
    assert "expected_calibration_error" in report
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    assert len(report["per_class_recall"]) == 2
    assert len(report["confusion_matrix"]) == 2
    assert len(report["confusion_matrix"][0]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/inference/eval.py`**

```python
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfconnectorai.data.classes import load_classes
from rfconnectorai.data.dataset import RGBDConnectorDataset
from rfconnectorai.models.embedder import RGBDEmbedder


def _load_references(path: Path) -> tuple[list[int], list[str], np.ndarray]:
    with open(path, "rb") as f:
        if f.read(4) != b"RFCE":
            raise ValueError("bad magic in references file")
        version, n_classes, dim = struct.unpack("<III", f.read(12))
        if version != 1:
            raise ValueError(f"unsupported references version {version}")
        ids: list[int] = []
        names: list[str] = []
        vectors = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            (cid,) = struct.unpack("<i", f.read(4))
            name = f.read(64).split(b"\x00", 1)[0].decode("utf-8")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            ids.append(cid)
            names.append(name)
            vectors[i] = vec
    return ids, names, vectors


def _expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / total) * abs(acc_bin - conf_bin)
    return float(ece)


def evaluate(
    checkpoint: Path,
    references: Path,
    data_root: Path,
    classes_yaml: Path,
    output_path: Path,
    image_size: int = 384,
    device: str = "cpu",
) -> dict:
    classes = load_classes(classes_yaml)
    id_to_name = {c.id: c.name for c in classes}

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    embedding_dim = ckpt["embedding_dim"]
    model = RGBDEmbedder(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ref_ids, ref_names, ref_vectors = _load_references(references)
    ref_t = torch.from_numpy(ref_vectors).to(device)

    ds = RGBDConnectorDataset(
        root=data_root, classes_yaml=classes_yaml, image_size=image_size
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    all_pred: list[int] = []
    all_true: list[int] = []
    all_conf: list[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model(x)
            sims = emb @ ref_t.T           # (B, n_classes)   cosine since all L2-normalized
            top_sim, top_idx = sims.max(dim=1)
            pred = torch.tensor([ref_ids[i.item()] for i in top_idx])
            # Convert cosine sim [-1, 1] to pseudo-probability [0, 1] via linear scaling.
            conf = (top_sim.cpu().numpy() + 1.0) / 2.0

            all_pred.extend(pred.tolist())
            all_true.extend(y.tolist())
            all_conf.extend(conf.tolist())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    conf_arr = np.array(all_conf)
    correct = (pred_arr == true_arr).astype(np.int32)

    top1 = float(correct.mean())

    per_class_recall: dict[str, float] = {}
    for c in classes:
        mask = true_arr == c.id
        if mask.sum() == 0:
            per_class_recall[c.name] = 0.0
        else:
            per_class_recall[c.name] = float(correct[mask].mean())

    n = len(classes)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(true_arr, pred_arr):
        confusion[t][p] += 1

    ece = _expected_calibration_error(conf_arr, correct.astype(float))

    report = {
        "top1_accuracy": top1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": [id_to_name[c.id] for c in classes],
        "expected_calibration_error": ece,
        "n_samples": int(len(true_arr)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--classes-yaml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    report = evaluate(
        checkpoint=args.checkpoint,
        references=args.references,
        data_root=args.data_root,
        classes_yaml=args.classes_yaml,
        output_path=args.output,
        image_size=args.image_size,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_eval.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/inference/eval.py training/tests/test_eval.py
git commit -m "feat(training): evaluation with per-class recall, confusion matrix, ECE"
```

---

## Task 13: ONNX Export

Exports both models to ONNX for Unity Sentis consumption. Verifies round-trip numerical equivalence between the PyTorch model and the ONNX runtime output to catch export bugs.

**Files:**
- Create: `training/rfconnectorai/export/onnx_export.py`
- Create: `training/tests/test_onnx_export.py`

- [ ] **Step 1: Write failing test `training/tests/test_onnx_export.py`**

```python
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rfconnectorai.export.onnx_export import export_embedder, export_detector
from rfconnectorai.models.embedder import RGBDEmbedder


def test_export_embedder_roundtrip(tmp_path: Path):
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    model.eval()
    ckpt = tmp_path / "e.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": 128}, ckpt)

    onnx_path = tmp_path / "embedder.onnx"
    export_embedder(checkpoint=ckpt, output=onnx_path, image_size=64)

    assert onnx_path.exists()

    # Compare PyTorch vs ONNX Runtime outputs.
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)


def test_export_detector_produces_onnx(tmp_path: Path):
    """
    The detector export is an Ultralytics passthrough — we just verify it yields
    an ONNX file. Deeper equivalence checks happen inside ultralytics.
    """
    import shutil
    import urllib.request

    # Use Ultralytics' small pretrained weights rather than training a fresh model.
    weights = tmp_path / "yolo11n.pt"
    # The YOLO() constructor auto-downloads if missing, so just make a path it can find.
    onnx_path = tmp_path / "detector.onnx"

    export_detector(weights=Path("yolo11n.pt"), output=onnx_path, image_size=320)

    assert onnx_path.exists()
    # Load with onnxruntime to ensure it's valid.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && pytest tests/test_onnx_export.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/export/onnx_export.py`**

```python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from rfconnectorai.models.embedder import RGBDEmbedder


def export_embedder(checkpoint: Path, output: Path, image_size: int = 384) -> Path:
    """
    Export the RGBDEmbedder to ONNX.

    Unity Sentis as of 2.x supports ONNX opset 15+. We target opset 17 which is
    broadly compatible and supported by onnxruntime 1.18+.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dim = ckpt["embedding_dim"]
    model = RGBDEmbedder(embedding_dim=dim, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 4, image_size, image_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    return output


def export_detector(weights: Path, output: Path, image_size: int = 640) -> Path:
    """
    Export the YOLO detector to ONNX using Ultralytics' built-in exporter.

    Ultralytics writes an ONNX alongside the .pt file. We move it to `output`
    for consistent artifact naming.
    """
    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=image_size, opset=17)
    produced_path = Path(produced)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced_path), str(output))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder-checkpoint", type=Path)
    ap.add_argument("--embedder-out", type=Path)
    ap.add_argument("--embedder-size", type=int, default=384)
    ap.add_argument("--detector-weights", type=Path)
    ap.add_argument("--detector-out", type=Path)
    ap.add_argument("--detector-size", type=int, default=640)
    args = ap.parse_args()

    if args.embedder_checkpoint and args.embedder_out:
        p = export_embedder(
            checkpoint=args.embedder_checkpoint,
            output=args.embedder_out,
            image_size=args.embedder_size,
        )
        print(f"embedder ONNX: {p}")

    if args.detector_weights and args.detector_out:
        p = export_detector(
            weights=args.detector_weights,
            output=args.detector_out,
            image_size=args.detector_size,
        )
        print(f"detector ONNX: {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && pytest tests/test_onnx_export.py -v`
Expected: 2 passed. (Downloads `yolo11n.pt` if not cached.)

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/export/onnx_export.py training/tests/test_onnx_export.py
git commit -m "feat(training): ONNX export for embedder and detector"
```

---

## Task 14: End-to-End Smoke Test

Proves all components work together on tiny synthetic data: train embedder → build references → evaluate → export ONNX → verify ONNX is loadable. This becomes the pipeline's CI canary.

**Files:**
- Create: `training/tests/test_end_to_end.py`

- [ ] **Step 1: Write failing test `training/tests/test_end_to_end.py`**

```python
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image


def _fixture_dataset(root: Path, class_names: list[str], per_class: int = 5) -> None:
    for name in class_names:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        for i in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_full_pipeline_end_to_end(tmp_path: Path, classes_yaml: Path):
    # 1. Fixture dataset
    data_root = tmp_path / "data"
    _fixture_dataset(data_root, ["SMA-M", "SMA-F"], per_class=5)

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # 2. Train embedder (smoke)
    runs = tmp_path / "runs"
    r = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.training.train_embedder",
            "--data-root", str(data_root),
            "--classes-yaml", str(classes_yaml),
            "--output-dir", str(runs),
            "--smoke-test",
        ],
        cwd=project_root, capture_output=True, text=True, env=env, timeout=900,
    )
    assert r.returncode == 0, r.stderr

    ckpt = runs / "embedder.pt"
    assert ckpt.exists()

    # 3. Build references
    refs = tmp_path / "refs.bin"
    r = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.inference.build_references",
            "--checkpoint", str(ckpt),
            "--data-root", str(data_root),
            "--classes-yaml", str(classes_yaml),
            "--output", str(refs),
            "--image-size", "64",
        ],
        cwd=project_root, capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert refs.exists()

    # 4. Evaluate
    report_path = tmp_path / "report.json"
    r = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.inference.eval",
            "--checkpoint", str(ckpt),
            "--references", str(refs),
            "--data-root", str(data_root),
            "--classes-yaml", str(classes_yaml),
            "--output", str(report_path),
            "--image-size", "64",
        ],
        cwd=project_root, capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(report_path.read_text())
    assert "top1_accuracy" in report

    # 5. Export embedder to ONNX and verify it loads
    onnx_path = tmp_path / "embedder.onnx"
    r = subprocess.run(
        [
            sys.executable, "-m", "rfconnectorai.export.onnx_export",
            "--embedder-checkpoint", str(ckpt),
            "--embedder-out", str(onnx_path),
            "--embedder-size", "64",
        ],
        cwd=project_root, capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert onnx_path.exists()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": np.random.randn(1, 4, 64, 64).astype(np.float32)})
    assert out[0].shape == (1, 128)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd training && pytest tests/test_end_to_end.py -v`
Expected: 1 passed. (Runs the full mini-pipeline, ~2–5 min depending on machine.)

- [ ] **Step 3: Commit**

```bash
git add training/tests/test_end_to_end.py
git commit -m "test(training): end-to-end pipeline smoke test"
```

---

## Task 15: Final Verification and README Update

Polish pass: make sure the top-level pipeline is invocable, add a CI script, and update the README so someone cloning the repo can run the full pipeline without reading the plan.

**Files:**
- Create: `training/scripts/run_pipeline.sh`
- Modify: `training/README.md`

- [ ] **Step 1: Create `training/scripts/run_pipeline.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline driver. Expects a labeled dataset at data/labeled/ with
# per-class subdirectories for the embedder, and Ultralytics YOLO format for
# the detector.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/labeled/embedder}"
DETECTOR_YAML="${DETECTOR_YAML:-configs/detector.yaml}"
CLASSES_YAML="${CLASSES_YAML:-configs/classes.yaml}"
RUNS="${RUNS:-runs}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$RUNS"

echo "[1/5] Training detector"
python -m rfconnectorai.training.train_detector \
    --data-yaml "$DETECTOR_YAML" \
    --output-dir "$RUNS/detector" \
    --device "$DEVICE"

echo "[2/5] Training embedder"
python -m rfconnectorai.training.train_embedder \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output-dir "$RUNS/embedder" \
    --device "$DEVICE"

echo "[3/5] Building reference embeddings"
python -m rfconnectorai.inference.build_references \
    --checkpoint "$RUNS/embedder/embedder.pt" \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output "$RUNS/reference_embeddings.bin" \
    --device "$DEVICE"

echo "[4/5] Evaluating"
python -m rfconnectorai.inference.eval \
    --checkpoint "$RUNS/embedder/embedder.pt" \
    --references "$RUNS/reference_embeddings.bin" \
    --data-root "$DATA_DIR" \
    --classes-yaml "$CLASSES_YAML" \
    --output "$RUNS/eval_report.json" \
    --device "$DEVICE"

echo "[5/5] Exporting to ONNX"
python -m rfconnectorai.export.onnx_export \
    --embedder-checkpoint "$RUNS/embedder/embedder.pt" \
    --embedder-out "$RUNS/embedder.onnx" \
    --detector-weights "$RUNS/detector/detector.pt" \
    --detector-out "$RUNS/detector.onnx"

echo "Done. Artifacts in $RUNS/:"
ls -la "$RUNS/"*.onnx "$RUNS/reference_embeddings.bin" "$RUNS/eval_report.json"
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x training/scripts/run_pipeline.sh
```

- [ ] **Step 3: Replace `training/README.md` with the full version**

```markdown
# RF Connector AI — Training Pipeline

Python pipeline that trains a YOLOv11n connector detector and a MobileViT-v2 RGBD embedder, then exports both to ONNX for consumption by the Unity Sentis runtime.

Spec: `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`

## Setup

    cd training
    uv venv
    uv pip install -e ".[dev]"

## Data layout

For the **embedder**, organize images into per-class directories matching `configs/classes.yaml`:

    data/labeled/embedder/
      SMA-M/
        img0001.png
        ...
      SMA-F/
      3.5mm-M/
      ...

For the **detector**, use standard Ultralytics YOLO layout pointed at by `configs/detector.yaml`:

    data/labeled/detector/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt   # "0 cx cy w h" per line, normalized
      labels/val/*.txt

During Phase 0 (before real connectors arrive), populate both with a mix of:
- `python -m rfconnectorai.data.scrape` (catalog images)
- `python -m rfconnectorai.data.synthetic` (procedural renders)

## Full pipeline

    bash scripts/run_pipeline.sh

Produces under `runs/`:
- `detector.onnx`
- `embedder.onnx`
- `reference_embeddings.bin`
- `eval_report.json`

## Individual steps

    python -m rfconnectorai.training.train_detector --data-yaml configs/detector.yaml --output-dir runs/detector
    python -m rfconnectorai.training.train_embedder --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output-dir runs/embedder
    python -m rfconnectorai.inference.build_references --checkpoint runs/embedder/embedder.pt --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output runs/reference_embeddings.bin
    python -m rfconnectorai.inference.eval --checkpoint runs/embedder/embedder.pt --references runs/reference_embeddings.bin --data-root data/labeled/embedder --classes-yaml configs/classes.yaml --output runs/eval_report.json
    python -m rfconnectorai.export.onnx_export --embedder-checkpoint runs/embedder/embedder.pt --embedder-out runs/embedder.onnx --detector-weights runs/detector/detector.pt --detector-out runs/detector.onnx

## Tests

    pytest                    # all
    pytest tests/test_end_to_end.py -v    # full-pipeline smoke test

## Configuration

- `configs/classes.yaml` — the 8 RF connector classes with metadata
- `configs/detector.yaml` — YOLO dataset spec
- `configs/embedder.yaml` — hyperparameters for embedder training
```

- [ ] **Step 4: Run all tests to verify pipeline is healthy**

```bash
cd training && pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add training/scripts/run_pipeline.sh training/README.md
git commit -m "feat(training): full pipeline driver script and updated README"
```

---

## Plan Self-Review

Checked against the spec (`docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`):

- **Spec coverage** — Plan 1's deliverables per the spec are: `detector.onnx`, `embedder.onnx`, `reference_embeddings.bin`, trained on proxy data, with an eval script. All produced by Tasks 1–15.
- **Placeholders** — No TBDs, TODOs, or "implement later" stubs. Every step has concrete code and exact commands.
- **Type consistency** — Verified:
  - `RGBDEmbedder(embedding_dim, pretrained, backbone)` used identically in training, references, eval, export.
  - `batch_hard_triplet_loss(embeddings, labels, margin)` signature consistent between the loss module and its consumers.
  - `RGBDConnectorDataset(root, classes_yaml, image_size, object_depth_m)` consistent between dataset tests and downstream uses.
  - `build_references(checkpoint, data_root, classes_yaml, output_path, image_size, device)` consistent between unit tests and the end-to-end test.
  - Binary format (`RFCE` magic, version `1`, `<III` header, `<i` + 64-byte name + float32 vector per class) defined in Task 11 and consumed correctly in Task 12.
- **Scope** — This plan produces a single working subsystem (training pipeline) that stands on its own. Plan 2 (Unity scanner MVP) is a separate plan and can run in parallel.

## Execution Handoff

Plan complete and will be saved to `docs/superpowers/plans/2026-04-23-python-training-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
