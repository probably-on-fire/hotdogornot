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
