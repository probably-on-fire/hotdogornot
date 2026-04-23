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
