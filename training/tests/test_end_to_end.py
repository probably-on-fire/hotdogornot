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
