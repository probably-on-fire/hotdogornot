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
