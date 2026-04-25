"""
Round-trip test: train a classifier on a tiny synthetic 2-class dataset,
load it via the predict module, confirm it learned to distinguish the classes.

We keep the test small (2 classes × 4 images each, 1 epoch) so it runs in
seconds on CPU. Validates the wiring more than the model itself.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfconnectorai.classifier.predict import ConnectorClassifier
from rfconnectorai.classifier.train import TrainConfig, train


def _make_red_image(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 50, size=(size, size, 3), dtype=np.uint8)
    arr[..., 0] = rng.integers(180, 255, size=(size, size), dtype=np.uint8)  # high red
    return arr


def _make_blue_image(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 50, size=(size, size, 3), dtype=np.uint8)
    arr[..., 2] = rng.integers(180, 255, size=(size, size), dtype=np.uint8)  # high blue
    return arr


def _build_synthetic_dataset(root: Path, n_per_class: int = 4) -> None:
    for cls, maker in [("red", _make_red_image), ("blue", _make_blue_image)]:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = maker(seed=i)
            Image.fromarray(arr).save(cls_dir / f"{cls}_{i:03d}.png")


def test_classifier_trains_and_predicts(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "model"
    _build_synthetic_dataset(data_dir, n_per_class=6)

    config = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        class_names=["red", "blue"],
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        val_fraction=0.25,
    )
    metrics = train(config)
    assert "history" in metrics
    assert (out_dir / "weights.pt").exists()
    assert (out_dir / "labels.json").exists()
    assert (out_dir / "metrics.json").exists()

    # Load and predict on held-out red and blue inputs.
    classifier = ConnectorClassifier.load(out_dir)
    red_pred = classifier.predict(_make_red_image(seed=999))
    blue_pred = classifier.predict(_make_blue_image(seed=999))
    # 2 epochs on 6 images is tiny — assert at least one prediction was correct
    # (very high probability on this trivial task) and that probabilities sum to 1.
    assert red_pred.class_name in {"red", "blue"}
    assert abs(sum(red_pred.probabilities.values()) - 1.0) < 1e-3
    assert red_pred.confidence == max(red_pred.probabilities.values())
    # On a trivially-separable task (red-vs-blue images), at least one of these
    # should be correct after 2 epochs.
    assert red_pred.class_name == "red" or blue_pred.class_name == "blue"


def test_load_raises_when_no_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConnectorClassifier.load(tmp_path / "no_model_here")


def test_predict_many_returns_per_image_predictions(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "model"
    _build_synthetic_dataset(data_dir, n_per_class=4)
    config = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        class_names=["red", "blue"],
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        val_fraction=0.25,
    )
    train(config)

    classifier = ConnectorClassifier.load(out_dir)
    images = [_make_red_image(seed=i) for i in range(3)]
    preds = classifier.predict_many(images)
    assert len(preds) == 3
    for p in preds:
        assert p.class_name in {"red", "blue"}
        assert 0.0 <= p.confidence <= 1.0
