"""
Tests for the auto-retrain decision logic. Stubs the actual train() call
so we don't run real ResNet training — only verifying the "should we
retrain" branch handling.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import auto_retrain


def _seed_dataset(data_root: Path, n_per_class: int = 5) -> None:
    """Drop n images per class so ConnectorFolderDataset finds them."""
    for cls in auto_retrain.CANONICAL_CLASSES:
        cls_dir = data_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(
                np.full((64, 64, 3), [i * 10, i * 10, i * 10], dtype=np.uint8),
            )
            img.save(cls_dir / f"img_{i:03d}.png")


def test_skips_when_no_new_data(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    model = tmp_path / "model"
    model.mkdir(parents=True)
    _seed_dataset(data, n_per_class=5)
    # version.json claims last train used same number of samples → delta = 0.
    (model / "version.json").write_text(json.dumps({"n_train_samples": 5 * 8}))

    argv = [
        "auto_retrain",
        "--data-dir", str(data),
        "--model-dir", str(model),
        "--min-new-samples", "10",
    ]
    with patch.object(sys, "argv", argv), patch.object(auto_retrain, "train") as mock_train:
        rc = auto_retrain.main()
    assert rc == 0
    mock_train.assert_not_called()


def test_runs_when_enough_new_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    model = tmp_path / "model"
    model.mkdir(parents=True)
    _seed_dataset(data, n_per_class=10)
    # Last train at 5 per class → delta = (10 - 5) * 8 = 40 new samples.
    (model / "version.json").write_text(json.dumps({"n_train_samples": 5 * 8}))

    argv = [
        "auto_retrain",
        "--data-dir", str(data),
        "--model-dir", str(model),
        "--min-new-samples", "20",
    ]
    with patch.object(sys, "argv", argv), patch.object(auto_retrain, "train") as mock_train:
        mock_train.return_value = {"history": [{"val_acc": 0.7}]}
        rc = auto_retrain.main()
    assert rc == 0
    mock_train.assert_called_once()


def test_first_run_no_prior_model(tmp_path):
    """No version.json yet — last_size returns None → 0; full dataset is delta."""
    data = tmp_path / "data"
    model = tmp_path / "model"
    _seed_dataset(data, n_per_class=4)

    argv = [
        "auto_retrain",
        "--data-dir", str(data),
        "--model-dir", str(model),
        "--min-new-samples", "20",
        # Test seeds 4 per class; relax the per-class threshold so the
        # test still exercises the no-prior-model code path. Production
        # default is MIN_SAMPLES_PER_CLASS=5 which would drop these.
        "--min-samples-per-class", "1",
    ]
    with patch.object(sys, "argv", argv), patch.object(auto_retrain, "train") as mock_train:
        mock_train.return_value = {"history": [{"val_acc": 0.5}]}
        rc = auto_retrain.main()
    assert rc == 0
    mock_train.assert_called_once()


def test_force_flag_overrides_threshold(tmp_path):
    data = tmp_path / "data"
    model = tmp_path / "model"
    model.mkdir(parents=True)
    _seed_dataset(data, n_per_class=5)
    (model / "version.json").write_text(json.dumps({"n_train_samples": 5 * 8}))   # delta=0

    argv = [
        "auto_retrain",
        "--data-dir", str(data),
        "--model-dir", str(model),
        "--min-new-samples", "1000",   # would normally skip
        "--force",
    ]
    with patch.object(sys, "argv", argv), patch.object(auto_retrain, "train") as mock_train:
        mock_train.return_value = {"history": [{"val_acc": 0.5}]}
        rc = auto_retrain.main()
    assert rc == 0
    mock_train.assert_called_once()


def test_returns_2_when_data_dir_missing(tmp_path):
    argv = [
        "auto_retrain",
        "--data-dir", str(tmp_path / "nonexistent"),
        "--model-dir", str(tmp_path / "model"),
    ]
    with patch.object(sys, "argv", argv):
        rc = auto_retrain.main()
    assert rc == 2


def test_returns_2_when_dataset_too_small(tmp_path):
    data = tmp_path / "data"
    model = tmp_path / "model"
    # Only 1 image per class = 8 total, below the 16-sample minimum.
    _seed_dataset(data, n_per_class=1)

    argv = [
        "auto_retrain",
        "--data-dir", str(data),
        "--model-dir", str(model),
        "--force",
    ]
    with patch.object(sys, "argv", argv):
        rc = auto_retrain.main()
    assert rc == 2
