"""
Tests for model versioning + manifest writes.
"""
import json
from pathlib import Path

import pytest

from rfconnectorai.classifier.versioning import (
    LABELS_FILENAME,
    LATEST_WEIGHTS,
    MANIFEST_FILENAME,
    VERSION_FILENAME,
    bump_version,
    current_version,
    read_manifest,
)


def _write_fake_weights(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "weights.pt"
    weights_path.write_bytes(b"fake-weights-bytes" * 100)
    (model_dir / LABELS_FILENAME).write_text(json.dumps({
        "class_names": ["SMA-M", "SMA-F"], "input_size": 224,
    }))
    return weights_path


def test_current_version_zero_when_no_manifest(tmp_path):
    assert current_version(tmp_path) == 0


def test_first_bump_returns_one_and_writes_artifacts(tmp_path):
    _write_fake_weights(tmp_path)
    new_version = bump_version(tmp_path)
    assert new_version == 1
    # versioned snapshot, latest pointer, version.json, manifest.json all exist.
    assert (tmp_path / "weights.0001.pt").exists()
    assert (tmp_path / LATEST_WEIGHTS).exists()
    assert (tmp_path / VERSION_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_subsequent_bumps_increment(tmp_path):
    _write_fake_weights(tmp_path)
    assert bump_version(tmp_path) == 1
    # Re-write the source weights as if a second training run happened.
    (tmp_path / "weights.pt").write_bytes(b"second-set-of-weights" * 100)
    assert bump_version(tmp_path) == 2
    assert (tmp_path / "weights.0002.pt").exists()
    # Version.json now reads 2.
    assert current_version(tmp_path) == 2


def test_manifest_has_sha256_of_weights(tmp_path):
    _write_fake_weights(tmp_path)
    bump_version(tmp_path)
    manifest = read_manifest(tmp_path)
    assert "weights_sha256" in manifest
    assert len(manifest["weights_sha256"]) == 64
    assert manifest["weights_filename"] == "weights.0001.pt"
    assert manifest["labels_filename"] == LABELS_FILENAME


def test_latest_pointer_matches_newest_version(tmp_path):
    _write_fake_weights(tmp_path)
    bump_version(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"new-content-for-v2" * 100)
    bump_version(tmp_path)
    latest = (tmp_path / LATEST_WEIGHTS).read_bytes()
    v2 = (tmp_path / "weights.0002.pt").read_bytes()
    assert latest == v2


def test_read_manifest_raises_when_unversioned(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path)


def test_bump_with_metrics_recorded_in_version_json(tmp_path):
    _write_fake_weights(tmp_path)
    bump_version(tmp_path, val_acc=0.83, n_train_samples=120)
    blob = json.loads((tmp_path / VERSION_FILENAME).read_text())
    assert blob["val_acc"] == 0.83
    assert blob["n_train_samples"] == 120
