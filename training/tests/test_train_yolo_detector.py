from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfconnectorai.detector.train_yolo import (
    SUPPORTED_MODELS,
    TrainerConfig,
    emit_run_metadata,
    main,
    parse_args,
    validate_config,
)


def _write_data_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "data.yaml"
    p.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  - SMA\n",
        encoding="utf-8",
    )
    return p


def _write_dataset_lock(tmp_path: Path) -> Path:
    p = tmp_path / "dataset.lock.json"
    p.write_text(
        json.dumps({
            "dataset_id": "rfconnectors_2026_05_10_001",
            "taxonomy_sha256": "abcdef0123456789",
        }),
        encoding="utf-8",
    )
    return p


def test_parse_args_minimal(tmp_path: Path):
    data = _write_data_yaml(tmp_path)
    args = parse_args([
        "--data", str(data),
        "--out", str(tmp_path / "out"),
        "--artifact-out", str(tmp_path / "artifacts"),
    ])
    assert args.data == data
    assert args.epochs == 100  # default
    assert args.dry_run is False


def test_validate_config_rejects_unknown_model(tmp_path: Path):
    cfg = TrainerConfig(
        data=_write_data_yaml(tmp_path),
        model="totally-fake.pt",
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
        out=tmp_path / "out",
        artifact_out=tmp_path / "art",
    )
    with pytest.raises(ValueError, match="not in supported"):
        validate_config(cfg)


def test_validate_config_rejects_missing_data(tmp_path: Path):
    cfg = TrainerConfig(
        data=tmp_path / "missing.yaml",
        model="yolo11n.pt",
        epochs=1, imgsz=64, batch=1, device="cpu",
        out=tmp_path / "out", artifact_out=tmp_path / "art",
    )
    with pytest.raises(FileNotFoundError):
        validate_config(cfg)


def test_validate_config_requires_positive_epochs_imgsz_batch(tmp_path: Path):
    data = _write_data_yaml(tmp_path)
    for kwargs in (
        {"epochs": 0},
        {"imgsz": 0},
        {"batch": 0},
    ):
        cfg = TrainerConfig(
            data=data, model="yolo11n.pt", epochs=1, imgsz=32, batch=1,
            device="cpu", out=tmp_path / "out", artifact_out=tmp_path / "art",
            **kwargs,
        )
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_emit_run_metadata_writes_config_and_record(tmp_path: Path):
    data = _write_data_yaml(tmp_path)
    lock = _write_dataset_lock(tmp_path)
    cfg = TrainerConfig(
        data=data, model="yolo11n.pt", epochs=2, imgsz=64, batch=2, device="cpu",
        out=tmp_path / "run", artifact_out=tmp_path / "art",
        dataset_lock=lock, seq=3,
    )
    info = emit_run_metadata(cfg)
    assert (cfg.out / "model_record.json").exists()
    assert (cfg.out / "config.json").exists()
    record = json.loads((cfg.out / "model_record.json").read_text(encoding="utf-8"))
    assert record["model_type"] == "detector"
    assert record["architecture"] == "yolo11n"
    assert "rfconnectors_2026_05_10_001" in record["trained_on"]
    assert record["taxonomy_version"] == "abcdef0123456789"
    assert record["extra"]["epochs"] == 2
    assert info["config"]["seq"] == 3


def test_main_dry_run_does_not_train(tmp_path: Path):
    data = _write_data_yaml(tmp_path)
    rc = main([
        "--data", str(data),
        "--model", "yolo11n.pt",
        "--epochs", "1",
        "--imgsz", "32",
        "--batch", "1",
        "--device", "cpu",
        "--out", str(tmp_path / "run"),
        "--artifact-out", str(tmp_path / "art"),
        "--dry-run",
    ])
    assert rc == 0
    assert (tmp_path / "run" / "model_record.json").exists()


def test_supported_models_set_is_explicit():
    assert "yolo11n.pt" in SUPPORTED_MODELS
    assert all(m.endswith(".pt") for m in SUPPORTED_MODELS)
