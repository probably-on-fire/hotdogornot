from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rfconnectorai.classifier.train_multihead import (
    MultiHeadTrainerConfig,
    emit_run_metadata,
    head_sizes_for_dataset,
    main,
    parse_args,
    validate_config,
)


def _write_attributes_csv(path: Path) -> None:
    fields = [
        "instance_id", "split", "source_image", "family", "precision_family",
        "side_a_gender", "side_b_gender", "polarity", "mount_style",
        "orientation", "termination", "finish_material_cue",
        "label_confidence", "source_type",
    ]
    rows = [
        ["i_001", "train", "Images/SMA-M/001.jpg", "SMA", "standard_sma",
         "male_pin", "not_applicable", "standard", "cable_mount",
         "straight", "solder", "gold", "human_verified", "real_photo"],
        ["i_002", "val", "Images/SMA-F/001.jpg", "SMA", "standard_sma",
         "female_socket", "not_applicable", "standard", "cable_mount",
         "straight", "crimp", "nickel_silver", "human_verified", "real_photo"],
        ["i_003", "test", "Images/BNC/001.jpg", "BNC", "not_applicable",
         "female_socket", "not_applicable", "not_applicable", "cable_mount",
         "straight", "crimp", "unknown", "human_verified", "real_photo"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


def test_parse_args_defaults(tmp_path: Path):
    args = parse_args([
        "--dataset", str(tmp_path),
        "--out", str(tmp_path / "run"),
        "--artifact-out", str(tmp_path / "art"),
    ])
    assert args.backbone == "efficientnet_v2_s"
    assert args.epochs == 80


def test_validate_config_rejects_unknown_backbone(tmp_path: Path):
    cfg = MultiHeadTrainerConfig(
        dataset=tmp_path, backbone="nonexistent_net",
        epochs=1, batch=1, imgsz=32, device="cpu",
        out=tmp_path / "run", artifact_out=tmp_path / "art",
    )
    with pytest.raises(ValueError, match="not in supported"):
        validate_config(cfg)


def test_validate_config_rejects_zero_or_negative(tmp_path: Path):
    for kwargs in ({"epochs": 0}, {"batch": 0}, {"imgsz": 0}):
        cfg = MultiHeadTrainerConfig(
            dataset=tmp_path, backbone="resnet18",
            epochs=1, batch=1, imgsz=32, device="cpu",
            out=tmp_path / "r", artifact_out=tmp_path / "a",
            **kwargs,
        )
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_head_sizes_for_dataset_includes_all_heads(tmp_path: Path):
    _write_attributes_csv(tmp_path / "attributes.csv")
    sizes, vocabs = head_sizes_for_dataset(tmp_path)
    assert "family" in sizes
    assert sizes["family"] == vocabs["family"].num_classes
    assert sizes["polarity"] == vocabs["polarity"].num_classes


def test_head_sizes_for_dataset_requires_attributes_csv(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        head_sizes_for_dataset(tmp_path)


def test_emit_run_metadata_writes_record_and_vocabs(tmp_path: Path):
    _write_attributes_csv(tmp_path / "attributes.csv")
    (tmp_path / "dataset.lock.json").write_text(
        json.dumps({"dataset_id": "rfconnectors_2026_05_10_001", "taxonomy_sha256": "deadbeef"}),
        encoding="utf-8",
    )
    cfg = MultiHeadTrainerConfig(
        dataset=tmp_path, backbone="resnet18",
        epochs=1, batch=2, imgsz=32, device="cpu",
        out=tmp_path / "run", artifact_out=tmp_path / "art",
    )
    info = emit_run_metadata(cfg)
    assert (cfg.out / "model_record.json").exists()
    assert (cfg.out / "config.json").exists()
    assert (cfg.out / "head_vocabs.json").exists()
    record = json.loads((cfg.out / "model_record.json").read_text(encoding="utf-8"))
    assert record["model_type"] == "multihead_classifier"
    assert "rfconnectors_2026_05_10_001" in record["trained_on"]
    assert record["taxonomy_version"] == "deadbeef"
    head_vocabs = json.loads((cfg.out / "head_vocabs.json").read_text(encoding="utf-8"))
    assert "family" in head_vocabs


def test_main_dry_run(tmp_path: Path):
    _write_attributes_csv(tmp_path / "attributes.csv")
    rc = main([
        "--dataset", str(tmp_path),
        "--backbone", "resnet18",
        "--epochs", "1",
        "--batch", "1",
        "--imgsz", "32",
        "--device", "cpu",
        "--out", str(tmp_path / "run"),
        "--artifact-out", str(tmp_path / "art"),
        "--dry-run",
    ])
    assert rc == 0
    assert (tmp_path / "run" / "model_record.json").exists()
