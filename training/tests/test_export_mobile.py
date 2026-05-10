from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfconnectorai.export.export_mobile import (
    ExportTarget,
    SUPPORTED_FORMATS,
    main,
    parse_target,
    plan_exports,
    validate_targets,
    write_manifest,
)
from rfconnectorai.models.registry import make_model_record, write_record


def _write_record(path: Path, *, model_type: str = "detector", architecture: str = "yolo11n") -> None:
    record = make_model_record(
        model_type=model_type,
        architecture=architecture,
        trained_on="datasets/rfconnectors@abc",
        taxonomy_version="taxonomy_sha",
        metrics_path="reports/metrics.json",
        artifact_path=str(path.with_suffix(".pt")),
        seq=1,
    )
    write_record(record, path)


def test_parse_target_round_trip():
    spec = "detector:models/detector/best.pt:reports/det/model_record.json:onnx,tflite"
    target = parse_target(spec)
    assert target.name == "detector"
    assert target.formats == ("onnx", "tflite")


def test_parse_target_rejects_bad_form():
    with pytest.raises(Exception):
        parse_target("just_one_thing")


def test_validate_targets_rejects_unknown_format(tmp_path: Path):
    record = tmp_path / "rec.json"
    _write_record(record)
    target = ExportTarget(
        name="detector",
        artifact=tmp_path / "best.pt",
        record=record,
        formats=("onnx", "fake_format"),
    )
    with pytest.raises(ValueError, match="unsupported"):
        validate_targets([target])


def test_validate_targets_rejects_missing_record(tmp_path: Path):
    target = ExportTarget(
        name="detector",
        artifact=tmp_path / "best.pt",
        record=tmp_path / "missing.json",
        formats=("onnx",),
    )
    with pytest.raises(FileNotFoundError):
        validate_targets([target])


def test_plan_exports_creates_entry_per_format(tmp_path: Path):
    record_path = tmp_path / "rec.json"
    _write_record(record_path, model_type="multihead_classifier", architecture="efficientnet_v2_s")
    target = ExportTarget(
        name="classifier",
        artifact=tmp_path / "best.pt",
        record=record_path,
        formats=("onnx", "tflite"),
    )
    manifest = plan_exports([target], out_dir=tmp_path / "out")
    assert len(manifest.entries) == 2
    formats = {e["format"] for e in manifest.entries}
    assert formats == {"onnx", "tflite"}
    for entry in manifest.entries:
        assert entry["model_id"]
        assert entry["taxonomy_version"] == "taxonomy_sha"


def test_main_dry_run_writes_manifest(tmp_path: Path):
    record_path = tmp_path / "rec.json"
    _write_record(record_path)
    rc = main([
        "--target",
        f"detector:{tmp_path / 'best.pt'}:{record_path}:onnx,coreml",
        "--out", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 0
    manifest = tmp_path / "out" / "exports_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2


def test_supported_formats_canonical_set():
    assert SUPPORTED_FORMATS == ("onnx", "tflite", "coreml")
