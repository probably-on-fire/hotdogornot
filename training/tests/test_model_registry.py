from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rfconnectorai.models.registry import (
    ModelRecord,
    VALID_MODEL_TYPES,
    make_model_record,
    read_record,
    write_record,
)


def test_make_model_record_id_format():
    record = make_model_record(
        model_type="detector",
        architecture="yolo11n",
        trained_on="datasets/rfconnectors@abc123",
        taxonomy_version="taxonomy_sha",
        metrics_path="reports/metrics.json",
        artifact_path="models/detector/best.pt",
        seq=1,
        now=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )
    assert record.model_id == "detector_yolo11n_2026-05-10_001"
    assert record.created_at.startswith("2026-05-10")


def test_make_model_record_rejects_unknown_type():
    with pytest.raises(ValueError, match="model_type"):
        make_model_record(
            model_type="oracle",
            architecture="anything",
            trained_on="x",
            taxonomy_version="x",
            metrics_path="x",
            artifact_path="x",
        )


def test_make_model_record_rejects_bad_seq():
    with pytest.raises(ValueError, match="seq"):
        make_model_record(
            model_type="detector",
            architecture="yolo11n",
            trained_on="x",
            taxonomy_version="x",
            metrics_path="x",
            artifact_path="x",
            seq=1000,
        )


def test_record_round_trip(tmp_path: Path):
    record = make_model_record(
        model_type="multihead_classifier",
        architecture="efficientnet_v2_s",
        trained_on="datasets/rfconnectors@deadbeef",
        taxonomy_version="taxonomy_cafe",
        metrics_path="reports/metrics.json",
        artifact_path="models/multihead/best.pt",
        seq=7,
        now=datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc),
        extra={"backbone_pretrained": True, "input_size": 384},
    )

    out_path = tmp_path / "model_record.json"
    write_record(record, out_path)

    loaded = read_record(out_path)
    assert loaded.model_id == record.model_id
    assert loaded.architecture == "efficientnet_v2_s"
    assert loaded.trained_on == "datasets/rfconnectors@deadbeef"
    assert loaded.extra["backbone_pretrained"] is True
    assert loaded.extra["input_size"] == 384


def test_valid_model_types_includes_core_roles():
    assert "detector" in VALID_MODEL_TYPES
    assert "classifier" in VALID_MODEL_TYPES
    assert "multihead_classifier" in VALID_MODEL_TYPES
    assert "embedder" in VALID_MODEL_TYPES


def test_record_dict_is_stable():
    record = ModelRecord(
        model_id="detector_yolo11n_2026-05-10_001",
        model_type="detector",
        architecture="yolo11n",
        trained_on="datasets/rfconnectors@abc",
        taxonomy_version="abc",
        metrics_path="metrics.json",
        artifact_path="best.pt",
        created_at="2026-05-10T00:00:00+00:00",
    )
    payload = record.to_dict()
    assert payload["model_id"] == "detector_yolo11n_2026-05-10_001"
    assert payload["extra"] == {}
