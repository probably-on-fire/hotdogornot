from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfconnectorai.schemas.instance import (
    ConnectorInstance,
    ConnectorSide,
    GeometryLabel,
    LabelConfidence,
    SourceType,
    instance_from_dict,
    validate_instance,
    VALID_GENDERS,
    VALID_POLARITIES,
)


def _minimal_instance(**overrides) -> ConnectorInstance:
    base = dict(
        instance_id="test_001",
        source_image="training/Images/example.webp",
        crop_path="datasets/rfconnectors/crops/example_0001.jpg",
        bbox_xyxy=(120, 80, 420, 360),
        label_confidence=LabelConfidence.HUMAN_VERIFIED,
        source_type=SourceType.REAL_PHOTO,
        family="SMA",
    )
    base.update(overrides)
    return ConnectorInstance(**base)


def test_minimal_instance_validates():
    inst = _minimal_instance()
    validate_instance(inst)
    out = inst.to_dict()
    assert out["instance_id"] == "test_001"
    assert out["bbox_xyxy"] == [120, 80, 420, 360]
    assert out["label_confidence"] == "human_verified"
    assert out["source_type"] == "real_photo"
    assert out["side_a"] is None
    assert out["side_b"] is None


def test_adapter_requires_side_b():
    bad = _minimal_instance(mount_style="adapter")
    with pytest.raises(ValueError, match="adapter"):
        validate_instance(bad)

    good = _minimal_instance(
        mount_style="adapter",
        side_a=ConnectorSide(family="SMA", gender="male_pin", polarity="standard"),
        side_b=ConnectorSide(
            family="BNC", gender="female_socket", polarity="not_applicable", coupling="bayonet"
        ),
    )
    validate_instance(good)


def test_invalid_gender_rejected():
    bad = _minimal_instance(side_a_gender="alien_contact")
    with pytest.raises(ValueError, match="VALID_GENDERS"):
        validate_instance(bad)


def test_invalid_polarity_rejected():
    bad = _minimal_instance(polarity="diagonal")
    with pytest.raises(ValueError, match="VALID_POLARITIES"):
        validate_instance(bad)


def test_bbox_must_be_positive_extent():
    bad = _minimal_instance(bbox_xyxy=(100, 100, 50, 50))
    with pytest.raises(ValueError, match="bbox_xyxy"):
        validate_instance(bad)


def test_geometry_defaults_require_calibration():
    inst = _minimal_instance()
    assert inst.geometry.requires_calibrated_reference is True
    assert inst.geometry.thread_diameter_mm is None


def test_unknown_and_insufficient_view_are_valid_genders():
    assert "unknown" in VALID_GENDERS
    assert "insufficient_view" in VALID_GENDERS
    assert "not_applicable" in VALID_GENDERS


def test_unknown_and_not_applicable_are_valid_polarities():
    assert "unknown" in VALID_POLARITIES
    assert "not_applicable" in VALID_POLARITIES
    assert "insufficient_view" in VALID_POLARITIES


def test_to_dict_round_trips_through_instance_from_dict(tmp_path: Path):
    inst = _minimal_instance(
        mount_style="adapter",
        side_a=ConnectorSide(
            family="SMA", precision_family="standard_sma", gender="male_pin",
            polarity="standard", threaded=True,
        ),
        side_b=ConnectorSide(
            family="BNC", precision_family="not_applicable", gender="female_socket",
            polarity="not_applicable", coupling="bayonet",
        ),
        geometry=GeometryLabel(
            thread_diameter_mm=4.0, body_length_mm=20.0,
            requires_calibrated_reference=False,
        ),
    )

    payload = inst.to_dict()
    out_path = tmp_path / "instances.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    with open(out_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    parsed = instance_from_dict(json.loads(line))

    assert parsed.instance_id == inst.instance_id
    assert parsed.bbox_xyxy == inst.bbox_xyxy
    assert parsed.label_confidence == LabelConfidence.HUMAN_VERIFIED
    assert parsed.side_a is not None
    assert parsed.side_a.family == "SMA"
    assert parsed.side_b is not None
    assert parsed.side_b.coupling == "bayonet"
    assert parsed.geometry.requires_calibrated_reference is False


def test_label_confidence_enum_values_match_protocol():
    expected = {
        "human_verified",
        "weak_folder_label",
        "synthetic_verified",
        "model_suggested",
        "unknown",
    }
    assert {member.value for member in LabelConfidence} == expected


def test_source_type_enum_includes_real_and_synthetic():
    values = {member.value for member in SourceType}
    assert "real_photo" in values
    assert "synthetic_render" in values
    assert "unknown" in values
