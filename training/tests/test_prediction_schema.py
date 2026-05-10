from __future__ import annotations

import pytest

from rfconnectorai.schemas.instance import ConnectorSide, GeometryLabel
from rfconnectorai.schemas.prediction import (
    ConfidenceState,
    Detection,
    LabelConfidence,
    LatencyMs,
    LegacyPrediction,
    PredictResponse,
    SpecLookup,
    TopAlternative,
    detection_to_legacy_prediction,
    empty_no_connector_response,
    need_scale_reference_response,
    need_second_angle_response,
)


def _high_confidence(label: str, score: float = 0.9) -> LabelConfidence:
    return LabelConfidence(label=label, confidence=score)


def _detection(
    *,
    bbox=(120, 80, 420, 360),
    family_label: str = "SMA",
    family_score: float = 0.96,
    confidence_state: ConfidenceState = ConfidenceState.HIGH_CONFIDENCE,
    side_b: ConnectorSide | None = None,
    top_alternatives: tuple[TopAlternative, ...] = (),
) -> Detection:
    return Detection(
        bbox=bbox,
        family=LabelConfidence(family_label, family_score),
        precision_family=_high_confidence("standard_sma", 0.91),
        polarity=_high_confidence("standard", 0.92),
        side_a_gender=_high_confidence("male_pin", 0.94),
        side_b_gender=_high_confidence("female_socket" if side_b else "not_applicable", 0.86),
        mount_style=_high_confidence("adapter" if side_b else "cable_mount", 0.90),
        orientation=_high_confidence("right_angle" if side_b else "straight", 0.88),
        termination=_high_confidence("not_applicable", 0.83),
        finish_material_cue=_high_confidence("gold", 0.7),
        side_a=ConnectorSide(family=family_label, gender="male_pin", polarity="standard", threaded=True),
        side_b=side_b,
        geometry=GeometryLabel(thread_diameter_mm=4.0, requires_calibrated_reference=False),
        confidence_state=confidence_state,
        top_alternatives=top_alternatives,
        spec=SpecLookup(impedance_ohms=50, frequency_range="DC-18 GHz typical", coupling="threaded"),
    )


# Fixture cases enumerated in TASKS.md Epic 10 P0 ---------------------------


def test_old_compatible_response_keeps_legacy_fields():
    detection = _detection()
    legacy = detection_to_legacy_prediction(detection)
    response = PredictResponse(
        image_width=1920,
        image_height=1080,
        predictions=[legacy],
        detections=[detection],
        detected=True,
        confidence_state=ConfidenceState.HIGH_CONFIDENCE,
        latency_ms=LatencyMs(preprocess=12, detector=31, classifier=18, total=74),
        request_id="abc-123",
    )
    payload = response.to_dict()
    # Legacy fields preserved:
    assert payload["image_width"] == 1920
    assert payload["image_height"] == 1080
    assert payload["predictions"][0]["class_name"] == "SMA"
    assert payload["predictions"][0]["bbox"] == {"x": 120, "y": 80, "w": 300, "h": 280}
    # Rich fields present:
    assert payload["detected"] is True
    assert payload["detections"][0]["family"]["label"] == "SMA"
    assert payload["latency_ms"]["total"] == 74
    assert payload["request_id"] == "abc-123"


def test_no_connector_response_fixture():
    response = empty_no_connector_response(image_width=640, image_height=480)
    payload = response.to_dict()
    assert payload["detected"] is False
    assert payload["detections"] == []
    assert payload["predictions"] == []
    assert payload["confidence_state"] == "no_connector_detected"


def test_ambiguous_response_includes_top_alternatives():
    detection = _detection(
        family_label="SMA",
        family_score=0.55,
        confidence_state=ConfidenceState.AMBIGUOUS,
        top_alternatives=(
            TopAlternative(label="RP-SMA right-angle adapter", confidence=0.41),
            TopAlternative(label="3.5mm precision adapter", confidence=0.18),
        ),
    )
    response = PredictResponse(
        image_width=1920, image_height=1080,
        detections=[detection], detected=True,
        confidence_state=ConfidenceState.AMBIGUOUS,
    )
    payload = response.to_dict()
    assert payload["confidence_state"] == "ambiguous"
    alts = payload["detections"][0]["top_alternatives"]
    assert len(alts) == 2
    assert alts[0]["label"] == "RP-SMA right-angle adapter"


def test_multi_detection_adapter_carries_side_b():
    side_b = ConnectorSide(
        family="BNC", gender="female_socket",
        polarity="not_applicable", coupling="bayonet",
    )
    cable = _detection()
    adapter = _detection(bbox=(500, 200, 800, 500), side_b=side_b)
    response = PredictResponse(
        image_width=1920, image_height=1080,
        detections=[cable, adapter], detected=True,
        confidence_state=ConfidenceState.HIGH_CONFIDENCE,
    )
    payload = response.to_dict()
    assert len(payload["detections"]) == 2
    adapter_payload = payload["detections"][1]
    assert adapter_payload["side_b"]["family"] == "BNC"
    assert adapter_payload["side_b"]["coupling"] == "bayonet"
    assert adapter_payload["mount_style"]["label"] == "adapter"


def test_need_second_angle_response_fixture():
    response = need_second_angle_response(image_width=1280, image_height=720)
    payload = response.to_dict()
    assert payload["confidence_state"] == "need_second_angle"
    assert payload["warnings"]
    assert payload["detected"] is False


def test_need_scale_reference_response_fixture():
    response = need_scale_reference_response(image_width=1280, image_height=720)
    payload = response.to_dict()
    assert payload["confidence_state"] == "need_scale_reference"
    assert payload["warnings"]


# Schema invariants ---------------------------------------------------------


def test_label_confidence_rejects_out_of_range():
    with pytest.raises(ValueError, match="confidence"):
        LabelConfidence("SMA", 1.5)
    with pytest.raises(ValueError, match="confidence"):
        LabelConfidence("SMA", -0.1)


def test_detection_rejects_invalid_bbox():
    with pytest.raises(ValueError, match="bbox"):
        Detection(
            bbox=(10, 10, 5, 5),
            family=_high_confidence("SMA"),
            precision_family=_high_confidence("standard_sma"),
            polarity=_high_confidence("standard"),
            side_a_gender=_high_confidence("male_pin"),
            side_b_gender=_high_confidence("not_applicable"),
            mount_style=_high_confidence("cable_mount"),
            orientation=_high_confidence("straight"),
            termination=_high_confidence("solder"),
        )


def test_detection_to_legacy_prediction_uses_xywh_bbox():
    detection = _detection(bbox=(100, 200, 300, 500))
    legacy = detection_to_legacy_prediction(detection)
    assert legacy.bbox == {"x": 100, "y": 200, "w": 200, "h": 300}


def test_response_round_trip_through_to_dict_is_stable():
    detection = _detection()
    response = PredictResponse(
        image_width=1920, image_height=1080,
        detections=[detection], detected=True,
        latency_ms=LatencyMs(total=42),
    )
    payload_a = response.to_dict()
    payload_b = response.to_dict()
    assert payload_a == payload_b
