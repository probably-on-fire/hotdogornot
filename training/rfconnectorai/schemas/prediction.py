"""Prediction response schema for the connector identification API.

Designed to be a *strict superset* of the legacy ``/predict`` response so
existing Flutter clients keep working. New consumers can read the rich
structured fields. Reuses :class:`GeometryLabel` and :class:`ConnectorSide`
from :mod:`rfconnectorai.schemas.instance` so labels and predictions share
one geometry/side-aware schema.

This module is dependency-light (stdlib + dataclasses); it does not pull
in pydantic or FastAPI so it stays unit-testable on the local CPU PC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from rfconnectorai.schemas.instance import ConnectorSide, GeometryLabel


class ConfidenceState(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_VIEW = "insufficient_view"
    NEED_SECOND_ANGLE = "need_second_angle"
    NEED_SCALE_REFERENCE = "need_scale_reference"
    UNSUPPORTED_CONNECTOR = "unsupported_connector"
    NO_CONNECTOR_DETECTED = "no_connector_detected"


@dataclass(frozen=True)
class LabelConfidence:
    """A label paired with its model confidence score in [0, 1]."""

    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"confidence for {self.label!r} must be in [0,1]; got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": float(self.confidence)}


@dataclass(frozen=True)
class TopAlternative:
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": float(self.confidence)}


@dataclass(frozen=True)
class SpecLookup:
    """Spec data joined onto a detection from the connector taxonomy."""

    impedance_ohms: int | list[int] | None = None
    frequency_range: str | None = None
    coupling: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "impedance_ohms": self.impedance_ohms,
            "frequency_range": self.frequency_range,
            "coupling": self.coupling,
        }


@dataclass(frozen=True)
class LegacyPrediction:
    """Old ``/predict[].predictions[]`` shape kept for legacy Flutter clients.

    ``probabilities`` is preserved as a free-form mapping for backwards
    compatibility with existing parsers.
    """

    class_name: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    bbox: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "probabilities": dict(self.probabilities),
        }
        if self.bbox is not None:
            out["bbox"] = dict(self.bbox)
        return out


@dataclass(frozen=True)
class Detection:
    """One connector instance found inside an image."""

    bbox: tuple[int, int, int, int]
    family: LabelConfidence
    precision_family: LabelConfidence
    polarity: LabelConfidence
    side_a_gender: LabelConfidence
    side_b_gender: LabelConfidence
    mount_style: LabelConfidence
    orientation: LabelConfidence
    termination: LabelConfidence
    finish_material_cue: LabelConfidence | None = None
    side_a: ConnectorSide | None = None
    side_b: ConnectorSide | None = None
    geometry: GeometryLabel = field(default_factory=GeometryLabel)
    confidence_state: ConfidenceState = ConfidenceState.HIGH_CONFIDENCE
    warnings: tuple[str, ...] = ()
    top_alternatives: tuple[TopAlternative, ...] = ()
    spec: SpecLookup | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("bbox must have 4 ints")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox must satisfy x2>x1 and y2>y1; got {self.bbox}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bbox": list(self.bbox),
            "family": self.family.to_dict(),
            "precision_family": self.precision_family.to_dict(),
            "polarity": self.polarity.to_dict(),
            "side_a_gender": self.side_a_gender.to_dict(),
            "side_b_gender": self.side_b_gender.to_dict(),
            "mount_style": self.mount_style.to_dict(),
            "orientation": self.orientation.to_dict(),
            "termination": self.termination.to_dict(),
            "side_a": self.side_a.to_dict() if self.side_a else None,
            "side_b": self.side_b.to_dict() if self.side_b else None,
            "geometry": self.geometry.to_dict(),
            "confidence_state": self.confidence_state.value,
            "warnings": list(self.warnings),
            "top_alternatives": [t.to_dict() for t in self.top_alternatives],
        }
        if self.finish_material_cue is not None:
            out["finish_material_cue"] = self.finish_material_cue.to_dict()
        if self.spec is not None:
            out["spec"] = self.spec.to_dict()
        return out


@dataclass(frozen=True)
class LatencyMs:
    preprocess: float = 0.0
    detector: float = 0.0
    classifier: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "preprocess": float(self.preprocess),
            "detector": float(self.detector),
            "classifier": float(self.classifier),
            "total": float(self.total),
        }


@dataclass(frozen=True)
class PredictResponse:
    """Top-level response. ``image_width``/``image_height``/``predictions``
    are the legacy fields that must stay for old clients. ``detected``,
    ``detections``, ``latency_ms``, and ``request_id`` are the rich fields
    new clients consume.
    """

    image_width: int
    image_height: int
    predictions: list[LegacyPrediction] = field(default_factory=list)
    detected: bool = False
    detections: list[Detection] = field(default_factory=list)
    request_id: str | None = None
    latency_ms: LatencyMs = field(default_factory=LatencyMs)
    confidence_state: ConfidenceState = ConfidenceState.NO_CONNECTOR_DETECTED
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "detected": bool(self.detected),
            "detections": [d.to_dict() for d in self.detections],
            "latency_ms": self.latency_ms.to_dict(),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "predictions": [p.to_dict() for p in self.predictions],
            "confidence_state": self.confidence_state.value,
            "warnings": list(self.warnings),
        }


# Convenience constructors covering every Epic 10 fixture case --------------


def empty_no_connector_response(
    *,
    image_width: int,
    image_height: int,
    request_id: str | None = None,
) -> PredictResponse:
    return PredictResponse(
        image_width=image_width,
        image_height=image_height,
        request_id=request_id,
        confidence_state=ConfidenceState.NO_CONNECTOR_DETECTED,
    )


def need_second_angle_response(
    *,
    image_width: int,
    image_height: int,
    request_id: str | None = None,
    warning: str = "Need a second view from a different angle to disambiguate.",
) -> PredictResponse:
    return PredictResponse(
        image_width=image_width,
        image_height=image_height,
        request_id=request_id,
        confidence_state=ConfidenceState.NEED_SECOND_ANGLE,
        warnings=(warning,),
    )


def need_scale_reference_response(
    *,
    image_width: int,
    image_height: int,
    request_id: str | None = None,
    warning: str = "Place a scale reference in frame to enable size verification.",
) -> PredictResponse:
    return PredictResponse(
        image_width=image_width,
        image_height=image_height,
        request_id=request_id,
        confidence_state=ConfidenceState.NEED_SCALE_REFERENCE,
        warnings=(warning,),
    )


def detection_to_legacy_prediction(detection: Detection) -> LegacyPrediction:
    """Project a rich Detection into the legacy prediction shape."""
    x1, y1, x2, y2 = detection.bbox
    bbox = {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
    return LegacyPrediction(
        class_name=detection.family.label,
        confidence=float(detection.family.confidence),
        probabilities={},
        bbox=bbox,
    )
