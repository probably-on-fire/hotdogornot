"""Instance manifest schema for the connector dataset.

Validates rows in `datasets/rfconnectors/instances.jsonl`. One row equals one
visible connector instance, not one source image. Two-sided adapters must
populate both `side_a` and `side_b` blocks; single connectors may leave
`side_b` as `None` or fill it with `not_applicable` values.

Kept dependency-free (stdlib + dataclasses) so it can be imported in tooling
that does not pull in pydantic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LabelConfidence(str, Enum):
    HUMAN_VERIFIED = "human_verified"
    WEAK_FOLDER_LABEL = "weak_folder_label"
    SYNTHETIC_VERIFIED = "synthetic_verified"
    MODEL_SUGGESTED = "model_suggested"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    REAL_PHOTO = "real_photo"
    REAL_VIDEO_FRAME = "real_video_frame"
    PRODUCT_LISTING = "product_listing"
    SYNTHETIC_RENDER = "synthetic_render"
    UNKNOWN = "unknown"


VALID_GENDERS = {
    "male_pin",
    "female_socket",
    "rp_male_body_female_contact",
    "rp_female_body_male_contact",
    "not_applicable",
    "insufficient_view",
    "unknown",
}

VALID_POLARITIES = {
    "standard",
    "reverse_polarity",
    "not_applicable",
    "insufficient_view",
    "unknown",
}


@dataclass(frozen=True)
class GeometryLabel:
    """Geometry cues; nulls are normal when no scale reference is present."""

    thread_diameter_mm: float | None = None
    thread_pitch_or_count: float | str | None = None
    body_length_mm: float | None = None
    hex_size_mm: float | None = None
    aperture_mm: float | None = None
    requires_calibrated_reference: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_diameter_mm": self.thread_diameter_mm,
            "thread_pitch_or_count": self.thread_pitch_or_count,
            "body_length_mm": self.body_length_mm,
            "hex_size_mm": self.hex_size_mm,
            "aperture_mm": self.aperture_mm,
            "requires_calibrated_reference": self.requires_calibrated_reference,
        }


@dataclass(frozen=True)
class ConnectorSide:
    """One side of a connector or adapter.

    For two-sided adapters, both `side_a` and `side_b` must be populated. For
    single-ended connectors, `side_b` is typically `None` or filled with
    `not_applicable` values.
    """

    family: str
    precision_family: str = "not_applicable"
    gender: str = "unknown"
    polarity: str = "not_applicable"
    threaded: bool | None = None
    coupling: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "precision_family": self.precision_family,
            "gender": self.gender,
            "polarity": self.polarity,
            "threaded": self.threaded,
            "coupling": self.coupling,
        }


@dataclass(frozen=True)
class ConnectorInstance:
    """One row of `datasets/rfconnectors/instances.jsonl`."""

    instance_id: str
    source_image: str
    crop_path: str
    bbox_xyxy: tuple[int, int, int, int]
    label_confidence: LabelConfidence
    source_type: SourceType

    # Flat fields kept for backward compatibility with simple consumers.
    family: str
    precision_family: str = "not_applicable"
    side_a_gender: str = "unknown"
    side_b_gender: str = "not_applicable"
    polarity: str = "not_applicable"
    mount_style: str = "unknown"
    orientation: str = "unknown"
    termination: str = "not_applicable"
    finish_material_cue: str = "unknown"

    # Nested side-aware fields. `side_a` is required; `side_b` is optional and
    # only populated for two-sided adapters.
    side_a: ConnectorSide | None = None
    side_b: ConnectorSide | None = None

    geometry: GeometryLabel = field(default_factory=GeometryLabel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "source_image": self.source_image,
            "crop_path": self.crop_path,
            "bbox_xyxy": list(self.bbox_xyxy),
            "label_confidence": self.label_confidence.value,
            "source_type": self.source_type.value,
            "family": self.family,
            "precision_family": self.precision_family,
            "side_a_gender": self.side_a_gender,
            "side_b_gender": self.side_b_gender,
            "polarity": self.polarity,
            "mount_style": self.mount_style,
            "orientation": self.orientation,
            "termination": self.termination,
            "finish_material_cue": self.finish_material_cue,
            "side_a": self.side_a.to_dict() if self.side_a else None,
            "side_b": self.side_b.to_dict() if self.side_b else None,
            "geometry": self.geometry.to_dict(),
        }


def validate_instance(instance: ConnectorInstance) -> None:
    """Raise ValueError if `instance` violates manifest invariants."""

    if not instance.instance_id:
        raise ValueError("instance_id must not be empty")
    if not instance.source_image:
        raise ValueError(f"{instance.instance_id}: source_image must not be empty")
    if not instance.crop_path:
        raise ValueError(f"{instance.instance_id}: crop_path must not be empty")

    if len(instance.bbox_xyxy) != 4:
        raise ValueError(f"{instance.instance_id}: bbox_xyxy must have 4 ints")
    x1, y1, x2, y2 = instance.bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"{instance.instance_id}: bbox_xyxy must satisfy x2>x1 and y2>y1"
        )

    if instance.side_a_gender not in VALID_GENDERS:
        raise ValueError(
            f"{instance.instance_id}: side_a_gender '{instance.side_a_gender}' "
            f"not in VALID_GENDERS"
        )
    if instance.side_b_gender not in VALID_GENDERS:
        raise ValueError(
            f"{instance.instance_id}: side_b_gender '{instance.side_b_gender}' "
            f"not in VALID_GENDERS"
        )
    if instance.polarity not in VALID_POLARITIES:
        raise ValueError(
            f"{instance.instance_id}: polarity '{instance.polarity}' "
            f"not in VALID_POLARITIES"
        )

    if instance.mount_style == "adapter" and instance.side_b is None:
        raise ValueError(
            f"{instance.instance_id}: adapter mount_style requires side_b "
            f"populated; got None"
        )


def instance_from_dict(data: dict[str, Any]) -> ConnectorInstance:
    """Build a ConnectorInstance from a JSON-loaded dict, with light coercion."""

    side_a_raw = data.get("side_a")
    side_b_raw = data.get("side_b")

    return ConnectorInstance(
        instance_id=data["instance_id"],
        source_image=data["source_image"],
        crop_path=data["crop_path"],
        bbox_xyxy=tuple(data["bbox_xyxy"]),  # type: ignore[arg-type]
        label_confidence=LabelConfidence(data["label_confidence"]),
        source_type=SourceType(data.get("source_type", "unknown")),
        family=data["family"],
        precision_family=data.get("precision_family", "not_applicable"),
        side_a_gender=data.get("side_a_gender", "unknown"),
        side_b_gender=data.get("side_b_gender", "not_applicable"),
        polarity=data.get("polarity", "not_applicable"),
        mount_style=data.get("mount_style", "unknown"),
        orientation=data.get("orientation", "unknown"),
        termination=data.get("termination", "not_applicable"),
        finish_material_cue=data.get("finish_material_cue", "unknown"),
        side_a=ConnectorSide(**side_a_raw) if side_a_raw else None,
        side_b=ConnectorSide(**side_b_raw) if side_b_raw else None,
        geometry=GeometryLabel(**data.get("geometry", {})),
    )
