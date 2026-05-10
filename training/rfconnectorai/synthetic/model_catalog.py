"""Catalog of parametric synthetic connector models.

Defines, in pure data, the connector variants the cloud render pipeline
will instantiate. The catalog is decoupled from any 3D engine so it can be
unit-tested locally without Blender/Trimesh/Open3D installed.

Each :class:`ParametricConnectorModel` carries:

- a stable ``model_id`` that downstream renders will record on every
  emitted label, so synthetic samples remain traceable to their generator;
- a ``family`` from the taxonomy;
- ``side_a`` / ``side_b`` :class:`SideSpec` describing each side of the
  connector or adapter;
- a tuple of geometry overrides (thread diameter, body length, hex size,
  etc.) used when the renderer parameterizes the model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SideSpec:
    family: str
    gender: str = "male_pin"
    polarity: str = "standard"
    threaded: bool | None = None
    coupling: str | None = None


@dataclass(frozen=True)
class GeometryOverrides:
    thread_diameter_mm: float | None = None
    thread_pitch_or_count: float | None = None
    body_length_mm: float | None = None
    hex_size_mm: float | None = None
    aperture_mm: float | None = None


@dataclass(frozen=True)
class ParametricConnectorModel:
    model_id: str
    family: str
    description: str
    side_a: SideSpec
    side_b: SideSpec | None = None
    mount_style: str = "cable_mount"
    orientation: str = "straight"
    termination: str = "not_applicable"
    finish_material_cue: str = "nickel_silver"
    geometry: GeometryOverrides = field(default_factory=GeometryOverrides)
    is_confusing_negative: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def _pair_adapter(family_a: str, family_b: str, *, model_id: str, **kwargs) -> ParametricConnectorModel:
    return ParametricConnectorModel(
        model_id=model_id,
        family=family_a,
        description=f"{family_a}-to-{family_b} adapter",
        side_a=SideSpec(family=family_a, gender="male_pin", polarity="standard", threaded=family_a in {"SMA", "TNC"}),
        side_b=SideSpec(
            family=family_b,
            gender="female_socket",
            polarity="not_applicable",
            coupling="bayonet" if family_b == "BNC" else None,
            threaded=family_b in {"SMA", "TNC", "7/16 DIN"},
        ),
        mount_style="adapter",
        **kwargs,
    )


_BUILTIN_MODELS: tuple[ParametricConnectorModel, ...] = (
    ParametricConnectorModel(
        model_id="sma_male_straight",
        family="SMA",
        description="SMA male straight, threaded, gold center pin",
        side_a=SideSpec(family="SMA", gender="male_pin", polarity="standard", threaded=True),
        finish_material_cue="gold",
        geometry=GeometryOverrides(thread_diameter_mm=4.32, thread_pitch_or_count=36.0),
    ),
    ParametricConnectorModel(
        model_id="sma_female_straight",
        family="SMA",
        description="SMA female straight",
        side_a=SideSpec(family="SMA", gender="female_socket", polarity="standard", threaded=True),
        geometry=GeometryOverrides(thread_diameter_mm=4.32),
    ),
    ParametricConnectorModel(
        model_id="rp_sma_male_straight",
        family="RP-SMA",
        description="Reverse-polarity SMA male body, female center contact",
        side_a=SideSpec(
            family="RP-SMA",
            gender="rp_male_body_female_contact",
            polarity="reverse_polarity",
            threaded=True,
        ),
    ),
    ParametricConnectorModel(
        model_id="rp_sma_female_straight",
        family="RP-SMA",
        description="Reverse-polarity SMA female body, male center contact",
        side_a=SideSpec(
            family="RP-SMA",
            gender="rp_female_body_male_contact",
            polarity="reverse_polarity",
            threaded=True,
        ),
    ),
    ParametricConnectorModel(
        model_id="sma_right_angle_male",
        family="SMA",
        description="SMA right-angle male",
        side_a=SideSpec(family="SMA", gender="male_pin", polarity="standard", threaded=True),
        orientation="right_angle",
    ),
    ParametricConnectorModel(
        model_id="sma_tee_adapter",
        family="SMA",
        description="SMA tee/splitter adapter",
        side_a=SideSpec(family="SMA", gender="female_socket", polarity="standard", threaded=True),
        side_b=SideSpec(family="SMA", gender="female_socket", polarity="standard", threaded=True),
        mount_style="adapter",
        orientation="tee",
    ),
    ParametricConnectorModel(
        model_id="sma_bulkhead",
        family="SMA",
        description="SMA bulkhead/panel mount",
        side_a=SideSpec(family="SMA", gender="female_socket", polarity="standard", threaded=True),
        mount_style="bulkhead",
        geometry=GeometryOverrides(hex_size_mm=8.0),
    ),
    ParametricConnectorModel(
        model_id="sma_crimp_cable",
        family="SMA",
        description="SMA crimp on coax cable",
        side_a=SideSpec(family="SMA", gender="male_pin", polarity="standard", threaded=True),
        termination="crimp",
    ),
    _pair_adapter("SMA", "SMA", model_id="adapter_sma_to_sma"),
    _pair_adapter("SMA", "BNC", model_id="adapter_sma_to_bnc"),
    _pair_adapter("SMA", "TNC", model_id="adapter_sma_to_tnc"),
    _pair_adapter("SMA", "MCX", model_id="adapter_sma_to_mcx"),
    ParametricConnectorModel(
        model_id="precision_3_5mm_male",
        family="3.5mm",
        description="3.5mm precision male",
        side_a=SideSpec(family="3.5mm", gender="male_pin", polarity="standard", threaded=True),
    ),
    ParametricConnectorModel(
        model_id="precision_2_92mm_male",
        family="2.92mm",
        description="2.92mm/K/SMK precision male",
        side_a=SideSpec(family="2.92mm", gender="male_pin", polarity="standard", threaded=True),
    ),
    ParametricConnectorModel(
        model_id="precision_2_4mm_male",
        family="2.4mm",
        description="2.4mm precision male",
        side_a=SideSpec(family="2.4mm", gender="male_pin", polarity="standard", threaded=True),
    ),
    ParametricConnectorModel(
        model_id="precision_1_85mm_male",
        family="1.85mm",
        description="1.85mm/V precision male",
        side_a=SideSpec(family="1.85mm", gender="male_pin", polarity="standard", threaded=True),
    ),
    ParametricConnectorModel(
        model_id="precision_1_0mm_male",
        family="1.0mm",
        description="1.0mm/W precision male",
        side_a=SideSpec(family="1.0mm", gender="male_pin", polarity="standard", threaded=True),
    ),
    ParametricConnectorModel(
        model_id="bnc_male_straight",
        family="BNC",
        description="BNC bayonet male",
        side_a=SideSpec(family="BNC", gender="male_pin", polarity="standard", coupling="bayonet"),
    ),
    ParametricConnectorModel(
        model_id="confusing_neg_audio_3_5mm",
        family="unknown",
        description="3.5mm headphone TRS jack — common confusing negative",
        side_a=SideSpec(family="unknown", gender="male_pin", polarity="not_applicable"),
        is_confusing_negative=True,
    ),
)


def builtin_models() -> tuple[ParametricConnectorModel, ...]:
    return _BUILTIN_MODELS


def iter_models(
    *,
    families: Iterable[str] | None = None,
    include_confusing_negatives: bool = True,
) -> Iterator[ParametricConnectorModel]:
    family_filter = set(families) if families else None
    for model in _BUILTIN_MODELS:
        if family_filter and model.family not in family_filter:
            continue
        if not include_confusing_negatives and model.is_confusing_negative:
            continue
        yield model


def model_by_id(model_id: str) -> ParametricConnectorModel:
    for model in _BUILTIN_MODELS:
        if model.model_id == model_id:
            return model
    raise KeyError(f"unknown synthetic model id: {model_id}")
