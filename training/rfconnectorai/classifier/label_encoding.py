"""Multi-head label encoding for the RF connector dataset.

The classifier emits one head per attribute (family, precision_family,
side_a_gender, side_b_gender, polarity, mount_style, orientation,
termination, finish_material_cue). Each head is a categorical with its own
vocabulary, including ``unknown`` and where applicable ``not_applicable``
and ``insufficient_view``.

Missing labels are *first-class*: encoded as ``-1`` in the target tensor
and masked out of the loss in :mod:`rfconnectorai.classifier.train_multihead`.

This module is dependency-light (stdlib only) so unit tests can run on the
local CPU PC without torch.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


# Heads kept in stable order so the model output layout is reproducible
# across runs and across the export pipeline.
HEAD_ORDER: tuple[str, ...] = (
    "family",
    "precision_family",
    "side_a_gender",
    "side_b_gender",
    "polarity",
    "mount_style",
    "orientation",
    "termination",
    "finish_material_cue",
)

MISSING_INDEX = -1


@dataclass(frozen=True)
class HeadVocab:
    """Stable vocabulary for one classification head."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if "unknown" not in self.values:
            raise ValueError(f"head {self.name!r} vocab must include 'unknown'")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"head {self.name!r} vocab has duplicates")

    @property
    def num_classes(self) -> int:
        return len(self.values)

    @property
    def unknown_index(self) -> int:
        return self.values.index("unknown")

    def index_of(self, value: str | None) -> int:
        if value is None or value == "":
            return MISSING_INDEX
        try:
            return self.values.index(value)
        except ValueError:
            return MISSING_INDEX

    def value_of(self, index: int) -> str:
        if index == MISSING_INDEX:
            return "<missing>"
        return self.values[index]


def default_head_vocabs(
    families: Iterable[str],
) -> dict[str, HeadVocab]:
    """Return a stable default vocabulary keyed by head name.

    The ``families`` argument is the sorted set of connector families seen in
    the dataset. The taxonomy ``unknown`` value is always appended.
    """
    family_values = tuple(sorted({f for f in families if f}) + ["unknown"])
    return {
        "family": HeadVocab("family", family_values),
        "precision_family": HeadVocab(
            "precision_family",
            (
                "standard_sma", "rp_sma", "3.5mm", "2.92mm_k_smk", "2.4mm",
                "1.85mm_v", "1.0mm_w", "not_applicable", "unknown",
            ),
        ),
        "side_a_gender": HeadVocab(
            "side_a_gender",
            (
                "male_pin", "female_socket",
                "rp_male_body_female_contact", "rp_female_body_male_contact",
                "not_applicable", "insufficient_view", "unknown",
            ),
        ),
        "side_b_gender": HeadVocab(
            "side_b_gender",
            (
                "male_pin", "female_socket",
                "rp_male_body_female_contact", "rp_female_body_male_contact",
                "not_applicable", "insufficient_view", "unknown",
            ),
        ),
        "polarity": HeadVocab(
            "polarity",
            ("standard", "reverse_polarity", "not_applicable", "insufficient_view", "unknown"),
        ),
        "mount_style": HeadVocab(
            "mount_style",
            (
                "cable_mount", "panel_mount", "bulkhead", "pcb_through_hole",
                "pcb_edge_mount", "pcb_surface_mount", "adapter", "terminator", "unknown",
            ),
        ),
        "orientation": HeadVocab(
            "orientation",
            ("straight", "right_angle", "tee", "adapter_stack", "unknown"),
        ),
        "termination": HeadVocab(
            "termination",
            ("solder", "crimp", "clamp", "molded_cable", "not_applicable", "unknown"),
        ),
        "finish_material_cue": HeadVocab(
            "finish_material_cue",
            ("gold", "nickel_silver", "black_body", "mixed", "unknown"),
        ),
    }


@dataclass
class EncodedRow:
    """One labeled crop encoded for multi-head training."""

    instance_id: str
    crop_path: str
    targets: dict[str, int] = field(default_factory=dict)

    def target_vector(self, head_order: tuple[str, ...] = HEAD_ORDER) -> list[int]:
        return [self.targets.get(h, MISSING_INDEX) for h in head_order]


def encode_row(
    row: Mapping[str, str | None],
    vocabs: Mapping[str, HeadVocab],
) -> EncodedRow:
    targets: dict[str, int] = {}
    for head_name, vocab in vocabs.items():
        raw = row.get(head_name)
        if isinstance(raw, str):
            raw = raw.strip()
        targets[head_name] = vocab.index_of(raw if isinstance(raw, str) else None)
    return EncodedRow(
        instance_id=str(row.get("instance_id", "")),
        crop_path=str(row.get("crop_path") or row.get("source_image") or ""),
        targets=targets,
    )


def read_attributes_csv(path: Path) -> list[dict[str, str]]:
    """Read ``attributes.csv`` produced by build_yolo_dataset."""
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def encode_attributes(
    rows: Iterable[Mapping[str, str]],
    vocabs: Mapping[str, HeadVocab],
) -> list[EncodedRow]:
    return [encode_row(r, vocabs) for r in rows]


def class_balance(
    encoded: Iterable[EncodedRow],
    head_name: str,
    vocab: HeadVocab,
) -> dict[str, int]:
    counts = {value: 0 for value in vocab.values}
    counts["<missing>"] = 0
    for row in encoded:
        idx = row.targets.get(head_name, MISSING_INDEX)
        if idx == MISSING_INDEX:
            counts["<missing>"] += 1
        else:
            counts[vocab.values[idx]] += 1
    return counts
