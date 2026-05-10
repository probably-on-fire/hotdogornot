from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "specs" / "connectors.yaml"
)

REQUIRED_CONNECTOR_FIELDS = {
    "id",
    "display_name",
    "aliases",
    "impedance_ohms",
    "frequency_range",
    "coupling",
    "polarity",
    "compatibility",
    "visual_notes",
    "notes",
}

REQUIRED_CONNECTOR_IDS = {
    "sma",
    "rp_sma",
    "3_5mm",
    "2_92mm",
    "2_4mm",
    "1_85mm",
    "1_0mm",
    "ssma",
    "smb",
    "smc",
    "qma",
    "tnc",
    "bnc",
    "mcx",
    "7_16_din",
    "unknown",
}


@dataclass(frozen=True)
class ConnectorSpec:
    id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    impedance_ohms: int | list[int] | None = None
    frequency_range: str = "unknown"
    coupling: str = "unknown"
    polarity: str = "unknown"
    compatibility: dict[str, list[str]] = field(default_factory=dict)
    visual_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def all_names(self) -> set[str]:
        return {self.display_name, *self.aliases}


@dataclass(frozen=True)
class ConnectorTaxonomy:
    schema_version: int
    attribute_values: dict[str, list[str]]
    connectors: list[ConnectorSpec]

    def by_id(self) -> dict[str, ConnectorSpec]:
        return {spec.id: spec for spec in self.connectors}

    def get(self, connector_id: str) -> ConnectorSpec:
        try:
            return self.by_id()[connector_id]
        except KeyError as exc:
            raise KeyError(f"unknown connector id: {connector_id}") from exc


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> ConnectorTaxonomy:
    data = _load_yaml(path)
    taxonomy = _parse_taxonomy(data)
    validate_taxonomy(taxonomy)
    return taxonomy


def validate_taxonomy(taxonomy: ConnectorTaxonomy) -> None:
    if taxonomy.schema_version < 1:
        raise ValueError("schema_version must be >= 1")
    if not taxonomy.attribute_values:
        raise ValueError("attribute_values must not be empty")
    if "unknown" not in taxonomy.attribute_values.get("polarity", []):
        raise ValueError("polarity attribute values must include unknown")
    if "not_applicable" not in taxonomy.attribute_values.get("polarity", []):
        raise ValueError("polarity attribute values must include not_applicable")

    ids = [spec.id for spec in taxonomy.connectors]
    if len(ids) != len(set(ids)):
        raise ValueError(f"connector ids must be unique; got {ids}")

    missing = REQUIRED_CONNECTOR_IDS - set(ids)
    if missing:
        raise ValueError(f"taxonomy missing required connector ids: {sorted(missing)}")

    for spec in taxonomy.connectors:
        _validate_connector_spec(spec)


def _load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("taxonomy YAML must contain a mapping at the top level")
    return data


def _parse_taxonomy(data: dict[str, Any]) -> ConnectorTaxonomy:
    raw_connectors = data.get("connectors")
    if not isinstance(raw_connectors, list):
        raise ValueError("connectors must be a list")

    connectors = []
    for raw in raw_connectors:
        if not isinstance(raw, dict):
            raise ValueError("each connector spec must be a mapping")
        missing = REQUIRED_CONNECTOR_FIELDS - set(raw)
        if missing:
            raise ValueError(
                f"connector spec {raw.get('id', '<missing id>')} missing fields: "
                f"{sorted(missing)}"
            )
        connectors.append(ConnectorSpec(**{field: raw[field] for field in REQUIRED_CONNECTOR_FIELDS}))

    attribute_values = data.get("attribute_values", {})
    if not isinstance(attribute_values, dict):
        raise ValueError("attribute_values must be a mapping")
    for name, values in attribute_values.items():
        if not isinstance(name, str) or not isinstance(values, list):
            raise ValueError("attribute_values entries must be string -> list")
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"attribute_values.{name} must contain only strings")

    return ConnectorTaxonomy(
        schema_version=int(data.get("schema_version", 0)),
        attribute_values=attribute_values,
        connectors=connectors,
    )


def _validate_connector_spec(spec: ConnectorSpec) -> None:
    if not spec.id:
        raise ValueError("connector id must not be empty")
    if not spec.display_name:
        raise ValueError(f"connector {spec.id} display_name must not be empty")
    if not isinstance(spec.aliases, list):
        raise ValueError(f"connector {spec.id} aliases must be a list")
    if spec.impedance_ohms is not None:
        if isinstance(spec.impedance_ohms, int):
            pass
        elif (
            isinstance(spec.impedance_ohms, list)
            and spec.impedance_ohms
            and all(isinstance(value, int) for value in spec.impedance_ohms)
        ):
            pass
        else:
            raise ValueError(
                f"connector {spec.id} impedance_ohms must be int, list[int], or null"
            )
    if not isinstance(spec.compatibility, dict):
        raise ValueError(f"connector {spec.id} compatibility must be a mapping")
    for key in ("mates_with", "not_compatible_with"):
        values = spec.compatibility.get(key)
        if not isinstance(values, list):
            raise ValueError(f"connector {spec.id} compatibility.{key} must be a list")
        if not all(isinstance(value, str) for value in values):
            raise ValueError(
                f"connector {spec.id} compatibility.{key} must contain only strings"
            )
    if not all(isinstance(note, str) for note in spec.visual_notes):
        raise ValueError(f"connector {spec.id} visual_notes must contain only strings")
    if not all(isinstance(note, str) for note in spec.notes):
        raise ValueError(f"connector {spec.id} notes must contain only strings")
