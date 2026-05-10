from pathlib import Path

import pytest
import yaml

from rfconnectorai.schemas.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    REQUIRED_CONNECTOR_IDS,
    load_taxonomy,
)


def test_load_default_taxonomy_contains_required_connectors():
    taxonomy = load_taxonomy()
    assert set(taxonomy.by_id()) == REQUIRED_CONNECTOR_IDS


def test_unknown_is_first_class_connector():
    taxonomy = load_taxonomy()
    unknown = taxonomy.get("unknown")
    assert unknown.display_name == "unknown"
    assert "unsupported" in unknown.aliases
    assert unknown.impedance_ohms is None


def test_taxonomy_includes_required_attribute_values():
    taxonomy = load_taxonomy()
    assert "unknown" in taxonomy.attribute_values["polarity"]
    assert "not_applicable" in taxonomy.attribute_values["polarity"]
    assert "need_second_angle" in taxonomy.attribute_values["confidence_state"]
    assert "no_connector_detected" in taxonomy.attribute_values["confidence_state"]


def test_292mm_aliases_include_k_and_smk():
    taxonomy = load_taxonomy()
    aliases = set(taxonomy.get("2_92mm").aliases)
    assert "K" in aliases
    assert "SMK" in aliases


def test_missing_required_connector_fails_validation(tmp_path: Path):
    data = yaml.safe_load(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
    data["connectors"] = [
        spec for spec in data["connectors"] if spec["id"] != "unknown"
    ]
    path = tmp_path / "connectors.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required connector ids"):
        load_taxonomy(path)
