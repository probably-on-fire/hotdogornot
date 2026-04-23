from pathlib import Path
import pytest
from rfconnectorai.data.classes import load_classes, ConnectorClass


CONFIG = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


def test_load_classes_returns_eight():
    classes = load_classes(CONFIG)
    assert len(classes) == 8


def test_load_classes_ids_are_contiguous():
    classes = load_classes(CONFIG)
    assert [c.id for c in classes] == list(range(8))


def test_load_classes_names_match_spec():
    classes = load_classes(CONFIG)
    names = {c.name for c in classes}
    assert names == {
        "SMA-M", "SMA-F",
        "3.5mm-M", "3.5mm-F",
        "2.92mm-M", "2.92mm-F",
        "2.4mm-M", "2.4mm-F",
    }


def test_precision_classes_flagged_correctly():
    classes = load_classes(CONFIG)
    families = {c.name: c.family for c in classes}
    assert families["SMA-M"] == "sma"
    assert families["SMA-F"] == "sma"
    for name in ["3.5mm-M", "3.5mm-F", "2.92mm-M", "2.92mm-F", "2.4mm-M", "2.4mm-F"]:
        assert families[name] == "precision"


def test_connector_class_is_frozen():
    c = ConnectorClass(
        id=0, name="X", family="sma", gender="male",
        inner_pin_diameter_mm=1.0, frequency_ghz_max=18.0,
        impedance_ohms=50, mating_torque_in_lb=8.0,
    )
    with pytest.raises(Exception):
        c.id = 1  # frozen dataclass must not allow mutation
