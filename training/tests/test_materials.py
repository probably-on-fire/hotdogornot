import pytest
from rfconnectorai.synthetic.materials import (
    make_material,
    BRASS,
    STAINLESS_STEEL,
    GOLD_PLATED,
    PTFE,
    MATERIAL_LIBRARY,
)


def test_material_library_has_expected_entries():
    assert BRASS in MATERIAL_LIBRARY
    assert STAINLESS_STEEL in MATERIAL_LIBRARY
    assert GOLD_PLATED in MATERIAL_LIBRARY
    assert PTFE in MATERIAL_LIBRARY


@pytest.mark.parametrize("name", [BRASS, STAINLESS_STEEL, GOLD_PLATED, PTFE])
def test_each_material_has_required_fields(name):
    props = MATERIAL_LIBRARY[name]
    assert "base_color" in props
    assert "metallic" in props
    assert "roughness" in props
    # PTFE is dielectric — allow 0 metallic; metals should be high.
    if name == PTFE:
        assert props["metallic"] == 0.0
        assert "ior" in props  # dielectric needs refractive index
    else:
        assert props["metallic"] >= 0.8


def test_make_material_produces_dict_with_all_fields():
    spec = make_material(BRASS, roughness_jitter=0.05, seed=42)
    assert "base_color" in spec
    assert "roughness" in spec
    # Jitter applied: roughness is within ± jitter of the library value
    base_rough = MATERIAL_LIBRARY[BRASS]["roughness"]
    assert abs(spec["roughness"] - base_rough) <= 0.05
