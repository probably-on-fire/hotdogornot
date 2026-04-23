import numpy as np
import pytest
from rfconnectorai.data.synthetic import render_connector_sample, make_connector_mesh


def test_make_connector_mesh_returns_trimesh():
    import trimesh
    m = make_connector_mesh(gender="male", family="precision")
    assert isinstance(m, trimesh.Trimesh)
    assert m.vertices.shape[1] == 3
    assert m.faces.shape[1] == 3


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("family", ["sma", "precision"])
def test_make_connector_mesh_variants(gender, family):
    m = make_connector_mesh(gender=gender, family=family)
    assert len(m.vertices) > 0


def test_render_connector_sample_returns_rgb_depth_mask():
    rgb, depth, mask = render_connector_sample(
        gender="male", family="precision", image_size=128, seed=1
    )
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (128, 128)
    assert depth.dtype == np.float32
    assert mask.shape == (128, 128)
    assert mask.dtype == bool


def test_render_has_some_foreground():
    _, _, mask = render_connector_sample(
        gender="male", family="precision", image_size=128, seed=1
    )
    # At least 1% of pixels should be foreground.
    assert mask.mean() > 0.01


def test_render_determinism_per_seed():
    rgb1, d1, m1 = render_connector_sample("male", "precision", image_size=64, seed=7)
    rgb2, d2, m2 = render_connector_sample("male", "precision", image_size=64, seed=7)
    np.testing.assert_array_equal(rgb1, rgb2)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(m1, m2)
