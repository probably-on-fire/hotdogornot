import numpy as np
import pytest
from rfconnectorai.data.depth_utils import synthesize_depth_from_mask


def test_synthesize_depth_shape_matches_mask():
    mask = np.zeros((384, 384), dtype=bool)
    mask[100:284, 100:284] = True  # square foreground
    depth = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.12)
    assert depth.shape == (384, 384)
    assert depth.dtype == np.float32


def test_synthesize_depth_background_is_far():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    depth = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1)
    bg_depth = depth[0, 0]
    fg_depth = depth[16, 16]
    assert bg_depth > 1.0        # background should be pushed far (>1m)
    assert 0.05 < fg_depth < 0.2  # foreground near the specified object depth


def test_synthesize_depth_is_deterministic_for_same_seed():
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    d1 = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1, seed=42)
    d2 = synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1, seed=42)
    np.testing.assert_array_equal(d1, d2)


def test_synthesize_depth_rejects_wrong_dtype():
    mask = np.zeros((16, 16), dtype=np.float32)  # wrong dtype
    with pytest.raises(ValueError):
        synthesize_depth_from_mask(mask, focal_length_px=500.0, object_depth_m=0.1)
