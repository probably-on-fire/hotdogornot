from pathlib import Path

import pytest

# The Blender-based renderer requires bpy. We've moved to PIL + pyrender
# for active synthetic generation, but these tests still exist for the
# legacy bpy path; skip them cleanly when bpy isn't installed (e.g. CI).
pytest.importorskip("bpy")


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_render_single_sample_produces_rgb_and_depth(tmp_path: Path):
    """
    Renders the cube fixture once and verifies the outputs exist and have
    the right shape/dtype. Marked slow because bpy import is ~5s cold.
    """
    from rfconnectorai.synthetic.render import render_single
    from rfconnectorai.synthetic.scene import RenderConfig

    out_dir = tmp_path / "renders"
    out_dir.mkdir()

    result = render_single(
        mesh_path=FIXTURES / "cube.obj",
        out_dir=out_dir,
        config=RenderConfig(image_size=64, samples=4, depth_of_field=False),
        seed=0,
    )

    rgb_path = result["rgb_path"]
    depth_path = result["depth_path"]

    assert Path(rgb_path).exists()
    assert Path(depth_path).exists()

    from PIL import Image
    img = Image.open(rgb_path)
    assert img.size == (64, 64)

    import numpy as np
    depth = np.load(depth_path)
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
