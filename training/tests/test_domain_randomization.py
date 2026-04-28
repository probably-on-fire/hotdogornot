from pathlib import Path

import pytest

# Legacy bpy-based renderer; skip cleanly when bpy isn't installed (CI).
pytest.importorskip("bpy")


@pytest.mark.slow
def test_domain_randomization_with_no_hdri_dir_still_renders(tmp_path: Path):
    """When dr_config has no HDRI dir, render_single falls back to plain sky."""
    from rfconnectorai.synthetic.render import render_single
    from rfconnectorai.synthetic.scene import RenderConfig, DomainRandomizationConfig

    result = render_single(
        mesh_path=Path(__file__).parent / "fixtures" / "cube.obj",
        out_dir=tmp_path,
        config=RenderConfig(image_size=64, samples=4, depth_of_field=False),
        dr_config=DomainRandomizationConfig(hdri_dir=None),
        seed=1,
    )
    assert Path(result["rgb_path"]).exists()
