from pathlib import Path

import pytest

# Legacy bpy-based pipeline; skip cleanly when bpy isn't installed (CI).
pytest.importorskip("bpy")


@pytest.mark.slow
def test_render_class_with_cube_fixture(tmp_path: Path):
    """End-to-end: render 2 samples of the cube fixture into a class dir."""
    from rfconnectorai.synthetic.pipeline import ClassSpec, render_class
    from rfconnectorai.synthetic.scene import RenderConfig

    fixtures = Path(__file__).parent / "fixtures"
    spec = ClassSpec(
        name="TestClass",
        family="test",
        gender="neutral",
        mesh_path=fixtures / "cube.obj",
    )
    produced = render_class(
        spec=spec,
        n_samples=2,
        output_root=tmp_path,
        render_config=RenderConfig(image_size=64, samples=4, depth_of_field=False),
        dr_config=None,
    )
    assert produced == 2
    class_dir = tmp_path / "TestClass"
    pngs = list(class_dir.glob("*.png"))
    assert len(pngs) == 2
