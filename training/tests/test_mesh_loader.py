from pathlib import Path

import pytest
from rfconnectorai.synthetic.mesh_loader import load_mesh, MeshInfo


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_obj_returns_meshinfo():
    info = load_mesh(FIXTURES / "cube.obj")
    assert isinstance(info, MeshInfo)
    assert info.path == FIXTURES / "cube.obj"
    assert info.bbox_size_m[0] > 0
    assert info.bbox_size_m[1] > 0
    assert info.bbox_size_m[2] > 0


def test_load_mesh_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "nonexistent.obj")


def test_load_mesh_rejects_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "something.xyz"
    bad.write_text("fake")
    with pytest.raises(ValueError, match="Unsupported"):
        load_mesh(bad)
