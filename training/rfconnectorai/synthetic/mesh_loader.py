"""
Uniform mesh-loading interface for the render pipeline.

Accepts OBJ / STL / PLY natively via trimesh (no bpy dependency). STEP, GLB,
and GLTF are supported for render-time import by Blender (see render.py).
For bbox preflight on STEP files, the user must preconvert to GLB/OBJ via
FreeCAD (see addendum); this function raises NotImplementedError with a
helpful pointer in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTS = {".obj", ".stl", ".ply", ".step", ".stp", ".glb", ".gltf"}


@dataclass
class MeshInfo:
    path: Path
    bbox_size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]


def load_mesh(path: Path | str) -> MeshInfo:
    """
    Load a mesh file and return its bounding-box geometry. The mesh itself is
    not retained here (trimesh objects are heavy); instead the renderer will
    re-import it via Blender at scene-build time. This function exists so the
    pipeline can enumerate available meshes without loading them all.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh not found: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported mesh extension {ext!r}; "
            f"supported: {sorted(SUPPORTED_EXTS)}"
        )
    if ext in (".step", ".stp"):
        # trimesh doesn't natively handle STEP. Blender's bpy does, at render
        # time — but this function needs a mesh to report bbox. Direct the
        # user to preconvert STEP → GLB via FreeCAD.
        raise NotImplementedError(
            f"STEP bbox preflight not supported directly. "
            f"Preconvert {p} to GLB or OBJ via FreeCAD: "
            f"File → Export → glTF 2.0 Binary (.glb). Or rely on "
            f"bpy's STEP import at render time."
        )

    mesh = trimesh.load(str(p), force="mesh")
    bbox = mesh.bounds  # (2, 3) — min, max
    size = bbox[1] - bbox[0]
    center = (bbox[0] + bbox[1]) / 2
    return MeshInfo(
        path=p,
        bbox_size_m=tuple(float(x) for x in size),
        center_m=tuple(float(x) for x in center),
    )
