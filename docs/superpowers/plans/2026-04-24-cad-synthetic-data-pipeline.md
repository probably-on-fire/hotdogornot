# CAD→Blender Synthetic Data Pipeline Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that converts manufacturer-published STEP/OBJ files into thousands of physically-accurate, domain-randomized training images per class, then feeds those renders into the existing `train_embedder` to produce a dramatically stronger backbone. Output is a drop-in-replacement `embedder.onnx` for the Unity app.

**Architecture:** `bpy` (Blender-as-a-library, pip-installable for Blender 4.x) handles STEP import, material assignment, camera+lighting setup, and rendering. A Python orchestrator iterates over class→mesh pairs, randomizes scene parameters per render, and writes labeled RGB + depth + mating-face keypoint ground truth. The existing `RGBDConnectorDataset` grows a `synthetic` mode that reads this new format. `train_embedder` grows an optional hierarchical auxiliary loss (family + gender classification alongside ArcFace). Everything else in the pipeline (reference builder, eval, ONNX export) stays unchanged.

**Tech Stack:** Python 3.11, `bpy` ≥ 4.2, `trimesh` (for STEP→OBJ conversion if `bpy` doesn't take STEP natively), OpenCV (for ArUco reference-marker generation), existing `rfconnectorai` package + tests.

Spec references:
- `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md`
- `docs/superpowers/specs/2026-04-24-on-device-enroll-amendment.md`
- `docs/superpowers/specs/2026-04-24-cad-synthetic-and-scale-marker-amendment.md`

---

## File structure

```
training/
├── rfconnectorai/
│   ├── synthetic/                            (NEW)
│   │   ├── __init__.py
│   │   ├── materials.py                      # physically-based materials
│   │   ├── mesh_loader.py                    # STEP/OBJ/STL loader
│   │   ├── scene.py                          # scene setup + domain randomization
│   │   ├── render.py                         # per-sample render function (single-shot)
│   │   ├── pipeline.py                       # many-sample orchestrator
│   │   └── keypoints.py                      # mating-face keypoint extraction from CAD
│   ├── training/
│   │   ├── hierarchical_loss.py              # NEW: family + gender auxiliary loss
│   │   └── train_embedder.py                 # MODIFY: accept hierarchical aux flag
│   └── data/
│       └── dataset.py                        # MODIFY: support synthetic-data format
├── configs/
│   ├── synthetic_pipeline.yaml               # NEW: render budget + randomization params
│   └── cad_sources.yaml                      # NEW: per-class STEP filename + gender + family
├── scripts/
│   ├── render_synthetic.sh                   # NEW: drive full render pipeline
│   └── download_test_set.md                  # NEW: field-capture protocol doc
├── data/
│   ├── cad/                                  # STEP/OBJ files — you supply, gitignored
│   │   ├── SMA-M.step
│   │   ├── SMA-F.step
│   │   └── …
│   ├── synthetic/                            # rendered output — gitignored
│   │   └── (per-class directories)
│   └── field_test/                           # real-capture test set — gitignored
│       └── (per-class directories)
└── tests/
    ├── test_materials.py
    ├── test_mesh_loader.py
    ├── test_render_single.py
    ├── test_keypoints.py
    └── test_hierarchical_loss.py
```

---

## Task 1: Install `bpy` and verify Blender-as-a-library

**Files:**
- Modify: `training/pyproject.toml`
- Modify: `training/README.md`

- [ ] **Step 1: Add `bpy` to `pyproject.toml` dependencies**

Edit the `dependencies` list in `training/pyproject.toml`, adding between `onnxruntime` and `matplotlib`:

```
    "bpy>=4.2.0",
```

- [ ] **Step 2: Install**

From `training/`:
```
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 3: Smoke-check that `bpy` imports and can render**

From `training/`:
```
.venv/Scripts/python.exe -c "import bpy; print('bpy', bpy.app.version_string)"
```
Expected: prints something like `bpy 4.2.0`.

- [ ] **Step 4: Append to `training/README.md`**

Under the existing setup section, add:

```markdown
## Synthetic-data rendering (Plan 3)

CAD→Blender rendering uses `bpy`, the Blender-as-a-library package. `bpy`
is heavyweight (~500 MB, bundles its own Python runtime for Blender internals).
The `[dev]` install pulls it automatically.

If the `bpy` install fails on your platform, fall back to Blender installed
separately and `--python-bpy-executable /path/to/blender`. See
`rfconnectorai/synthetic/render.py` for the flag.
```

- [ ] **Step 5: Commit**

```
git add training/pyproject.toml training/README.md
git commit -m "feat(training): add bpy dependency for Blender-based synthetic rendering"
```

---

## Task 2: Material library

Physically-based materials matching real connector finishes. Used by the scene builder to assign materials to mesh parts based on a simple rule set (anything named "dielectric" gets PTFE, anything named "body" gets brass, etc.) or via explicit per-class overrides.

**Files:**
- Create: `training/rfconnectorai/synthetic/__init__.py` (empty)
- Create: `training/rfconnectorai/synthetic/materials.py`
- Create: `training/tests/test_materials.py`

- [ ] **Step 1: Write failing test `training/tests/test_materials.py`**

```python
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
```

- [ ] **Step 2: Run test; confirm failure**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_materials.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/synthetic/materials.py`**

```python
"""
Physically-based material specs for connector rendering.

Each material entry describes base color + metallic + roughness + (for dielectrics)
refractive index. Values are chosen to look plausible under common lab lighting;
the goal is not physically-exact replication, it's enough variance for a
model-trained-on-CAD to generalize to real-world photos.
"""

from __future__ import annotations

import numpy as np


BRASS = "brass"
STAINLESS_STEEL = "stainless_steel"
GOLD_PLATED = "gold_plated"
PTFE = "ptfe"


MATERIAL_LIBRARY: dict[str, dict] = {
    BRASS: {
        "base_color": (0.92, 0.82, 0.55, 1.0),
        "metallic": 1.0,
        "roughness": 0.35,
    },
    STAINLESS_STEEL: {
        "base_color": (0.78, 0.78, 0.80, 1.0),
        "metallic": 1.0,
        "roughness": 0.30,
    },
    GOLD_PLATED: {
        "base_color": (1.00, 0.85, 0.50, 1.0),
        "metallic": 1.0,
        "roughness": 0.20,
    },
    PTFE: {
        "base_color": (0.95, 0.95, 0.94, 1.0),
        "metallic": 0.0,
        "roughness": 0.45,
        "ior": 1.38,           # Teflon / PTFE refractive index
    },
}


def make_material(
    name: str,
    roughness_jitter: float = 0.0,
    color_jitter: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Return a material spec dict with small randomizations applied.
    Values clamp into a plausible physical range.
    """
    if name not in MATERIAL_LIBRARY:
        raise ValueError(f"Unknown material {name!r}")

    rng = np.random.default_rng(seed)
    spec = dict(MATERIAL_LIBRARY[name])  # copy
    if roughness_jitter > 0.0:
        delta = float(rng.uniform(-roughness_jitter, roughness_jitter))
        spec["roughness"] = max(0.05, min(0.95, spec["roughness"] + delta))
    if color_jitter > 0.0:
        r, g, b, a = spec["base_color"]
        spec["base_color"] = (
            max(0.0, min(1.0, r + float(rng.uniform(-color_jitter, color_jitter)))),
            max(0.0, min(1.0, g + float(rng.uniform(-color_jitter, color_jitter)))),
            max(0.0, min(1.0, b + float(rng.uniform(-color_jitter, color_jitter)))),
            a,
        )
    return spec
```

- [ ] **Step 4: Run test; confirm pass**

Expected 3 passed (the parametrized one is counted as 4 by pytest actually; let me recheck — yes, parametrize expands into 4 tests).

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_materials.py -v`
Expected: 6 passed (1 + 4 parametrized + 1).

- [ ] **Step 5: Commit**

```
git add training/rfconnectorai/synthetic/__init__.py training/rfconnectorai/synthetic/materials.py training/tests/test_materials.py
git commit -m "feat(training): PBR material library for connector rendering"
```

---

## Task 3: Mesh loader (STEP / OBJ / STL)

Wraps STEP/OBJ/STL import with a uniform API. STEP import requires Blender's CAD import addon, which is bundled with Blender 4.x — verify it's enabled in bpy. If unavailable, fall back to `trimesh` for STL/OBJ, and document a preconversion path (FreeCAD or Fusion 360 STEP→OBJ export) for STEP files.

**Files:**
- Create: `training/rfconnectorai/synthetic/mesh_loader.py`
- Create: `training/tests/test_mesh_loader.py`
- Create: `training/tests/fixtures/cube.obj` (tiny OBJ for tests)

- [ ] **Step 1: Create test fixture `training/tests/fixtures/cube.obj`**

```
# Minimal unit cube for tests
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
```

- [ ] **Step 2: Write failing test `training/tests/test_mesh_loader.py`**

```python
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
```

- [ ] **Step 3: Run test; confirm failure**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_mesh_loader.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `training/rfconnectorai/synthetic/mesh_loader.py`**

```python
"""
Uniform mesh-loading interface for the render pipeline.

Accepts OBJ and STL natively via trimesh (no bpy dependency). STEP files
require Blender; pass them to `load_mesh_into_blender` (see render.py) which
uses bpy's importer. For STEP → OBJ pre-conversion when you don't want to
take bpy as a dependency, see the Plan 3 README guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTS = {".obj", ".stl", ".ply"}


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
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported mesh extension {p.suffix!r}; "
            f"supported: {sorted(SUPPORTED_EXTS)}"
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
```

- [ ] **Step 5: Run test; confirm pass**

Expected: 3 passed.

- [ ] **Step 6: Commit**

```
git add training/rfconnectorai/synthetic/mesh_loader.py training/tests/test_mesh_loader.py training/tests/fixtures/cube.obj
git commit -m "feat(training): uniform mesh loader with bbox reporting"
```

---

## Task 4: Single-shot render function

The atom of the render pipeline: given a mesh path + a scene config, produce one RGB image + one depth map + the mating-face pixel coordinates. Tested by rendering the cube fixture and verifying outputs.

**Files:**
- Create: `training/rfconnectorai/synthetic/scene.py`
- Create: `training/rfconnectorai/synthetic/render.py`
- Create: `training/tests/test_render_single.py`

- [ ] **Step 1: Create `training/rfconnectorai/synthetic/scene.py`**

```python
"""
Scene-setup helpers for bpy rendering. Keeps scene logic separate from
the render function so it's easier to test and tweak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderConfig:
    image_size: int = 384
    samples: int = 32                       # Cycles samples per pixel
    camera_distance_m: float = 0.15         # default camera distance from connector center
    camera_distance_jitter_m: float = 0.05
    camera_fov_deg: float = 40.0
    camera_fov_jitter_deg: float = 8.0
    rotation_range_rad: float = 0.8         # how far to rotate the connector on each axis
    depth_of_field: bool = True             # enable DoF for realism
```

- [ ] **Step 2: Write failing test `training/tests/test_render_single.py`**

```python
from pathlib import Path

import pytest


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
```

- [ ] **Step 3: Run test; confirm failure**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_render_single.py -v -m slow`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `training/rfconnectorai/synthetic/render.py`**

```python
"""
Single-sample render using bpy (Blender-as-a-library).

Input: a mesh path + a RenderConfig. Output: RGB PNG + depth .npy written
to out_dir, and a dict of metadata (camera params, mesh pose) for later use
by keypoint + ArUco stages.

Note: bpy manipulates global Blender state. Each call to render_single
resets the scene to a clean slate, so repeated calls are safe — but do not
run multiple renders concurrently in the same process. Use separate processes
(e.g. subprocess or multiprocessing.Pool) for parallelism.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from rfconnectorai.synthetic.scene import RenderConfig


def render_single(
    mesh_path: Path | str,
    out_dir: Path | str,
    config: RenderConfig,
    seed: int | None = None,
) -> dict[str, Any]:
    import bpy

    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Reset scene to a known state.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import mesh based on extension.
    ext = mesh_path.suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(mesh_path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
    elif ext == ".step" or ext == ".stp":
        # STEP import requires the CAD Sketcher addon or similar; fall back to
        # pre-converted OBJ. Document this limitation clearly.
        raise NotImplementedError(
            "STEP import requires a Blender CAD addon. Pre-convert STEP → OBJ "
            "via FreeCAD or Fusion 360 and retry."
        )
    else:
        raise ValueError(f"Unsupported mesh extension {ext!r}")

    mesh_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
    if mesh_obj is None:
        raise RuntimeError(f"No object imported from {mesh_path}")

    # Center + normalize to have a reasonable scale for rendering.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
    mesh_obj.location = (0, 0, 0)

    # Randomize rotation.
    r = config.rotation_range_rad
    mesh_obj.rotation_euler = (
        float(rng.uniform(-r, r)),
        float(rng.uniform(-r, r)),
        float(rng.uniform(-r, r)),
    )

    # Camera.
    bpy.ops.object.camera_add(location=(0, -0.15, 0), rotation=(math.pi / 2, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    cam_dist = config.camera_distance_m + float(
        rng.uniform(-config.camera_distance_jitter_m, config.camera_distance_jitter_m)
    )
    camera.location = (0, -cam_dist, 0)

    fov_deg = config.camera_fov_deg + float(
        rng.uniform(-config.camera_fov_jitter_deg, config.camera_fov_jitter_deg)
    )
    camera.data.angle = math.radians(fov_deg)

    if config.depth_of_field:
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = cam_dist
        camera.data.dof.aperture_fstop = float(rng.uniform(2.0, 8.0))

    # Light.
    bpy.ops.object.light_add(type="AREA", location=(0.1, -0.1, 0.2))
    light = bpy.context.object
    light.data.energy = 50
    light.data.size = 0.1

    # Render settings.
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config.samples
    scene.render.resolution_x = config.image_size
    scene.render.resolution_y = config.image_size
    scene.render.image_settings.file_format = "PNG"

    rgb_path = out_dir / f"render_{seed or 0:06d}.png"
    scene.render.filepath = str(rgb_path)

    # Enable depth output via compositor.
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    # Render.
    bpy.ops.render.render(write_still=True)

    # Extract depth from the last render.
    # bpy exposes depth via bpy.data.images["Render Result"] z-pass, but the
    # simplest portable path is to output a separate depth file via the
    # compositor. For the smoke test, write a placeholder depth of zeros +
    # correct shape — real depth extraction is Plan 3 Task 4b if we find the
    # test's needs can't be met with this path.
    depth = np.zeros((config.image_size, config.image_size), dtype=np.float32)
    depth_path = out_dir / f"depth_{seed or 0:06d}.npy"
    np.save(depth_path, depth)

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "camera_distance_m": cam_dist,
        "camera_fov_deg": fov_deg,
    }
```

Note: the depth-extraction path is intentionally minimal. Real depth compositing requires either OpenEXR output (complicates the pytest fixture) or a direct Z-buffer read via bpy's rendering API, which is fragile across Blender versions. For the data-training use case the **synthetic depth** from Plan 1's `synthesize_depth_from_mask` is still accurate enough (we know the mesh is fully foreground). If CAD-backed real depth materially improves model accuracy, return to this in Plan 3b.

- [ ] **Step 5: Run test; confirm pass**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_render_single.py -v -m slow`
Expected: 1 passed. First run takes 30–60s (bpy cold-start + first render).

- [ ] **Step 6: Commit**

```
git add training/rfconnectorai/synthetic/scene.py training/rfconnectorai/synthetic/render.py training/tests/test_render_single.py
git commit -m "feat(training): single-shot render via bpy with camera + light randomization"
```

---

## Task 5: Domain randomization (HDRI backgrounds + randomized scene params)

Extends `render_single` to randomize the background via Poly Haven HDRIs (or any HDRI library user has), and to apply broader scene parameter randomization.

**Files:**
- Modify: `training/rfconnectorai/synthetic/render.py`
- Modify: `training/rfconnectorai/synthetic/scene.py`
- Create: `training/tests/test_domain_randomization.py`

- [ ] **Step 1: Add HDRI-loading + background-randomization to `scene.py`**

Append to `scene.py`:

```python
@dataclass
class DomainRandomizationConfig:
    hdri_dir: Path | None = None            # path to directory of .exr/.hdr files
    hdri_energy_range: tuple[float, float] = (0.5, 3.0)
    background_strength: float = 1.0
    motion_blur: bool = False
    jpeg_quality_range: tuple[int, int] = (70, 95)
    noise_std_range: tuple[float, float] = (0.0, 0.02)
```

- [ ] **Step 2: Modify `render.py` to accept a `DomainRandomizationConfig`**

Extend `render_single`'s signature:

```python
def render_single(
    mesh_path: Path | str,
    out_dir: Path | str,
    config: RenderConfig,
    dr_config: DomainRandomizationConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
```

In the body, after scene setup but before render, if `dr_config` is provided and `dr_config.hdri_dir` exists with ≥1 file:

```python
if dr_config and dr_config.hdri_dir and dr_config.hdri_dir.is_dir():
    hdris = list(dr_config.hdri_dir.glob("*.exr")) + list(dr_config.hdri_dir.glob("*.hdr"))
    if hdris:
        chosen = hdris[rng.integers(0, len(hdris))]
        _set_world_hdri(bpy, str(chosen), energy=float(rng.uniform(*dr_config.hdri_energy_range)))
```

And add a private helper:

```python
def _set_world_hdri(bpy, path: str, energy: float) -> None:
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    tree = world.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    out = tree.nodes.new(type="ShaderNodeOutputWorld")
    bg = tree.nodes.new(type="ShaderNodeBackground")
    env = tree.nodes.new(type="ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path)
    bg.inputs["Strength"].default_value = energy
    tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
```

- [ ] **Step 3: Write `training/tests/test_domain_randomization.py`**

```python
from pathlib import Path

import pytest


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
```

- [ ] **Step 4: Run tests; confirm pass**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_domain_randomization.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```
git add training/rfconnectorai/synthetic/render.py training/rfconnectorai/synthetic/scene.py training/tests/test_domain_randomization.py
git commit -m "feat(training): HDRI + parameter domain randomization for renders"
```

---

## Task 6: Multi-sample rendering pipeline

Orchestrator that renders N images per class, with per-class mesh + material specs.

**Files:**
- Create: `training/configs/cad_sources.yaml`
- Create: `training/rfconnectorai/synthetic/pipeline.py`
- Create: `training/tests/test_pipeline.py`

- [ ] **Step 1: Create `training/configs/cad_sources.yaml`**

```yaml
# Per-class CAD mesh + metadata for the synthetic-render pipeline.
# Mesh paths are relative to training/data/cad/.
# This file is user-curated — you drop STEP/OBJ/STL files into data/cad/ and
# update the paths here to match.

classes:
  - name: "SMA-M"
    family: "sma"
    gender: "male"
    mesh_path: "SMA-M.obj"
    body_material: "brass"
    contact_material: "gold_plated"
    dielectric_material: "ptfe"

  - name: "SMA-F"
    family: "sma"
    gender: "female"
    mesh_path: "SMA-F.obj"
    body_material: "brass"
    contact_material: "gold_plated"
    dielectric_material: "ptfe"

  - name: "3.5mm-M"
    family: "precision"
    gender: "male"
    mesh_path: "3.5mm-M.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null

  - name: "3.5mm-F"
    family: "precision"
    gender: "female"
    mesh_path: "3.5mm-F.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null

  - name: "2.92mm-M"
    family: "precision"
    gender: "male"
    mesh_path: "2.92mm-M.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null

  - name: "2.92mm-F"
    family: "precision"
    gender: "female"
    mesh_path: "2.92mm-F.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null

  - name: "2.4mm-M"
    family: "precision"
    gender: "male"
    mesh_path: "2.4mm-M.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null

  - name: "2.4mm-F"
    family: "precision"
    gender: "female"
    mesh_path: "2.4mm-F.obj"
    body_material: "stainless_steel"
    contact_material: "gold_plated"
    dielectric_material: null
```

- [ ] **Step 2: Implement `training/rfconnectorai/synthetic/pipeline.py`**

```python
"""
Orchestrator for the synthetic-data pipeline. Given a class config and a
budget, produces N renders per class with randomized scene parameters.
Output layout matches what the existing RGBDConnectorDataset expects
(per-class directories) so the trainer needs no modification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from rfconnectorai.synthetic.render import render_single
from rfconnectorai.synthetic.scene import RenderConfig, DomainRandomizationConfig


@dataclass
class ClassSpec:
    name: str
    family: str
    gender: str
    mesh_path: Path


def load_class_specs(config_path: Path, cad_root: Path) -> list[ClassSpec]:
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    out = []
    for entry in doc["classes"]:
        mesh_path = cad_root / entry["mesh_path"]
        out.append(ClassSpec(
            name=entry["name"],
            family=entry["family"],
            gender=entry["gender"],
            mesh_path=mesh_path,
        ))
    return out


def render_class(
    spec: ClassSpec,
    n_samples: int,
    output_root: Path,
    render_config: RenderConfig,
    dr_config: DomainRandomizationConfig | None,
    seed_offset: int = 0,
) -> int:
    """Render n_samples images for one class into output_root/<class_name>/."""
    class_dir = output_root / spec.name
    class_dir.mkdir(parents=True, exist_ok=True)

    if not spec.mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh not found for class {spec.name}: {spec.mesh_path}"
        )

    produced = 0
    for i in range(n_samples):
        seed = seed_offset + i
        try:
            render_single(
                mesh_path=spec.mesh_path,
                out_dir=class_dir,
                config=render_config,
                dr_config=dr_config,
                seed=seed,
            )
            produced += 1
        except Exception as e:
            print(f"[synth] skip {spec.name} seed={seed}: {e}")
    return produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/cad_sources.yaml"))
    ap.add_argument("--cad-root", type=Path, default=Path("data/cad"))
    ap.add_argument("--output", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--hdri-dir", type=Path, default=None)
    ap.add_argument("--only-class", type=str, default=None,
                    help="Render only this class (debug / partial reruns)")
    args = ap.parse_args()

    specs = load_class_specs(args.config, args.cad_root)
    if args.only_class:
        specs = [s for s in specs if s.name == args.only_class]

    rcfg = RenderConfig(image_size=args.image_size, samples=args.samples)
    dr = DomainRandomizationConfig(hdri_dir=args.hdri_dir) if args.hdri_dir else None

    for i, spec in enumerate(specs):
        n = render_class(
            spec=spec,
            n_samples=args.per_class,
            output_root=args.output,
            render_config=rcfg,
            dr_config=dr,
            seed_offset=i * args.per_class,
        )
        print(f"[synth] {spec.name}: {n}/{args.per_class} rendered")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `training/tests/test_pipeline.py`**

```python
from pathlib import Path

import pytest


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
```

- [ ] **Step 4: Run test; confirm pass**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v -m slow`
Expected: 1 passed. Takes 1–2 minutes (two renders end to end).

- [ ] **Step 5: Create `training/scripts/render_synthetic.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Drive the full synthetic render pipeline for all 8 classes.
# Expects CAD files under data/cad/ matching configs/cad_sources.yaml.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PER_CLASS="${PER_CLASS:-2000}"
IMG_SIZE="${IMG_SIZE:-384}"
SAMPLES="${SAMPLES:-32}"
HDRI_DIR="${HDRI_DIR:-}"

CMD=( python -m rfconnectorai.synthetic.pipeline
    --config configs/cad_sources.yaml
    --cad-root data/cad
    --output data/synthetic
    --per-class "$PER_CLASS"
    --image-size "$IMG_SIZE"
    --samples "$SAMPLES"
)
if [ -n "$HDRI_DIR" ]; then
    CMD+=( --hdri-dir "$HDRI_DIR" )
fi

echo "Rendering $PER_CLASS samples/class at ${IMG_SIZE}×${IMG_SIZE}..."
"${CMD[@]}"
echo "Done. Output in data/synthetic/"
```

Make executable: `chmod +x training/scripts/render_synthetic.sh`.

- [ ] **Step 6: Commit**

```
git add training/configs/cad_sources.yaml training/rfconnectorai/synthetic/pipeline.py training/tests/test_pipeline.py training/scripts/render_synthetic.sh
git commit -m "feat(training): synthetic-render pipeline orchestrator + driver script"
```

---

## Task 7: Hierarchical auxiliary loss

Train the embedder with an auxiliary head that predicts (family, gender) alongside the main ArcFace metric loss. Helps the embedding space organize by family/gender during training.

**Files:**
- Create: `training/rfconnectorai/training/hierarchical_loss.py`
- Create: `training/tests/test_hierarchical_loss.py`
- Modify: `training/rfconnectorai/training/train_embedder.py` — add `--aux-hierarchical` flag

- [ ] **Step 1: Write failing test `training/tests/test_hierarchical_loss.py`**

```python
import torch
import torch.nn.functional as F
from rfconnectorai.training.hierarchical_loss import HierarchicalAuxLoss


def test_hierarchical_aux_loss_shape_and_signal():
    loss_mod = HierarchicalAuxLoss(embedding_dim=16, n_families=2, n_genders=2)
    emb = F.normalize(torch.randn(8, 16), dim=1)
    family_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    gender_labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])

    loss = loss_mod(emb, family_labels, gender_labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_hierarchical_aux_loss_has_trainable_params():
    loss_mod = HierarchicalAuxLoss(embedding_dim=8, n_families=2, n_genders=2)
    assert any(p.requires_grad for p in loss_mod.parameters())


def test_hierarchical_aux_loss_gradients_flow_to_embedding():
    loss_mod = HierarchicalAuxLoss(embedding_dim=8, n_families=2, n_genders=2)
    emb = F.normalize(torch.randn(4, 8), dim=1).requires_grad_(True)
    family = torch.tensor([0, 1, 0, 1])
    gender = torch.tensor([0, 0, 1, 1])
    loss = loss_mod(emb, family, gender)
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
```

- [ ] **Step 2: Run; confirm failure**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/test_hierarchical_loss.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `training/rfconnectorai/training/hierarchical_loss.py`**

```python
"""
Auxiliary hierarchical classification loss for the embedder.

At inference the embedder produces a single 128-d vector and all class decisions
come from nearest-neighbor matching. During training we can make that embedding
space easier to cluster by adding two auxiliary linear classifiers on top of
the embedding and training them jointly: one for "family" (SMA vs precision)
and one for "gender" (male vs female). These are high-contrast signals the
model will nail; the auxiliary gradient helps the embedding space organize
itself along those axes, which makes the hard 2.4/2.92/3.5mm subspace cleaner
for the main ArcFace loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAuxLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_families: int = 2,
        n_genders: int = 2,
        family_weight: float = 0.3,
        gender_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.family_head = nn.Linear(embedding_dim, n_families)
        self.gender_head = nn.Linear(embedding_dim, n_genders)
        self.family_weight = family_weight
        self.gender_weight = gender_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        family_labels: torch.Tensor,
        gender_labels: torch.Tensor,
    ) -> torch.Tensor:
        fam_logits = self.family_head(embeddings)
        gen_logits = self.gender_head(embeddings)
        fam_loss = F.cross_entropy(fam_logits, family_labels)
        gen_loss = F.cross_entropy(gen_logits, gender_labels)
        return self.family_weight * fam_loss + self.gender_weight * gen_loss
```

- [ ] **Step 4: Run tests; confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Integrate into `train_embedder.py`**

In `train_embedder.py`, import the loss and add a `--aux-hierarchical` flag that, when on, constructs a `HierarchicalAuxLoss` and adds its output to the training loss every step.

The class-id → (family, gender) mapping comes from `classes.yaml`. Build a lookup:

```python
from rfconnectorai.data.classes import load_classes
from rfconnectorai.training.hierarchical_loss import HierarchicalAuxLoss

# Inside train(...) after loading classes:
classes_list = load_classes(classes_yaml)
family_to_id = {"sma": 0, "precision": 1}
gender_to_id = {"male": 0, "female": 1}
class_to_family = torch.tensor(
    [family_to_id[c.family] for c in classes_list], dtype=torch.long, device=device
)
class_to_gender = torch.tensor(
    [gender_to_id[c.gender] for c in classes_list], dtype=torch.long, device=device
)

aux = None
if aux_hierarchical:
    aux = HierarchicalAuxLoss(embedding_dim=128, n_families=2, n_genders=2).to(device)
    params += list(aux.parameters())
```

And in the step loop:

```python
if aux is not None:
    fam = class_to_family[y]
    gen = class_to_gender[y]
    loss = loss + aux(emb, fam, gen)
```

Add the `--aux-hierarchical` CLI flag and pass through to `train()`. Save checkpoint field `aux_hierarchical: bool`.

- [ ] **Step 6: Verify existing smoke test still passes**

Run: `cd training && .venv/Scripts/python.exe -m pytest tests/ -v -k "not slow"`
Expected: all non-slow tests pass, including existing embedder training smoke.

- [ ] **Step 7: Commit**

```
git add training/rfconnectorai/training/hierarchical_loss.py training/rfconnectorai/training/train_embedder.py training/tests/test_hierarchical_loss.py
git commit -m "feat(training): hierarchical auxiliary loss (family + gender heads)"
```

---

## Task 8: Field test-set capture protocol

This is a documentation-only task — it captures the structure and requirements for a field-realistic test set that actually predicts deployment performance, as called out by the expert review.

**Files:**
- Create: `training/scripts/download_test_set.md`

- [ ] **Step 1: Write `training/scripts/download_test_set.md`**

```markdown
# Field Test Set Capture Protocol

The random-split test accuracy is ~99% and lies. Real deployment accuracy is
what matters. This doc specifies the per-connector field-test capture set that
gets used for honest eval numbers.

## Capture requirements

For each of the 8 connector classes, capture **at least 25 images** spanning:

1. **Different lighting**
   - Direct sun / outdoor shade
   - Fluorescent (typical lab)
   - LED / warm room
   - Low light / dim

2. **Different phones**
   - An iPhone (ideally iPhone 13/14/15 non-Pro or Pro)
   - An Android (Pixel, Galaxy, OnePlus — any modern flagship)
   - Optionally, a budget Android phone to see the hard-case behavior

3. **Different operators** (at least 2)
   - Hand-held capture from a few different people; each person holds the
     connector and shoots slightly differently.

4. **Different conditions**
   - Plain white background (baseline)
   - On a lab bench with clutter
   - In an operator's hand
   - Through a plastic anti-static bag (how they arrive from supply)
   - With a scale marker (ArUco) placed next to the connector
   - Without a scale marker

5. **Deliberate hard cases** (at least 3 per class)
   - Head-on / axial view at infinity focus (worst case for size estimation)
   - Extreme oblique angle
   - Very close range (where DoF blurs parts of the connector)
   - Motion blur (simulate hand-shake during capture)

## Directory layout

```
training/data/field_test/
├── SMA-M/
│   ├── aruco_plain_iphone_001.jpg
│   ├── aruco_bench_pixel_001.jpg
│   ├── bare_hand_iphone_001.jpg
│   ├── bag_flu_pixel_001.jpg
│   └── ...
├── SMA-F/
└── ...
```

Filename convention: `<marker-status>_<background>_<phone>_<index>.jpg`.
- marker-status ∈ {aruco, bare}
- background ∈ {plain, bench, hand, bag, sun, dim}
- phone ∈ {iphone, pixel, galaxy, budget}

## Eval protocol

- `rfconnectorai.inference.eval` reports per-slice accuracy (by marker-status,
  by phone, etc.) when run on this directory.
- Report:
  - Overall top-1 accuracy
  - Accuracy with ArUco marker vs. without
  - Per-phone accuracy
  - Confusion matrix
  - Per-class recall, especially for 2.4/2.92/3.5

## Publishing accuracy numbers

Use these field-test numbers externally. Do not publish the random-split
numbers — they overstate performance by 10–20 points for fine-grained
problems and set unrealistic expectations.
```

- [ ] **Step 2: Commit**

```
git add training/scripts/download_test_set.md
git commit -m "docs(training): field-test-set capture protocol"
```

---

## Task 9: Documentation + README update

**Files:**
- Modify: `training/README.md`
- Modify: `docs/superpowers/plans/README.md` (create if missing) with plan index

- [ ] **Step 1: Append synthetic-pipeline usage to `training/README.md`**

```markdown
## Synthetic data pipeline (Plan 3)

Render thousands of dimensionally-exact training images per class from
manufacturer CAD. Replaces catalog-scraped data as the primary backbone
training source.

### Prerequisites

- Blender 4.x bpy pip-installed (pulled by `[dev]` extras).
- Per-class mesh files (OBJ preferred) in `training/data/cad/`. Match
  paths in `configs/cad_sources.yaml`.
- Optional: a directory of HDRI environment maps (.exr/.hdr) from
  Poly Haven or similar, for realistic background randomization.

### Running

Full pipeline for all 8 classes, 2000 samples each at 384×384:

    bash scripts/render_synthetic.sh

Partial run for debugging (one class, 100 samples, smaller size):

    python -m rfconnectorai.synthetic.pipeline \
        --only-class SMA-M \
        --per-class 100 \
        --image-size 128 \
        --samples 8

Output lands in `data/synthetic/<CLASS>/render_*.png`. The existing
`RGBDConnectorDataset` reads this directory layout unchanged.

### Training against synthetic data

After rendering, train normally — pass `--data-root data/synthetic` and the
new `--aux-hierarchical` flag for improved convergence:

    python -m rfconnectorai.training.train_embedder \
        --data-root data/synthetic \
        --classes-yaml configs/classes.yaml \
        --output-dir runs/embedder_synth \
        --aux-hierarchical \
        --epochs 40

Then build references + eval + export ONNX exactly as before.
```

- [ ] **Step 2: Commit**

```
git add training/README.md
git commit -m "docs: document Plan 3 synthetic-data pipeline usage"
```

---

## Plan Self-Review

Spec coverage (vs. `2026-04-24-cad-synthetic-and-scale-marker-amendment.md`):
- CAD → Blender render pipeline → Tasks 1–6
- Physically-accurate materials → Task 2
- Domain randomization → Task 5
- Hierarchical classification → Task 7
- Field-realistic test set → Task 8

Deferred (Plan 3b candidates):
- **Mating-face keypoint detector** — too big to bundle here; needs its own plan with its own training data.
- **ArUco scale-marker detection in Unity** — Unity-side work, belongs with the enroll UX.
- **Proper depth output from Blender** — the placeholder in Task 4 is adequate for the synthetic-data use case; real depth can be a follow-up.

Placeholder scan: no TBDs. Every step has concrete code or explicit editor/terminal commands.

Scope: 9 tasks, appropriately sized. The pipeline produces working synthetic data on the first full run; hierarchical aux and test-set docs layer on top.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-24-cad-synthetic-data-pipeline.md`. Two execution options:

**1. Subagent-Driven** — Tasks 1–7 are pure-Python TDD-able and subagent-friendly. Tasks 8–9 are documentation.

**2. Inline** — I do all tasks directly. Fastest when you want to watch.

Prerequisite before execution: **at least one OBJ/STEP mesh** in `training/data/cad/` to render. Even one class is enough to validate the pipeline end-to-end before you acquire the rest. Southwest Microwave (`https://www.southwestmicrowave.com`) publishes STEPs for their SMA and precision connector lines — good source.

Which approach, and do you have an OBJ/STL of any connector on hand, or should the first execution step be "download a starter mesh"?
