"""
Off-axis 3D renderer for RF connector meshes (pyrender).

Loads the procedural GLB meshes from `data/cad/verified/<CLASS>.glb` and
renders them from a sphere of camera positions around the mating face. Output
images include the side wall of the hex (visible at non-zero elevation),
which the frontal-only PIL renderer cannot show.

These angled views are NOT consumed by the current measurement pipeline (the
hex/aperture geometry assumes a perpendicular view). They exist for:

  - Visual demos showing what the connector looks like in 3D
  - Future ML-classifier training data that needs angle variation
  - FramingGate validation: feed in an angled image and confirm the gate
    rejects it instead of letting it through to a wrong-bucket prediction

Usage:
    python -m rfconnectorai.synthetic.angled_renderer --per-class 20 --image-size 384
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import yaml
from PIL import Image


# Same procedural background generator as face_renderer for visual consistency.
from rfconnectorai.synthetic.face_renderer import _make_background


def _camera_pose(distance_m: float, elevation_rad: float, azimuth_rad: float,
                 roll_rad: float, target_xyz: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 camera pose matrix.

    Camera looks at `target_xyz`. Position is on a sphere of radius `distance_m`
    around the target, with `elevation_rad` from the +Z axis (0 = directly
    in front of mating face, π/2 = looking along the side) and `azimuth_rad`
    around the Z axis. `roll_rad` rotates the camera about its view axis.
    """
    # Position in spherical coords around target (Z is connector's mating-face normal).
    px = target_xyz[0] + distance_m * math.sin(elevation_rad) * math.cos(azimuth_rad)
    py = target_xyz[1] + distance_m * math.sin(elevation_rad) * math.sin(azimuth_rad)
    pz = target_xyz[2] + distance_m * math.cos(elevation_rad)
    eye = np.array([px, py, pz], dtype=np.float64)

    # Camera basis: forward = target - eye, up world is +Z (then re-ortho).
    forward = target_xyz - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0, 0, 1.0])
    # If forward is parallel to world_up (looking straight down/up the Z axis),
    # pick a different up vector to avoid degeneracy.
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0, 1.0, 0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Apply roll about forward axis.
    if abs(roll_rad) > 1e-6:
        c, s = math.cos(roll_rad), math.sin(roll_rad)
        new_right = c * right + s * up
        new_up = -s * right + c * up
        right, up = new_right, new_up

    # pyrender camera convention: -Z is forward, +Y is up, +X is right.
    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = up
    rot[:3, 2] = -forward
    rot[:3, 3] = eye
    return rot


def _build_scene(mesh: trimesh.Trimesh, light_dir: np.ndarray,
                 ambient: float = 0.3) -> pyrender.Scene:
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],   # transparent — background composited later
        ambient_light=[ambient, ambient, ambient],
    )
    # If the loaded mesh is a Scene (multi-mesh GLB), enumerate its geometries
    # and add each as a pyrender Mesh so per-part vertex colors survive.
    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False))
    else:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    # Directional key light + softer fill from the opposite direction.
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    key_pose = np.eye(4)
    key_pose[:3, 3] = light_dir * 0.3
    # Orient the light to look at origin from light_dir.
    forward = -light_dir / np.linalg.norm(light_dir)
    world_up = np.array([0, 0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0, 1.0, 0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    key_pose[:3, 0] = right
    key_pose[:3, 1] = up
    key_pose[:3, 2] = -forward
    scene.add(key, pose=key_pose)

    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = -light_dir * 0.3
    fill_pose[:3, 0] = -right
    fill_pose[:3, 1] = up
    fill_pose[:3, 2] = forward
    scene.add(fill, pose=fill_pose)

    return scene


def _composite_over_background(rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Alpha-composite the rendered RGBA over the procedural background."""
    if rgba.shape[2] == 3:
        return rgba
    fg = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = fg * a + bg_rgb.astype(np.float32) * (1 - a)
    return out.clip(0, 255).astype(np.uint8)


def _mating_face_target(mesh: trimesh.Trimesh) -> np.ndarray:
    """The point in the mesh we want the camera to aim at (the mating face)."""
    # Mesh is built so the mating face is the highest-Z point, with +Z normal.
    if isinstance(mesh, trimesh.Scene):
        bounds = mesh.bounds
    else:
        bounds = mesh.bounds
    cx = (bounds[0, 0] + bounds[1, 0]) / 2
    cy = (bounds[0, 1] + bounds[1, 1]) / 2
    cz = bounds[1, 2]   # mating face = top of mesh in Z
    return np.array([cx, cy, cz], dtype=np.float64)


def render_angled(
    mesh_path: Path,
    image_size: int = 384,
    elevation_deg: float | None = None,
    azimuth_deg: float | None = None,
    distance_m: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Render one off-axis view of the connector mesh.

    `elevation_deg`: 0 = looking straight at the mating face; positive tilts
                     the camera off-axis. Practical range 0..70.
    `azimuth_deg`:   rotation around the connector's Z axis (which face of the
                     hex faces the camera). 0..360.
    `distance_m`:    camera distance from target in meters. Default scales
                     with the mesh size so the connector fills ~50% of frame.

    Random None values get sampled from `seed`.
    """
    rng = np.random.default_rng(seed)

    if elevation_deg is None:
        elevation_deg = float(rng.uniform(0.0, 60.0))
    if azimuth_deg is None:
        azimuth_deg = float(rng.uniform(0.0, 360.0))

    mesh = trimesh.load(mesh_path, force="scene")
    target = _mating_face_target(mesh)

    # Default distance: chosen so the body width fills ~50% of the frame at
    # 35° fov. Using the X-extent of the bounding box (≈ body OD) keeps
    # framing tight on the connector regardless of total length.
    bounds = mesh.bounds
    body_width = float(bounds[1, 0] - bounds[0, 0])
    if distance_m is None:
        distance_m = body_width * 4.0

    # Light from a random horizontal direction in the connector's local frame.
    light_az = float(rng.uniform(0, 2 * math.pi))
    light_el = float(rng.uniform(math.radians(20), math.radians(60)))
    light_dir = np.array([
        math.cos(light_az) * math.sin(light_el),
        math.sin(light_az) * math.sin(light_el),
        math.cos(light_el),
    ])

    scene = _build_scene(mesh, light_dir)

    # znear must be smaller than the closest geometry; the connector sits
    # only ~30mm from the camera so the default znear of 0.05m clips it.
    cam = pyrender.PerspectiveCamera(
        yfov=math.radians(35), aspectRatio=1.0, znear=0.001, zfar=10.0,
    )
    roll = float(rng.uniform(-math.radians(8), math.radians(8)))
    cam_pose = _camera_pose(
        distance_m=distance_m,
        elevation_rad=math.radians(elevation_deg),
        azimuth_rad=math.radians(azimuth_deg),
        roll_rad=roll,
        target_xyz=target,
    )
    scene.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    try:
        color, _depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()

    bg = _make_background(image_size, rng)
    out = _composite_over_background(color, bg)
    return out


def render_class(
    class_name: str,
    mesh_dir: Path,
    out_dir: Path,
    n_samples: int,
    image_size: int = 384,
    seed_offset: int = 0,
) -> int:
    mesh_path = mesh_dir / f"{class_name}.glb"
    if not mesh_path.exists():
        raise FileNotFoundError(f"missing GLB: {mesh_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        arr = render_angled(mesh_path, image_size=image_size, seed=seed_offset + i)
        Image.fromarray(arr).save(out_dir / f"angled_{i:04d}.png")
    return n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "cad" / "verified",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic_angled",
    )
    ap.add_argument(
        "--dimensions",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "datasheet_dimensions.yaml",
    )
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=384)
    args = ap.parse_args()

    with open(args.dimensions) as f:
        doc = yaml.safe_load(f)

    total = 0
    for entry in doc["classes"]:
        cls = entry["name"]
        out = args.output_dir / cls
        n = render_class(
            class_name=cls,
            mesh_dir=args.mesh_dir,
            out_dir=out,
            n_samples=args.per_class,
            image_size=args.image_size,
            seed_offset=hash(cls) % 100000,
        )
        total += n
        print(f"  {cls}: {n} angled images -> {out}")
    print(f"Done. {total} images total.")


if __name__ == "__main__":
    main()
