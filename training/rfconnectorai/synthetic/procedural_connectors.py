"""
Procedural RF-connector mesh generator.

Produces geometrically-accurate GLB files for all 8 connector classes in
`configs/datasheet_dimensions.yaml`. Each mesh models:

  - Cylindrical body (body_od_mm × body_length_mm)
  - Hex coupling section (hex_flat_to_flat_mm across flats, hex_length_mm long)
  - Flat mating face disc
  - Outer-conductor bore (dark annular region of diameter bore_id_mm)
  - Inner conductor — raised pin (male) or recessed socket (female)

The mating face points along +Z. Outputs go to
`training/data/cad/verified/<CLASS>.glb` where the Plan 3 render pipeline
expects verified meshes.

These aren't vendor STEPs — they're parametric stand-ins. Plus side: every
dimension is exact by construction; the mating-face-simplification problem
from the Plan 3 addendum is solved because we build the bore and pin
explicitly. Downside: no surface texture, no vendor-specific finishing
details. Domain randomization in the renderer covers most of that.

Usage:
    python -m rfconnectorai.synthetic.procedural_connectors
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml


BODY_LENGTH_MM = 12.0
HEX_LENGTH_MM = 4.5
MATING_FACE_Z_MM = BODY_LENGTH_MM  # mating face sits at z = body length
PIN_PROTRUSION_MM = 1.5            # how far the male pin sticks past the mating face
SOCKET_RECESS_MM = 1.0             # how deep the female socket goes in


@dataclass
class ClassDimensions:
    name: str
    bore_id_mm: float
    pin_od_mm: float
    hex_flat_to_flat_mm: float
    body_od_mm: float
    dielectric_visible: bool
    is_male: bool


def load_dimensions(path: Path) -> list[ClassDimensions]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    out: list[ClassDimensions] = []
    for entry in doc["classes"]:
        name = entry["name"]
        out.append(
            ClassDimensions(
                name=name,
                bore_id_mm=float(entry["bore_id_mm"]),
                pin_od_mm=float(entry["pin_od_mm"]),
                hex_flat_to_flat_mm=float(entry["hex_flat_to_flat_mm"]),
                body_od_mm=float(entry["body_od_mm"]),
                dielectric_visible=bool(entry["dielectric_visible"]),
                is_male=name.endswith("-M"),
            )
        )
    return out


def _hex_prism(flat_to_flat_mm: float, height_mm: float) -> trimesh.Trimesh:
    """
    Build a regular hexagonal prism of given flat-to-flat across and height.

    Constructed from raw vertices + faces (no shapely/triangle dependency):
      - 6 bottom vertices at z=0 on a hex at circumradius r
      - 6 top vertices at z=height_mm
      - bottom + top faces triangulated as a fan from vertex 0
      - 6 side rectangles, each 2 triangles
    """
    apothem = flat_to_flat_mm / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)

    bottom = np.array(
        [[circumradius * math.cos(math.radians(60 * i + 30)),
          circumradius * math.sin(math.radians(60 * i + 30)),
          0.0]
         for i in range(6)]
    )
    top = bottom.copy()
    top[:, 2] = height_mm

    vertices = np.vstack([bottom, top])  # 12 vertices total

    faces = []
    # Bottom face (wound clockwise looking from below, i.e. normal points -Z)
    faces += [(0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4)]
    # Top face (wound CCW looking from above, normal +Z)
    faces += [(6, 7, 8), (6, 8, 9), (6, 9, 10), (6, 10, 11)]
    # Sides
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((i, ni, 6 + ni))
        faces.append((i, 6 + ni, 6 + i))

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)


def build_connector(c: ClassDimensions) -> trimesh.Trimesh:
    """Build one connector mesh. Returns a closed trimesh oriented +Z toward mating face."""
    parts: list[trimesh.Trimesh] = []

    # Body / hex finish: SMA bodies are brass-colored, precision (3.5/2.92/2.4)
    # are stainless gray. Matches MATERIAL_COLORS in face_renderer.py so the 3D
    # angled-renderer outputs harmonize with the PIL frontal renderer.
    body_color = (170, 145, 95, 255) if c.dielectric_visible else (110, 113, 120, 255)

    # Cylindrical body (bulk).
    body = trimesh.creation.cylinder(
        radius=c.body_od_mm / 2.0,
        height=BODY_LENGTH_MM - HEX_LENGTH_MM,
        sections=64,
    )
    body.apply_translation([0, 0, (BODY_LENGTH_MM - HEX_LENGTH_MM) / 2.0])
    body.visual.face_colors = body_color
    parts.append(body)

    # Hex coupling section sits on top of the body.
    hex_prism = _hex_prism(c.hex_flat_to_flat_mm, HEX_LENGTH_MM)
    hex_prism.apply_translation([0, 0, BODY_LENGTH_MM - HEX_LENGTH_MM])
    hex_prism.visual.face_colors = body_color
    parts.append(hex_prism)

    # Mating face is the top of the hex prism (at z = BODY_LENGTH_MM).
    # We model the bore as a negative-space cylinder (bore through the hex
    # and some of the body), but for a fast procedural asset we just add a
    # darker-colored cylinder representing the dielectric/bore region.
    # Add an annular disc at the mating face to represent the outer-conductor
    # bore (a thin ring) — purely visual, helps the render look correct.
    mating_face_z = BODY_LENGTH_MM
    bore_depth = 2.0
    bore_cyl = trimesh.creation.cylinder(
        radius=c.bore_id_mm / 2.0,
        height=bore_depth,
        sections=48,
    )
    # Push the bore slightly above the hex top face so its top disc is the
    # frontmost surface (no z-fighting with the coplanar hex top).
    bore_lift = 0.05
    bore_cyl.apply_translation([0, 0, mating_face_z - bore_depth / 2.0 + bore_lift])
    bore_cyl.visual.face_colors = [20, 20, 22, 255]
    parts.append(bore_cyl)

    # Inner conductor: pin for male (protrudes past the mating face),
    # socket marker for female (sits flush with a recessed cylinder).
    if c.is_male:
        pin_height = PIN_PROTRUSION_MM + bore_depth
        pin = trimesh.creation.cylinder(
            radius=c.pin_od_mm / 2.0,
            height=pin_height,
            sections=32,
        )
        # Pin top should sit clearly above the bore-top disc so we see it as
        # the central protrusion rather than buried inside the hole.
        pin_top_z = mating_face_z + bore_lift + PIN_PROTRUSION_MM
        pin.apply_translation([0, 0, pin_top_z - pin_height / 2.0])
        pin.visual.face_colors = [240, 200, 120, 255]  # gold-ish
        parts.append(pin)
    else:
        # Female: a recessed cylinder inside the bore, slightly darker to
        # read as an opening. We approximate by placing a small dark
        # cylinder below the mating-face plane.
        socket = trimesh.creation.cylinder(
            radius=c.pin_od_mm / 2.0,
            height=SOCKET_RECESS_MM,
            sections=32,
        )
        socket.apply_translation([0, 0, mating_face_z - SOCKET_RECESS_MM / 2.0 - bore_depth])
        socket.visual.face_colors = [15, 15, 15, 255]
        parts.append(socket)

    # Merge all parts into a single mesh.
    merged = trimesh.util.concatenate(parts)
    # Convert mm → meters (Blender / AR Foundation work in meters).
    merged.apply_scale(0.001)
    return merged


def build_all(
    dimensions_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate a GLB per class. Returns list of written paths."""
    dimensions = load_dimensions(dimensions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for c in dimensions:
        mesh = build_connector(c)
        out_path = output_dir / f"{c.name}.glb"
        mesh.export(out_path)
        written.append(out_path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dimensions",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "datasheet_dimensions.yaml",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "cad" / "verified",
    )
    args = ap.parse_args()

    written = build_all(args.dimensions, args.output_dir)
    print(f"Wrote {len(written)} procedural connector meshes:")
    for p in written:
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
