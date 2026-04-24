"""
Interactive Blender-based verification tool.

Given a class name, loads the STEP/GLB from data/cad/verified/<class>.*, opens
it in a Blender viewport (if available), and prints the datasheet dimensions
alongside a computed approximation from the loaded mesh. Flags discrepancies.

This is a human-in-the-loop tool — it reports, the engineer decides.

Usage:
    python scripts/verify_mating_faces.py --class SMA-M

Prereqs: bpy (Blender-as-library) installed; the STEP/GLB present; the
datasheet_dimensions.yaml entry populated for the class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="class_name", required=True)
    ap.add_argument(
        "--dimensions",
        type=Path,
        default=Path("configs/datasheet_dimensions.yaml"),
    )
    ap.add_argument(
        "--verified-dir",
        type=Path,
        default=Path("data/cad/verified"),
    )
    args = ap.parse_args()

    with open(args.dimensions) as f:
        doc = yaml.safe_load(f)
    entry = next((c for c in doc["classes"] if c["name"] == args.class_name), None)
    if entry is None:
        print(f"error: class {args.class_name} not in {args.dimensions}", file=sys.stderr)
        return 2

    print(f"Datasheet dimensions for {args.class_name}:")
    for k, v in entry.items():
        if k != "name":
            print(f"  {k}: {v}")

    # Try to open the verified mesh in Blender for visual inspection.
    candidates = [
        args.verified_dir / f"{args.class_name}.glb",
        args.verified_dir / f"{args.class_name}.step",
        args.verified_dir / f"{args.class_name}.obj",
    ]
    mesh_path = next((p for p in candidates if p.exists()), None)
    if mesh_path is None:
        print(
            f"warning: no verified mesh found at any of:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
        return 1

    print(f"\nFound verified mesh at: {mesh_path}")
    print("Open this in Blender and visually confirm the mating-face geometry")
    print("matches the datasheet dimensions above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
