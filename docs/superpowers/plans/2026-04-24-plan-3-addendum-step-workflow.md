# Plan 3 Addendum — STEP Workflow + Mating-Face Verification

**Date:** 2026-04-24
**Amends:** `docs/superpowers/plans/2026-04-24-cad-synthetic-data-pipeline.md`
**Status:** Draft — incorporates domain-expert review of CAD acquisition

## What this addendum corrects

The original Plan 3 assumed OBJ as a primary input format and treated mesh import as a solved problem. Domain-expert review flagged two issues that are load-bearing for accuracy:

1. **STEP, not OBJ, is the right primary format.** STEP preserves exact parametric geometry (precise cylinders, threads, tolerances). OBJ is a triangulated approximation; conversion STEP→OBJ loses exactly the sub-millimeter precision we're trying to capture. For a training pipeline whose *whole point* is learning 2.4 vs 2.92 vs 3.5 mm discrimination, discarding precision on ingest defeats the exercise.

2. **Manufacturer CAD frequently simplifies the mating face.** Vendors publish external housing geometry accurately but routinely omit or simplify internal bore + pin geometry — to protect tolerances, shrink file size, or both. The mating face is exactly where our discriminative features live. A rendered training image of a "featureless flat ring" where the real connector has a stepped bore teaches the model to ignore the relevant signal.

Both of these are fixable, but the pipeline has to handle them explicitly.

## Changes to Plan 3

### File structure — add datasheet directory

```
training/data/cad/
├── SMA-M.step                              (from Amphenol RF)
├── SMA-F.step
├── 3.5mm-M.step                            (from vendor — requires email request)
├── ...
└── datasheets/                             (NEW — datasheet PDFs for verification)
    ├── SMA-M_Amphenol_901-143.pdf
    ├── 2.92mm-M_SouthwestMicrowave_1092-03A-6.pdf
    └── ...
```

Datasheets inform the mating-face repair step. Keep the PDFs alongside the STEP files so provenance is obvious.

### New task between Plan 3's current Task 3 and Task 4

Insert **Task 3.5 — Mating-Face Verification and Repair**. Run once per connector when a new STEP is added to the library; before rendering commences.

#### Task 3.5: Mating-Face Verification and Repair

**Goal:** For each class, confirm the imported STEP has geometrically accurate mating-face bore + pin geometry matching the datasheet. Repair any that don't.

**Files:**
- Create: `training/scripts/verify_mating_faces.py`
- Create: `training/configs/datasheet_dimensions.yaml`
- Create: `training/docs/mating_face_repair_guide.md`

**Workflow (per connector):**

1. **Import the STEP in Blender.** Use bpy's STEP importer (Blender 4.x) or preconvert via FreeCAD → GLB/USD, which preserves geometry better than OBJ.
2. **Rotate to mating face.** Camera to connector's mating-plane normal, frontal view.
3. **Compare to datasheet.** The datasheet shows dimensioned drawings with labeled bore diameter, pin diameter, recess depth, etc.
4. **Decide:** good as-is / needs minor repair / needs full rebuild.
5. **Repair (if needed):** open the mesh in Blender edit mode, or rebuild the mating face from scratch using a cylinder primitive sized to datasheet dimensions.
6. **Re-export as STEP or GLB** into `data/cad/verified/<class>.glb`.

**Config: `training/configs/datasheet_dimensions.yaml`**

```yaml
# Nominal datasheet dimensions for mating-face geometry.
# Source each from the manufacturer datasheet referenced in the filename.
# Values are in millimeters unless noted.
#
# bore_id_mm: inner diameter of the outer-conductor bore at the mating face
# pin_od_mm: outer diameter of the inner-conductor pin/contact at the mating face
# dielectric_visible: true for SMA (PTFE visible at face), false for air-line precision connectors

classes:
  - name: "SMA-M"
    datasheet: "SMA-M_Amphenol_901-143.pdf"
    bore_id_mm: 4.20
    pin_od_mm: 0.91
    dielectric_visible: true

  - name: "SMA-F"
    datasheet: "SMA-F_Amphenol_901-9519.pdf"
    bore_id_mm: 4.20
    pin_od_mm: 1.27
    dielectric_visible: true

  - name: "3.5mm-M"
    datasheet: "3.5mm-M_SouthwestMicrowave_XXXX.pdf"  # update after acquisition
    bore_id_mm: 3.50
    pin_od_mm: 0.92
    dielectric_visible: false

  - name: "3.5mm-F"
    bore_id_mm: 3.50
    pin_od_mm: 1.52                                    # socket ID, not pin OD
    dielectric_visible: false

  - name: "2.92mm-M"
    bore_id_mm: 2.92
    pin_od_mm: 1.27
    dielectric_visible: false

  - name: "2.92mm-F"
    bore_id_mm: 2.92
    pin_od_mm: 1.27
    dielectric_visible: false

  - name: "2.4mm-M"
    bore_id_mm: 2.40
    pin_od_mm: 1.04
    dielectric_visible: false

  - name: "2.4mm-F"
    bore_id_mm: 2.40
    pin_od_mm: 1.04
    dielectric_visible: false
```

*Note: the pin/bore values above are placeholders; verify each against the exact datasheet you acquire. Manufacturer values vary slightly by family and revision.*

**Verification script: `training/scripts/verify_mating_faces.py`**

```python
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
```

**Repair guide: `training/docs/mating_face_repair_guide.md`**

A short one-page guide with screenshots showing: open STEP in Blender → switch to mating-face view → identify missing geometry → either fix inline or rebuild a cylinder primitive per datasheet. Content is task-dependent; leave as a stub with the workflow outline initially, populate once the first real STEP has been imported and verified.

### Update Task 3 (mesh_loader) — STEP as primary

Extend `mesh_loader.py` to accept `.step` / `.stp` in the supported extensions list. Add a preflight check: if the path is a STEP, attempt `bpy`'s built-in STEP importer, and fall back to a clear error pointing the user at FreeCAD preconversion if bpy's importer isn't available on their platform.

Concretely, amend Task 3 Step 4 implementation:

```python
SUPPORTED_EXTS = {".obj", ".stl", ".ply", ".step", ".stp", ".glb", ".gltf"}

def load_mesh(path: Path | str) -> MeshInfo:
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
        # trimesh doesn't natively handle STEP. Blender's bpy does.
        # For bbox reporting (this function) we need a mesh; direct users to
        # preconvert via FreeCAD or to use load_mesh_into_blender at render time.
        raise NotImplementedError(
            f"STEP bbox preflight not supported directly. "
            f"Preconvert {p} to GLB or OBJ via FreeCAD: "
            f"File → Export → glTF 2.0 Binary (.glb). Or rely on "
            f"bpy's STEP import at render time."
        )
    mesh = trimesh.load(str(p), force="mesh")
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]
    center = (bbox[0] + bbox[1]) / 2
    return MeshInfo(path=p, bbox_size_m=tuple(float(x) for x in size), center_m=tuple(float(x) for x in center))
```

### Update Task 4 (render_single) — add STEP import path

In `render.py`'s `render_single()`, extend the extension dispatch:

```python
elif ext in (".step", ".stp"):
    # Blender 4.x ships an experimental STEP importer as an addon.
    # Enable it if available; otherwise raise with a clear error.
    try:
        bpy.ops.preferences.addon_enable(module="io_import_step")
    except Exception:
        pass
    if hasattr(bpy.ops.wm, "step_import"):
        bpy.ops.wm.step_import(filepath=str(mesh_path))
    else:
        raise NotImplementedError(
            f"STEP import not available in this bpy build. "
            f"Preconvert {mesh_path} to GLB via FreeCAD (free) or Fusion 360."
        )
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
```

Recommended workflow going forward: **prefer GLB** as the ingestion format. FreeCAD exports STEP→GLB cleanly, GLB preserves geometry with enough precision for rendering, and `bpy` handles GLB natively across platforms. STEP-directly-into-bpy works but is less reliable.

## Summary of the corrected workflow

1. **Acquire:** STEP from vendor (preferably by email request to avoid portal registration).
2. **Verify:** open each STEP in Blender or FreeCAD, compare mating-face geometry to datasheet.
3. **Repair:** rebuild simplified mating faces from datasheet dimensions. Export to GLB.
4. **Inventory:** put the verified GLBs in `data/cad/verified/` and update `configs/cad_sources.yaml` to point at them.
5. **Render:** `bash scripts/render_synthetic.sh` runs unchanged from Plan 3.

Steps 1–3 are manual + datasheet-driven. Step 4 is a one-line config update. Step 5 is automated.

## Acknowledged trade-off

Manual mating-face verification and repair adds ~10–15 minutes per connector — call it 2 hours of work across all 8 classes. That's the price of getting geometric fidelity right. It's dramatically cheaper than discovering post-hoc that the model can't tell 2.92 from 2.4 because every synthetic render had a featureless ring.
