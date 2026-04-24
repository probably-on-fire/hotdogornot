# Mating-Face Repair Guide

Short workflow for repairing simplified or missing mating-face geometry in
vendor-supplied STEP files. Run once per class when a new STEP is added.

## Workflow

1. **Open the STEP in Blender.**
   - File → Import → STEP (requires Blender 4.x with STEP import enabled),
     or preconvert via FreeCAD: File → Export → glTF 2.0 Binary (.glb).
2. **Switch to mating-face view.**
   - Select the connector, Numpad-1 / Numpad-3 to align camera to the
     mating-plane normal, frontal view.
3. **Compare to datasheet.**
   - Datasheet shows dimensioned drawings (bore ID, pin OD, recess depth).
   - Look for: simplified flat ring where a stepped bore should be; missing
     pin geometry; missing dielectric visible ring (SMA).
4. **Decide: good / minor repair / full rebuild.**
5. **Repair**
   - **Minor:** enter Edit mode, adjust verts/faces.
   - **Full rebuild:** delete the simplified mating face, add a cylinder
     primitive sized to the datasheet bore ID, extrude depth to match,
     add a pin primitive at center.
6. **Re-export as GLB.**
   - File → Export → glTF 2.0 Binary (.glb).
   - Save to `training/data/cad/verified/<class>.glb`.

## Checklist per class

- [ ] STEP imported without errors
- [ ] Mating-face viewed frontally
- [ ] Datasheet dimensions read into `configs/datasheet_dimensions.yaml`
- [ ] Bore ID matches (± 0.05 mm)
- [ ] Pin OD matches (± 0.05 mm)
- [ ] Dielectric ring present where expected (SMA only)
- [ ] Exported to `data/cad/verified/<class>.glb`
- [ ] `configs/cad_sources.yaml` updated to point at the verified mesh

## When to escalate

If a vendor's STEP is fundamentally unusable (e.g. entire connector is a
single solid block with no visible internal geometry), contact the vendor
and request a more detailed model. Many vendors will provide one when
pressed for engineering use.
