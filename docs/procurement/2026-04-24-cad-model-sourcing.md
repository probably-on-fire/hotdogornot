# CAD Model Sourcing — RF Connector Training Set

**Date:** 2026-04-24
**Purpose:** Acquire verified STEP files for all 8 initial connector classes. Direct-to-vendor email is the fastest path for the precision connectors (3.5mm / 2.92mm / 2.4mm).

## Target inventory

| Class | Vendor | Source path | Est. difficulty |
|---|---|---|---|
| SMA-M | Amphenol RF | 3D library, no auth | Easy |
| SMA-F | Amphenol RF | 3D library, no auth | Easy |
| 3.5mm-M | Southwest Microwave / Rosenberger | Email request | Medium |
| 3.5mm-F | Southwest Microwave / Rosenberger | Email request | Medium |
| 2.92mm-M | Southwest Microwave ("K connector") | Email request | Medium |
| 2.92mm-F | Southwest Microwave | Email request | Medium |
| 2.4mm-M | Southwest Microwave / Huber+Suhner | Email request | Medium |
| 2.4mm-F | Southwest Microwave / Huber+Suhner | Email request | Medium |

## Workflow

1. **SMA (today):** download from Amphenol RF's public 3D library (no registration for most parts). Both plug and jack for end-launch or PCB-mount variants — geometry of the mating face is the same across mounting styles, so pick whichever STEP looks cleanest.
2. **Precision connectors (this week):** send the email template below to **sales@southwestmicrowave.com**. Copy the same template to Rosenberger North America and Huber+Suhner if you want alternatives. Expect a response in 1–2 business days.
3. **On receipt:** inspect each STEP in Blender or FreeCAD. Verify the mating-face bore + pin per datasheet. See `docs/superpowers/plans/2026-04-24-plan-3-addendum-step-workflow.md` for the verification step.

## Email template — Southwest Microwave

**To:** sales@southwestmicrowave.com
**Subject:** STEP files for 2.4/2.92/3.5 mm precision connectors — evaluating for Anduril program

> Hello,
>
> I'm evaluating your precision coaxial connectors as part of a vision-based
> connector-identification R&D effort for an Anduril program, and I'd like to
> request STEP files for the following connector families (both plug and jack):
>
> - 2.4 mm (Southwest series, typical part family for SMPM-equivalent plug + jack)
> - 2.92 mm (K Connector series — plug + jack)
> - 3.5 mm (plug + jack)
>
> End-launch or panel-mount variants are fine; any accurate parametric STEP
> of the mating-face geometry is what I need. The application is training a
> synthetic-data pipeline, so geometric fidelity at the mating face
> (bore + pin dimensions) is the primary concern.
>
> If you have datasheets or drawings referencing those specific part numbers
> I can include in my evaluation, that would also be helpful.
>
> Thanks very much,
> Chris
> chris@aired.com

Notes:
- Don't oversell — "evaluating for Anduril program" is accurate given the stated context; avoid specifics you can't commit to.
- Ask for datasheets *alongside* STEPs — the datasheet is what you'll use to verify / repair the mating-face geometry in Blender.

## Email template — Rosenberger North America

**To:** info.nam@rosenberger.com (or your regional sales contact)
**Subject:** STEP file request — 02K / 03K / 04K series coaxial connectors (plug + jack)

> Hello,
>
> I'm evaluating Rosenberger precision RF connectors for an R&D program and
> would like to request STEP files for the following series (plug + jack):
>
> - 04K series (2.4 mm)
> - 02K series (2.92 mm)
> - 03K series (3.5 mm)
>
> Any representative part from each series is fine. I'm primarily concerned
> with accurate mating-face geometry for a synthetic-rendering pipeline, so
> geometric fidelity at the interface is what matters most.
>
> Datasheets for each would also be welcome if you can include them.
>
> Thanks,
> Chris
> chris@aired.com

Notes:
- Rosenberger's North America team is generally responsive; their K-line product families are standard in RF/microwave R&D.
- "02K / 03K / 04K" are Rosenberger's internal series codes for these sizes. Confirm by checking their catalog before sending.

## Email template — Huber+Suhner

**To:** info@hubersuhner.com
**Subject:** STEP file request — 2.4 mm / 2.92 mm / 3.5 mm precision RF connectors

> Hello,
>
> I'd like to request STEP CAD files for your 2.4 mm, 2.92 mm, and 3.5 mm
> precision coaxial connectors (both plug and jack). I'm building a
> computer-vision system to identify these connector types in lab imagery,
> and accurate mating-face geometry is critical for synthetic training data.
>
> Any representative part from each size family is fine. Datasheets welcome.
>
> Thanks,
> Chris
> chris@aired.com

## SMA — direct download links

**Note on automation:** Amphenol RF's site (`amphenolrf.com`) is behind Cloudflare and returns HTTP 403 to direct `curl`/`wget` — automated scraping is not viable. Use a browser with normal cookies; the files themselves are free.

Amphenol RF publishes STEPs directly. Search by part number at their 3D model library. Representative parts that are both widely available and have clean STEPs:

- **SMA-M** (plug): part family `901-143` or any `SMA Plug, PCB-mount`
- **SMA-F** (jack): part family `901-9519` or any `SMA Jack, PCB-mount`

Direct browsing: https://www.amphenolrf.com/3d-models/

GrabCAD is a secondary source: https://grabcad.com/library/tag/sma-connector

## After acquisition — verification checklist

For each connector after download or receipt:

- [ ] STEP file opens cleanly in Blender (File → Import → STEP, or via FreeCAD preconversion)
- [ ] Mating face is the correct end (some parts import flipped — orient so mating face faces +Z)
- [ ] Bore inner diameter matches datasheet within 0.1 mm (measure in Blender: select bore edge loop, N-panel "Mesh Display → Edge Info → Length")
- [ ] Inner pin (male) or socket (female) is present and sized per datasheet
- [ ] Dielectric material (PTFE) is visible for SMA, absent for precision connectors
- [ ] Save the datasheet PDF next to the STEP in `training/data/cad/datasheets/`
- [ ] Update `configs/datasheet_dimensions.yaml` with the verified per-class dimensions
- [ ] Export to GLB for reliable bpy import at render time

Estimated time per connector: 10–15 minutes. Full inventory: ~2 hours.

## Legal / conduct notes

- **Use vendor-published models under their stated terms.** Most publish STEPs for design-in use; redistribution requires written permission. For training data, the model is used internally to produce a rendered derivative, which is almost always fine. Keep a copy of the download page / email grant for each file as provenance.
- **Don't claim more than is true in the emails.** "Evaluating for an Anduril program" is accurate if that's your stated context; don't promise procurement commitments you can't back.
- **Thank the vendor** when they respond. Even if you don't end up using their products, the connector industry is small — goodwill compounds.
