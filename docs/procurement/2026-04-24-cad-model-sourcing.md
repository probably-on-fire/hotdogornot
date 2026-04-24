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
**Subject:** STEP file request — 2.4/2.92/3.5 mm precision connectors (AIRED R&D)

> Hello,
>
> I'm Chris at AIRED, working on a computer-vision R&D effort to identify
> precision RF connectors from camera imagery. I'd like to request STEP
> files for the following connector families (both plug and jack):
>
> - 2.4 mm precision (plug + jack)
> - 2.92 mm / K Connector series (plug + jack)
> - 3.5 mm precision (plug + jack)
>
> End-launch or panel-mount variants are fine; any accurate parametric STEP
> of the mating-face geometry is what I need. The application is training
> a synthetic-data pipeline, so geometric fidelity at the mating face
> (bore + inner-pin dimensions) is the primary concern.
>
> If you have datasheets or mechanical drawings for the specific parts you
> send, I'd appreciate those as well — they help me verify dimensions
> against the CAD before using it downstream.
>
> Thanks very much,
>
> Chris
> AIRED
> chris@aired.com

Notes:
- Don't oversell — "evaluating for Anduril program" is accurate given the stated context; avoid specifics you can't commit to.
- Ask for datasheets *alongside* STEPs — the datasheet is what you'll use to verify / repair the mating-face geometry in Blender.

## Email template — Rosenberger North America

**To:** info.nam@rosenberger.com (or your regional sales contact)
**Subject:** STEP file request — 02K / 03K / 04K series coaxial connectors (AIRED R&D)

> Hello,
>
> I'm Chris at AIRED, working on a computer-vision R&D effort to identify
> precision RF connectors from camera imagery. I'd like to request STEP
> files for the following Rosenberger series, both plug and jack:
>
> - 04K series (2.4 mm)
> - 02K series (2.92 mm)
> - 03K series (3.5 mm)
>
> Any representative part from each series is fine — I'm primarily concerned
> with accurate mating-face geometry for a synthetic-rendering pipeline,
> so geometric fidelity at the interface matters most.
>
> If you can confirm the series codes above are still correct for these
> sizes in your current catalog, and include datasheets with each STEP,
> that would be helpful.
>
> Thanks,
>
> Chris
> AIRED
> chris@aired.com

Notes:
- Rosenberger's North America team is generally responsive; their K-line product families are standard in RF/microwave R&D.
- "02K / 03K / 04K" are Rosenberger's internal series codes for these sizes. Confirm by checking their catalog before sending.

## Email template — Huber+Suhner

**To:** info@hubersuhner.com
**Subject:** STEP file request — 2.4 mm / 2.92 mm / 3.5 mm precision RF connectors (AIRED R&D)

> Hello,
>
> I'm Chris at AIRED, working on a computer-vision R&D effort to identify
> precision RF connectors from camera imagery. I'd like to request STEP
> CAD files for your 2.4 mm, 2.92 mm, and 3.5 mm precision coaxial connectors
> (both plug and jack). Accurate mating-face geometry is critical for the
> synthetic training data the pipeline generates.
>
> Any representative part from each size family is fine. Datasheets alongside
> the STEPs would also be welcome — I use the dimensioned drawings to
> verify the CAD matches the real part before using it downstream.
>
> Thanks,
>
> Chris
> AIRED
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
