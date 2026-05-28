# docs/historical/

Earlier-phase planning documents and an outdated architecture poster. Preserved here for context; **do not treat as current**. The README at the repo root is the source of truth for what's actually in production.

## What's in here and why it was moved

| File | What it was | Why historical |
|---|---|---|
| `IMPLEMENTATION_PLAN.md` | Authoritative product/architecture roadmap (2026-05-10) | Pre-dates combined_v2 deploy (2026-05-25), combined_v5/v7 work, and the realistic-distance discovery. The plan was followed in pieces but the project arc diverged. |
| `TASKS.md` | Epic-by-epic implementation backlog (2026-05-10) | Same vintage as the plan. Many items completed in different form; many superseded by the data-side fix. |
| `REPO_AUDIT.md` | Repository audit + safety baseline | Audit of an earlier state of the repo. Current state differs. |
| `MULTI_ARCHITECTURE_TRANSITION.md` (+ `.dot/.svg/.png`) | Plan for evolving from ResNet-only to detector + multi-head classifier | Production went to YOLO + EfficientNetV2-S single-head (not multi-head). The transition diagram describes a path that diverged. |
| `ACCEPTANCE_GATES.md` | Per-batch G0-G5 acceptance gates | Internal execution-gating language from the earlier planning phase. |
| `CLIENT_DEMO_README.md` | Entry point for an earlier client-facing demo | Demo style no longer used. |
| `DEMO_SCRIPT.md` | 5-10 minute walkthrough script for the above demo | Same reason. |
| `LIMITATIONS_AND_NEXT_STEPS.md` | Honest limitations + roadmap (earlier era) | Limitations list and roadmap don't match current state; superseded by the README's Production Status + Challenge sections. |
| `Readme_System_Architecture.png` | Earlier marketing-style architecture poster | Shows "Current Baseline" as Hough + ResNet-18 (production has been YOLO + EffNet since 2026-05-25) and lists BNC/TNC/SMB/MCX/etc. families that aren't trained. Replaced in the README by `docs/README_ARCHITECTURE.png`. |

## What's NOT here (still current, still in `docs/`)

- `docs/capture_protocol_distance_2026-05-28.md` — the 2026-05-28 realistic-distance capture protocol
- `docs/CONNECTOR_TAXONOMY.md` — connector family taxonomy
- `docs/MODEL_TRAINING_PIPELINE_SPEC.md` — training pipeline spec
- `docs/MODEL_CARD_TEMPLATE.md` — model card template (still useful for promoted artifacts)
- `docs/ANNOTATION_PROTOCOL.md` — human-labeling rulebook
- `docs/DIAGRAM_RENDERING.md` — how to regenerate Graphviz diagrams
- `docs/README_ARCHITECTURE.{dot,png,svg}` — current production-pipeline diagram (the one embedded at the top of README.md)
- `docs/SYSTEM_ARCHITECTURE_POSTER.{dot,svg,png}` — full system poster (kept; may also need updating eventually)
- `docs/SOFTWARE_ARCHITECTURE.{dot,svg,png}` — software architecture diagram (similar caveat)
- `docs/README_TECHNICAL_OVERVIEW.{dot,svg,png}` — high-level technical overview diagram
- `docs/printables/`, `docs/procurement/`, `docs/superpowers/` — markers, sourcing, completed-work history
