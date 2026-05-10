# Annotation Protocol

This protocol is the human-labeling rulebook for the RF connector
identification dataset. Folder names alone are not enough for this project.
Every labeled example must follow these rules so that downstream training,
evaluation, and abstention behavior stay coherent.

## Labeling Unit

One row equals one visible connector instance, not one source image.

A single source image may produce zero, one, or many instance rows. A
multi-connector image must be split into multiple instance rows that each
preserve the source image path and a per-instance bounding box.

## Required Labels

Every instance row must populate the following fields. Use `unknown`,
`not_applicable`, or `insufficient_view` rather than guessing silently.

- `family`
- `precision_family`
- `side_a_gender`
- `side_b_gender`
- `polarity`
- `mount_style`
- `orientation`
- `termination`
- `finish_material_cue`
- `bbox_xyxy`
- `label_confidence`

Adapters must additionally populate the nested `side_a` / `side_b` blocks
defined in the instance schema.

## Label Confidence

Every row records exactly one `label_confidence` value:

- `human_verified` - a human directly inspected the image and applied the labels.
- `weak_folder_label` - labels inferred from folder/file naming only.
- `synthetic_verified` - labels derived from a parametric synthetic render.
- `model_suggested` - labels proposed by a model and not yet verified.
- `unknown` - labels insufficient or pending review.

Weak labels must remain marked weak; do not promote a `weak_folder_label`
row to `human_verified` without an explicit human review pass.

## Adapter Rules

Two-sided adapters must be labeled on both sides. A single-family label is
not acceptable for an adapter.

For adapters, populate at minimum:

- `side_a_family`
- `side_b_family`
- `side_a_gender`
- `side_b_gender`
- `side_a_polarity`
- `side_b_polarity`

If a side is occluded or impossible to disambiguate from the available
view, mark it `insufficient_view` and request a second angle in the
contributor flow. Do not invent the missing side.

## Unknown Rules

Use `unknown`, `not_applicable`, or `insufficient_view` rather than
guessing silently. These values are first-class outcomes throughout the
pipeline. Forced wrong guesses are unacceptable.

- `unknown` - the attribute exists but cannot be determined from this view.
- `not_applicable` - the attribute does not apply to this connector type.
- `insufficient_view` - more visual evidence is needed before this attribute can be labeled.

## Holdout Rules

Real phone holdout images are never used for:

- training
- synthetic tuning
- prompt tuning
- model selection

Holdout integrity is protected by Epic 4 split-by-specimen rules and
Epic 9 reporting. Any leakage between training data and the phone-photo
holdout invalidates accuracy claims and must be reported as a blocker.

## Operational Notes

- Originals are never modified or moved. Crops live alongside originals with
  full source-image lineage in the instance manifest.
- Synthetic-vs-real provenance must be recorded so reports can separate
  real-data accuracy from synthetic-aided accuracy.
- New annotators should review this protocol before labeling and re-read
  it whenever a new attribute or family is added to the taxonomy.
