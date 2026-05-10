# Limitations And Next Steps

The connector identification system is real, working, and honest about
what it does not yet do. Read this before showing the demo to anyone.

## Current Limitations

### Holdout Is Small

The current real-phone holdout is approximately 8 images. One miss
shifts accuracy by ~12.5 percentage points. Any "X% accuracy" claim
based on this holdout should be qualified.

### ResNet-18 Is The Baseline, Not The Production Model

The current production endpoint runs an ImageNet-pretrained ResNet-18
with a linear head. It is the baseline / fallback. The production target
is a detector + multi-head classifier + geometry/spec verification
pipeline, described in
[`MULTI_ARCHITECTURE_TRANSITION.md`](MULTI_ARCHITECTURE_TRANSITION.md).

### Geometry Without Scale Is Estimation Only

The system can flag thread diameter, hex size, body length, and aperture,
but only when there is a calibrated scale reference in frame
(``requires_calibrated_reference: true``). Without a scale marker, the
geometry block is left null and the response state may be
``need_scale_reference``.

### Two-Sided Adapters Are Hard

Adapters require both sides to be observable. When a side is occluded,
the response is ``insufficient_view`` for that side, not a guess. This is
correct behavior but means many adapter shots will request a second angle.

### Synthetic Data Helps But Does Not Replace Real Data

The synthetic render suite covers many families and configurations, but
synthetic accuracy is not real accuracy. Real-phone holdout accuracy is
the gate, not synthetic accuracy.

## Roadmap (Honest)

The acceptance gates in [`ACCEPTANCE_GATES.md`](ACCEPTANCE_GATES.md)
walk from current state to demo readiness.

### Next Batch (Batch 2)

- Dataset audit (`docs/DATASET_AUDIT.md`).
- Instance manifest validation against
  ``training/rfconnectorai/schemas/instance.py``.

### Next Gate (G2)

- Connector instance catalog and crop workflow.
- Synthetic-vs-real provenance for every label.

### Production Path

1. Detector training (Kaggle/Colab) with ``yolo11n`` baseline (Epic 5).
2. Multi-head classifier training (Epic 6) with EfficientNetV2-S baseline
   and a head-by-head bake-off.
3. Geometry/measurement integration (Epic 8).
4. API upgrade to the structured response defined in
   ``training/rfconnectorai/schemas/prediction.py`` (Epic 10).
5. Flutter UX upgrade for confidence/ambiguity states (Epic 11).
6. Mobile/server export (Epic 12, see
   [`exports/mobile/README.md`](../exports/mobile/README.md)).

## What Will Not Be Claimed

- 99.99% accuracy without a large, diverse, independently held-out
  validation set.
- "Always identifies the right connector" — the system explicitly
  abstains when the view is insufficient.
- Mobile-on-device inference until a per-device latency / thermal
  benchmark is documented under ``reports/experiments/<run>/``.
