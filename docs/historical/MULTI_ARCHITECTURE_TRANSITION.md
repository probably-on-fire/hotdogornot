# Multi-Architecture Transition Plan

This plan defines how the project moves from the current ResNet-only
classifier baseline into a multi-architecture object detection and attribute
classification system.

The current ResNet-18 model is not being discarded. It remains the baseline,
fallback, and regression comparison point. The production target is a staged
pipeline where each model family is used for the job it is best suited for.

Diagram source and render outputs:

- `docs/MULTI_ARCHITECTURE_TRANSITION.dot`
- `docs/MULTI_ARCHITECTURE_TRANSITION.svg`
- `docs/MULTI_ARCHITECTURE_TRANSITION.png`

## Why ResNet Alone Is Not Enough

ResNet-18 is a useful image classifier, but the next version of this project
requires more than flat image classification:

- detect connector instances in full camera frames,
- separate multiple connectors in one image,
- reject backgrounds/no-connector images,
- classify several connector attributes independently,
- support geometry and scale reasoning,
- produce uncertainty and abstention states,
- deploy both server-side and eventually on-device.

ResNet can still participate, but it should no longer be the only model.

## Target Model Roles

| Role | Candidate Architectures | Purpose |
|---|---|---|
| Connector detection | YOLO11n, YOLO11s, RT-DETR small | Locate connector instances and separate multi-connector images |
| Fast mobile detection | YOLO11n, MobileNet-SSD fallback | Low-latency edge inference candidates |
| Fine-grained crop classification | EfficientNetV2-S, ConvNeXt-Tiny, MobileViT, MobileNetV3 | Classify family and visual attributes from detected crops |
| Baseline/fallback classification | ResNet-18 | Maintain comparison to current production behavior |
| Attribute output | custom multi-head classifier | Predict family, polarity, gender/contact, mount, orientation, termination |
| Geometry validation | deterministic CV/measurement modules | Thread/diameter/length/aperture reasoning when scale exists |
| Spec validation | taxonomy/spec lookup | Reject impossible or incompatible predictions |
| 3D verification | render/silhouette/edge matching | Optional second-pass verification for top candidates |

## Bake-Off Strategy

Every candidate architecture should be evaluated against the same data,
same holdout split, and same report format.

Minimum comparison set:

```text
ResNet-18 baseline
YOLO11n detector
YOLO11s detector
RT-DETR small detector if practical
EfficientNetV2-S classifier
MobileNetV3 classifier
MobileViT classifier
ConvNeXt-Tiny classifier
```

Evaluation criteria:

- family accuracy,
- gender/contact accuracy,
- polarity accuracy,
- mount/orientation/termination accuracy,
- mAP for detection,
- false positives on no-connector images,
- top-k accuracy,
- macro F1,
- calibration error,
- abstention-aware correctness,
- model size,
- server latency,
- mobile/export latency,
- failure gallery quality.

## Deployment Rule

Do not choose the final model because it is newer or more sophisticated.
Choose it because it wins on the measured contract requirements:

```text
accuracy + reliability + latency + deployability + maintainability
```

Expected final shape:

```text
YOLO-style detector
  + EfficientNet/MobileViT/ConvNeXt multi-head classifier
  + geometry/spec verification
  + ResNet baseline fallback
  + optional 3D render verification
```

## Testing Location

Heavy model tests and training should be run in Kaggle or Colab, not on the
local development PC. Local commits may include code/docs/scripts, but model
training, architecture bake-offs, and expensive validation should be executed
in the cloud and reported back through experiment metrics.
