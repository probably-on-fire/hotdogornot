# Classifier baseline findings — pre-video (DDG images only)

Date: 2026-04-25
Dataset: 234 images fetched from DuckDuckGo image search across 8 classes
Model: ResNet-18 (ImageNet pretrained), final FC swapped for 8-class output

## Per-class image counts

| Class       | Images |
|-------------|--------|
| SMA-M       | 31     |
| SMA-F       | 30     |
| 3.5mm-M     | 30     |
| 3.5mm-F     | 30     |
| 2.92mm-M    | 30     |
| 2.92mm-F    | 25     |
| 2.4mm-M     | 39     |
| 2.4mm-F     | 30     |
| **Total**   | **245** (incl. 1 chrome_ + 234 ddg_ + 10 ddg_ from earlier test)

## Training results (8 epochs, default config)

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|-----------:|----------:|---------:|--------:|
| 1     |      1.943 |     0.265 |    2.519 |   0.216 |
| 2     |      1.417 |     0.477 |    2.069 |   0.340 |
| 3     |      1.091 |     0.603 |    2.066 |   0.278 |
| 4     |      0.907 |     0.652 |    2.353 |   0.268 |
| 5     |      0.832 |     0.706 |    2.678 |   0.278 |
| 6     |      0.790 |     0.698 |    2.586 |   0.330 |
| 7     |      0.752 |     0.729 |    3.299 |   0.258 |
| 8     |      0.714 |     0.745 |    2.812 |   0.309 |

## Read

- **Final val_acc 30.9%** vs random baseline 12.5% — the classifier is
  learning real signal, but not enough to be useful as a primary predictor.
- **Train loss falls steadily, val loss climbs** — classic overfitting.
  The dataset is too small (~30 images per class) for ResNet-18 to
  generalize without strong regularization.
- **No single epoch is much better than another on val** — early stopping
  wouldn't help much. The model is at the limit of what 30 images per
  class can support.

## Why this is OK as a starting point

- Most of the 234 images are vendor catalog product shots taken at 30-45°
  angles. The 8 classes are visually similar at these angles (especially
  3.5mm vs 2.92mm — nearly identical proportions).
- The measurement pipeline rejects these images (1.2% accuracy in the
  earlier eval) because they're not perpendicular. So the classifier's
  job is genuinely "learn fine-grained metallic-connector class from
  varied product shots" — a known-hard problem for small datasets.
- The accuracy gap will close when we add Monday's video frames (~60
  per class, biased toward perpendicular mating-face captures with
  consistent lighting).

## What to expect with Monday's video

Adding ~60 frames per class (total ~90 per class), with augmentation already
in the pipeline (RandomResizedCrop, ColorJitter, HorizontalFlip):

- **val_acc 60-80%** is realistic
- **Together with the measurement pipeline as a cross-check**, the ensemble
  should hit >95% on perpendicular shots (measurement carries the load) and
  reasonable accuracy on angled shots (classifier carries the load)

## What to do if val_acc is still <60% after Monday

- More data: aim for 100+ images per class
- Stronger augmentation: add RandomRotation, RandomAffine, MixUp
- Bigger backbone: ResNet-50 or EfficientNet-B0 (still fits on CPU)
- Class-balanced sampling if any class is under-represented
- Cross-class confusion matrix → identify which pairs are confused, weight
  the loss to help

These are all incremental moves. The architecture is right; the data is
the bottleneck.
