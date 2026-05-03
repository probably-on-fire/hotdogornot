# RF Connector Classifier — Help Wanted

## What we're building

iPhone/Android AR app to identify RF coaxial connectors in the field.
For an Anduril R&D pitch. The app should overlay the connector's class
on a live camera feed.

**Eight target classes** (4 families × 2 genders):
SMA-M, SMA-F, 3.5mm-M, 3.5mm-F, 2.92mm-M, 2.92mm-F, 2.4mm-M, 2.4mm-F.

The four families are precision RF connectors that look very similar:
- **Hex coupling nut** (where the wrench grips): same size on all four
  families per the user's hardware (one wrench fits all).
- **Mating face outer diameter**: 3.6 mm (2.4mm), 4.5 mm (2.92mm),
  5.0 mm (3.5mm), 4.95 mm (SMA). **Sub-mm differences across classes.**
- **Inner aperture diameter**: ~1.04 mm (2.4), 1.27 mm (2.92), 1.5 mm
  (3.5), 1.27 mm (SMA, but with PTFE dielectric vs air).
- **Gender**: M has a copper pin sticking out of the center. F has a
  hollow tube/socket recessed in the center.

SMA and 3.5mm are visually nearly identical except for the dielectric
(PTFE plug vs air gap).

## Architecture today

Two-stage detect-then-classify pipeline:

1. **Detect**: Hough Circles on the frame finds the bright/round
   mating-face circle of each connector. Reliable.
2. **Classify**: tight padded crop (~2.7× face radius) goes to a
   ResNet-18 fine-tuned on labeled crops. Outputs 8-class softmax.

Plus a parallel deterministic geometry pipeline
(`rfconnectorai/measurement/`) that uses hex_detector + aperture_detector
+ aruco_detector + family/gender detectors to compute mm dimensions and
look up the matching class. Designed for use with an ArUco scale marker
in frame.

## Data

**Hard constraint**: only 3 source videos exist. ~10 sec each, shot on
a wood bench under one lighting condition. No SMA video at all. We
extract Hough crops of the connector mating face from these.

After cleanup (manual via a custom labeler we built):
- 296 cleaned originals across 6 classes (no SMA samples)
- Class imbalance: 86 + 40 + 22 + 40 + 73 + 35 (roughly skewed F)
- 8 held-out test images (phone shots, similar wood bench but different
  framing/light/distance from training)

We've expanded synthetically:
- 296 → ~888 with center-cropped zoom variants
- → 1184 with circle-masked variants (gray-noise background)
- → 2664 with 5 procedural backgrounds per crop (solid, gradient,
  noise, stripes, speckle)

## Training pipeline already in place

ResNet-18 fine-tune via torchvision, with:
- ImageNet pretrained init
- WeightedRandomSampler for class balance
- RandomResizedCrop (scale 0.55-1.0), HFlip, ColorJitter
  (brightness 0.25, contrast 0.25, saturation 0.15, hue 0.02),
  RandomRotation 20°, GaussianBlur (p=0.2), small RandomErasing
- Label smoothing 0.1
- AdamW, lr=3e-4, weight_decay=5e-4
- CosineAnnealingLR, 15 epochs
- TTA at inference: avg over [orig, hflip, ±10° rot, 90% center crop]

## What we've tried (held-out: 8 images)

| Run | Train | Val | Held-out Full | Family | Gender |
|---|---|---|---|---|---|
| Baseline | 99.6% | 98.3% | 12% | 25% | 75% |
| Aggressive aug + sampler | 96% | 95% | 0% | 62% | 0% |
| Background masking | 98% | 100% | 12% | 50% | 25% |
| Smoothing + cosine + WD | 99% | 99% | 25% | 38% | 38% |
| Two-head architecture | 99% | 99% | ~12% | ? | ? |
| Two-head + 2664 samples | 100% | 99% | 25% | 38% | 38% |

Pattern across all runs: **near-perfect training/val accuracy
(in-distribution), held-out collapses to ~25%/38%/38% with the model
predicting "3.5mm-F" for 6-8 of the 8 held-out images** at
moderately-confident probabilities (50-90%).

Geometric pipeline is dead in the water — hex detection only fires on
~5/8 held-out images and returns physically-impossible measurements
(hex/face ratio < 1, but hex must be larger than face). Wood-grain
background and small connector-in-frame size defeat contour detection.

## What we haven't tried (and want Codex's input on)

1. **Foundation model features** — DINOv2 or CLIP image encoder + linear
   probe on top, instead of fine-tuning ResNet-18 from scratch.
2. **Zero-shot CLIP classification** — encode "an SMA male connector"
   etc. as text prompts, find best match.
3. **Distillation from a vision-language model** — pseudo-label more
   data with GPT-4V or similar, then train.
4. **Ensemble of weak signals** — face diameter from Hough + center
   brightness for gender + ML logits, combined.
5. **Better data synthesis** — segment connector via Hough mask and
   composite onto **realistic** backgrounds (current procedural ones
   are too unlike real-world).
6. **A fundamentally different architecture** — e.g. metric learning
   (siamese / triplet loss) since each class has only ~50 examples.

## Specific questions for Codex

1. **Given 296 hand-labeled training crops and 8 held-out test images,
   is there a more reliable approach than what we're doing?** Specifically
   for 6 visually-similar precision RF connector classes that differ by
   sub-mm dimensions.

2. **Is there a known technique for monocular sub-mm metrology** that
   doesn't require an ArUco scale reference but works on small objects
   shot on cluttered backgrounds?

3. **Would a foundation vision model (DINOv2, CLIP, SigLIP, SAM)
   meaningfully improve held-out accuracy** in the small-data regime
   compared to ResNet-18 fine-tuning?

4. **Is there a smarter data-synthesis pipeline** that would generate
   training samples actually resembling the held-out distribution
   (different camera, different bench, different lighting)?

5. **What's the highest-ROI experiment we should run next?**

## Code layout

- `rfconnectorai/classifier/` — ResNet-18 trainer + predictor
- `rfconnectorai/measurement/` — geometric pipeline (hex, aperture,
  aruco, family, gender detectors + class_predictor)
- `rfconnectorai/data_fetch/connector_crops.py` — Hough Circle
  connector detector used at extraction + inference time
- `rfconnectorai/server/predict_service.py` — FastAPI inference service
- `rfconnectorai/server/labeler.py` — HTMX-based labeler at
  https://aired.com/rfcai/labeler/ (cleanup tool)
- `data/labeled/embedder/<CLASS>/` — training crops
- `data/test_holdout/<CLASS>/` — 8-image golden held-out set
- `data/videos/` — the 3 source videos
- `data/measure_crops/<FAMILY>/` — wider crops we tried for hex
  measurement (didn't help)
