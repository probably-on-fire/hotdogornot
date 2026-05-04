# Connector Classifier — Project Journey & Findings

This doc captures the iteration history, key findings, and current state
of the RF connector classifier. Read this before making more changes.

## Goal

iPhone/Android (eventually) classifier for 8 RF connector classes:
SMA / 3.5mm / 2.92mm / 2.4mm × M / F. Supports a take-a-photo demo
plus a continuous-learning loop where phone uploads grow the dataset.

## Hard data constraint

- **3 source videos**, ~10s each, all on the same wood bench under one
  lighting condition. Filenames: `2_4mm.MOV`, `2_92mm.MOV`, `3_5mm.MOV`.
- **No SMA video** at all — the SMA classes have zero training samples
  by definition.
- **8 phone-shot held-out images** in `data/test_holdout/` for evaluation.
- Source videos are 1920×1080, held-out photos are 4032×3024 (12 MP).

## Two critical labeling/data findings

### 1. Labeling convention swap (2026-05-02)

Originally the user labeled M/F by **plumbing convention**: external
threads visible on the body = "M" (like a male pipe end), smooth/hex
outside = "F". This is **opposite** of standard RF connector convention,
which goes by the contact-end pin/socket: pin = M, socket = F. Plumbing
M = RF F and vice versa.

We swapped all `<family>-M/` ↔ `<family>-F/` folders to align with RF
spec. Backups at `/home/rfcai/backups/embedder_pre_swap_*.tar.gz` and
mirrored to local `backups/`.

**Going forward: M = pin side, F = socket side.** Click that way in the
labeler when reviewing or uploading new samples.

### 2. Resolution mismatch (2026-05-02)

Training crops are 850×850 saved, but the *real* signal is the
50–100px Hough crop from the original 1920×1080 video frame, *upscaled*.
The central pin/hole feature in the original video is ~5–10 pixels —
**below the visual resolution at which gender can be reliably
classified**. Held-out photos at 4032×3024 have the central feature at
~20–50 pixels, which IS distinguishable.

This explains the entire gender accuracy oscillation across all our
experiments: there is no reliable gender feature in the training data
to learn. **Family is partly learnable from videos; gender requires
photo-resolution training data.**

## Architecture

Two-stage detect-then-classify, plus a parallel deterministic
geometry pipeline:

```
Phone frame
   ↓
Hough Circle (rfconnectorai/data_fetch/connector_crops.py)
   ↓
Tight padded crop (~2.7× face radius)
   ↓
ResNet-18 fine-tune (rfconnectorai/classifier/predict.py)
   → 8-class softmax
```

Best variant: **two-classifier split** — separate family classifier
on rembg-clean full crops, separate gender classifier on tight central
crops (script: `scripts/_exp_gender_v2.py`).

Secondary path (built but not currently usable on phone shots):
geometric pipeline using hex / aperture / family / gender detectors
plus optional ArUco for absolute scale (`rfconnectorai/measurement/`).

## Experiments tried (held-out: 8 images, ±12% per-prediction noise)

Note: gender numbers below were measured against the OLD plumbing
labels. After the 2026-05-02 swap they're effectively inverted (75% →
25% for the same physical predictions). Family numbers are unaffected.

| Run | Train | Held-out Full | Family | Gender (old labels) |
|---|---|---|---|---|
| Baseline ResNet-18 | 99.6% | 12% | 25% | 75% |
| + aggressive aug + class-balanced sampler | 96% | 0% | **62%** | 0% |
| + label smoothing + cosine LR + WD | 99% | 25% | 38% | 38% |
| + zoom synth (888 samples) | 98% | 12% | 38% | 50% |
| + bg-mask synth (1184 samples) | 98% | 12% | 50% | 25% |
| Two-head architecture | 99% | 25% | 38% | 38% |
| Two-head + 2664 synth-bg samples | 100% | 25% | 38% | 38% |
| rembg-clean train + rembg eval | 100% | 12% | 50% | 25% |
| Two-classifier (resnet+resnet, central crop) | 99% | 25% | 62% | 50% |
| **Two-classifier v2 (mobilenet-gen + 8x TTA)** | 99% | **38%** | 38% | **75%** |
| DINOv2 linear probe + gender v2 | 100% | 25% | 38% | 62% |
| DINOv2 + linear probe (3-class, no SMA) | 100% | 25% | 38% | 62% |

## 2026-05-04 training-data quality audit

A look at actual training crops uncovered a major issue: many "training
crops" are not tight crops on a connector — they're 1080×1500 wide
shots of a wood desk with a connector tiny in one corner. The original
edge-density extractor (`detect_connector_crops`, `pad_frac=0.35`) can
return wide bounding rects that include multiple connectors or texture
regions, producing crops where the connector is < 5% of the frame.

The model had been "learning" 2.4mm-M = wood-grain pattern at the
bottom of a 1080×1200 image. No wonder it doesn't generalize to
held-out phone shots that look very different.

`scripts/audit_training_quality.py` runs the same rembg fg filter
the predict service uses, but with a relaxed `--min-fg 0.02` (vs the
0.05 inference threshold) so it kills only crops with essentially no
foreground silhouette while keeping moderate-zoom captures. Also
groups bases with their rembg-derived variants (`_clean`, `_bg0-4`,
`_central`, etc.) so a quarantined base takes its variants with it.

Result on the 4042-file pre-clean dataset: **2934 files quarantined
(72%)** moved to `data/labeled/_quarantine_lowq/`. Per-class survivors:

| Class | Pre | Post |
|-------|-----|------|
| 2.4mm-F | 877 | 108 |
| 2.4mm-M | 575 | 27 |
| 2.92mm-F | 272 | 56 |
| 2.92mm-M | 628 | 63 |
| 3.5mm-F | 484 | 267 |
| 3.5mm-M | 1206 | 587 |

3.5mm classes lost the least (~50% kept), 2.4mm-M lost the most
(95% killed). The 2.4mm source video must have been captured in a
way that consistently produced wide bbox crops with the connector
small. **This is a source-video quality issue: the data extracted
from `2_4mm.MOV` is mostly junk no matter how we slice it.** Fix
requires re-shooting that video with the connector filling more of
the frame.

## 2026-05-04 retrain trials (v5 → v8, after pipeline-bug fixes)

Four back-to-back single-seed retrains on the box's full data
(3544 images across 6 populated classes; SMA-M/SMA-F still at 0
samples and dropped per the new auto_retrain logic). val_acc 97-98%
with the new dHash-grouped split.

| Model | aug   | WRS cap | Class balance | Full | Family | Gender |
|-------|-------|---------|---------------|------|--------|--------|
| v5    | mild  | 5 (no-op bug) | counts as-is | 25% | 25% | 62.5% |
| v6    | heavy | 2.0 (real cap)| counts as-is | 25% | 25% | **87.5%** |
| v7    | heavy | 10 (off)      | counts as-is | 12.5% | 12.5% | 62.5% |
| v8    | heavy | 10 (off)      | **subsampled to 261/class** | **37.5%** | **37.5%** | 75% |
| v9    | heavy | 10 (off)      | rembg-cleaned, no subsample | 25% | 25% | 75% |
| v10   | heavy | 10 (off)      | rembg-cleaned + subsampled to 27/class | 25% | 25% | 62.5% |

v9 (cleaned, no subsample) and v10 (cleaned + subsample) both
underperformed v8. The cleaning step at `--min-fg 0.02` quarantined
72% of training data — more importantly, it amplified the 4× class
imbalance into a 22× imbalance because the 2.4mm source video had
worse framing than the 3.5mm source video. v9 then learned to
predict 3.5mm 7/8 of the time. v10 with subsample-to-smallest=27
didn't have enough data per class to learn anything generalizable.

**Lesson: data quality cleanup helps only when the resulting per-class
counts are still adequate, AND when cleanup affects all classes
equally.** Our cleanup is gated by source-video framing quality, which
varies wildly between the 3 source videos. Result: cleaning
worsened the imbalance instead of fixing the bias. v8 (uncleaned +
subsampled to 261) remains best.

**Held-out is 8 images — single-correct = 12.5 percentage points,
so Full/Family deltas are within noise.** Gender 7/8 → 5/8 across v6
and v7 with the same augmentation hints that the WRS cap=2 in v6
may have helped by acting as a regularizer (reduces minority-class
oversampling, model learns more general gender features). But
single-trial variance makes this hard to confirm without a 5-seed
ensemble re-run.

**Decision: v8 promoted as production model** (best held-out
overall — first to break the 25% Family ceiling). Subsampling each
class to 261 (the smallest's count) gave us 1566 training samples
from 3544 raw, but eliminated the count-driven 3.5mm bias. The
deployed model is `weights.0007.pt` containing the v8 balanced
training run.

The source code in master keeps `balance_to_smallest=False` as the
default — explicit `--balance-to-smallest` on auto_retrain (or the
`_kick_retrain.sh` helper which now passes it) opts in. Old weights
preserved on disk: weights.0004.pt (pre-fixes), weights.0006.pt
(v6 = best gender trial), v5/v7 overwritten by v8.

**Re-extracting at higher fps was considered and rejected.** Only
3 source videos exist on the box (`2_4mm.MOV`, `2_92mm.MOV`,
`3_5mm.MOV`). Extracting at fps=12 instead of fps=4 would 3× the
frame count, but adjacent frames at fps=12 are 80ms apart vs 250ms
— dHash clustering would lump them together as the same scene, so
the grouped split puts them all on one side of train/val and they
add ~zero generalization signal. The source-video diversity is the
real ceiling, and the only path to fix it is more videos in
varied environments via the contribute screen.

**3.5mm bias is data-bound, not pipeline-bound.** All three trials
predicted 3.5mm 4-6 of 8 times. The training set has 1032 unique
3.5mm-M images vs 261 for 2.92mm-F (4× imbalance), and the model
learns the variety of 3.5mm features better. WRS rebalancing helps
per-batch class distribution but can't synthesize new 3.5mm-distinct
features from a small minority set. Real fix: more phone shots in
varied conditions for the minority families, contributed via the
Flutter app's contribute screen over time.

## ⚠ The "data ceiling" claim is suspect — likely measurement bugs

A 2026-05-03 code review surfaced three training-pipeline bugs that
together undermine the basis for calling the held-out plateau a
"data ceiling":

1. **Train/val split leaks adjacent video frames** (was `train.py:81-86`,
   now fixed). Frames extracted at fps=4 from short videos are visually
   near-identical (~250ms apart). The old random index split put frame
   N in train and frame N+1 in val, so the 97.7% val_acc on
   `models/connector_classifier/version.json` measured memorization,
   not generalization. The held-out gap of 38% wasn't a data ceiling —
   it was the only honest metric we had. **Fix shipped: dHash-grouped
   stratified split (`_grouped_stratified_split`).** Retrain to get a
   real val_acc number.
2. **3 of 8 classes have zero training data** but still occupied output
   slots (SMA-M, SMA-F, 2.4mm-F). Label-smoothing at 0.1 distributed
   uniform mass to those slots every batch, leaking gradient into
   classes the model could never predict correctly. On any noisy crop
   the model could still emit "SMA-M @ 0.30" purely from this
   dynamic. **Fix shipped: `auto_retrain._populated_classes` drops
   classes with < MIN_SAMPLES_PER_CLASS=5.**
3. **`WeightedRandomSampler` over-sampled minority classes** by 25-30×
   (3-image class at weight 1/3 = 0.33 vs majority at 1/80 = 0.0125),
   producing batches dominated by augmentation noise on the same 2-3
   crops. **Fix shipped: `max_oversample_ratio=5.0` cap in `TrainConfig`.**

After the next retrain, the held-out numbers below should be
re-measured. The plateau may move significantly. Until then, treat the
"failure modes" list as findings against a broken yardstick.

## Failure modes proven exhausted

These were tried and don't work on this data:

- **Hex polygon detection** (geometric): fails on wood-grain background;
  Otsu lumps connector + grain together. 0/8 held-out detections.
- **GrabCut body segmentation** (geometric): 70–97% failure rate; can't
  separate metallic gray from light wood.
- **Wide-pad re-extraction for hex measurement**: even more wood grain
  in frame → noise dominates.
- **Thread-pitch FFT** for absolute scale: fires on every image but the
  detected ppm is 2× wrong — FFT locks onto reflections / sensor
  patterns instead of the threaded barrel (which isn't actually
  visible in face-on shots).
- **Body/bore diameter ratio** (geometric): scaffolded in
  `scripts/_exp_barrel_scale.py`. The pitch was: rembg silhouette →
  body OD circle, then darkness-thresholded central blob → bore. Take
  the ratio, look up against datasheet `bore_id_mm / body_od_mm`
  (SMA=0.66, 3.5mm=0.55, 2.92mm=0.46, 2.4mm=0.47). **Result: 0/8 on
  held-out.** Two reasons it's stuck:
    1. **The datasheet `bore_id_mm` is the inner-conductor bore** —
       a small feature buried inside the chamfered mating-face cavity.
       What we can detect from a face-on photo is the *outer* dark
       recess (the entire chamfered cavity), which is geometrically a
       different, larger ring than the bore the datasheet specifies.
       Measured ratios came in at 0.5–0.73 against expected 0.46–0.66.
    2. Even if we ignored the datasheet and used empirical signatures,
       2.4mm and 2.92mm produce nearly identical visible-recess ratios
       (~0.6 each) — same near-indistinguishability the trained
       classifier hits, for the same physical reason (the
       discriminating features are sub-resolution).
  Scraped product shots in `data/archive/scraped/*` are side-view
  marketing photography, not face-on, so they can't be used to build
  per-class empirical distributions either.
- **DINOv2 linear probe**: 100% in-distribution val but same held-out
  plateau as everything else — feature distribution shift dominates.
- **Procedural background composite synthesis** (5 backgrounds per
  crop): didn't break OOD generalization because procedural patterns
  don't look like real bench/desk backgrounds.

## What works

- **`rembg`** (U²-Net) as a foreground-segmentation tool. Clean
  silhouettes on cluttered backgrounds. Used for background-removed
  training and for inference-time preprocessing.
- **`scripts/pages/1_Training_Data.py`** Streamlit labeler with
  filters (only-disagreements, only-multi-object, only-no-circle,
  hide-near-duplicates, blur threshold), per-tile delete + flip,
  Focus mode with hotkeys.
- **`rfconnectorai/server/labeler.py`** FastAPI + HTMX labeler at
  `https://aired.com/rfcai/labeler/` (admin / 663800c2f2a8f2c4e33f2c43)
  with held-out upload, training-video upload, multi-class filter,
  bulk-delete, image lightbox.
- **VLM upper-bound test**: Claude classifies the 8 held-out photos
  at 87.5% (would be 8/8 except one image is mislabeled — `IMG_0274`
  is byte-identical to `2_4mm-m.jpeg` but lives in different folder).
  This proves the task is *doable from the held-out resolution*; the
  ResNet-18 fine-tune just can't learn it from the video crops alone.
- **rembg foreground pre-filter** (`predict_service._crop_passes_fg_filter`):
  the trained classifier has no "background" class, so on any frame
  without a real connector the softmax still confidently picks one of
  the 8 classes (typically 2.92mm-M @ 0.85+). The fix runs each Hough
  crop through rembg and computes (a) foreground fraction and (b)
  inner-vs-outer-density ratio. Real connector crops show one of two
  patterns: silhouette filling the tight crop edge-to-edge (low
  ratio) OR a small centered object (very high ratio). Wood-texture
  false positives that fool rembg consistently land in the middle
  ratio range (2-4) and get rejected. Local benchmark: 8/8 held-out
  connectors pass, 0/3 background scenarios (wood, noise, solid)
  produce false detections. Tunable via env vars on the box; set
  `RFCAI_FG_FILTER=0` to disable.

## Known data-quality issues

- **`IMG_0274.jpeg` and `2_4mm-m.jpeg` are byte-identical** but live
  in different held-out folders — one of the labels is wrong, so the
  practical held-out ceiling is 7/8 by construction. Should be deleted
  from one folder.
- After the M/F swap, gender labels in the held-out folders may need
  re-verification by a human who can physically inspect the actual
  connector each photo represents.

## Current best model

Two-classifier setup (script: `scripts/_exp_gender_v2.py`):
- Family classifier: ResNet-18 fine-tune on rembg-clean crops
- Gender classifier: MobileNetV3-Small on tight rembg-cleaned central
  crops (96×96)
- Heavy 8× TTA at inference (4 rotations × 2 flips)
- ~38% Full / 38% Family / 75% Gender (pre-swap labels)

Live at `aired.com/rfcai/predict` via the predict service. Demo at
`aired.com/demo/`. Labeler at `aired.com/rfcai/labeler/`.

## Tooling built (reusable regardless of model)

- `rembg` integration for foreground segmentation
- HTMX-based labeler (faster than Streamlit, no scroll-jump)
- Backup script: `scripts/backup_training_data.sh`
- Held-out evaluation harness with no-TTA / TTA comparison
- VLM-as-classifier validation (Claude reads images directly)
- Procedural background synthesis pipeline
- Synthetic zoom variants (`_z70`, `_z50`)
- Background masking pipeline (`_mask`, `_clean`)
- Distillation pool: 1002 fresh crops from videos at fps=12 in
  `data/distill_pool/`

## Path forward (per Codex review + 2026-05-02 findings)

1. **Take phone PHOTOS, not videos**, for new training data. ~30 per
   class. 12 MP gives the central pin/socket feature plenty of
   resolution. Drop into the labeler upload UI.
2. **Vary the scene** when shooting: different surfaces, different
   lighting, different angles. Diversity is what lets the model
   generalize beyond the bench.
3. **Use Claude (via this conversation, no API cost on Max plan) to
   pre-label** new photos as they're uploaded — drag photo, Claude
   classifies, user confirms.
4. **Train an on-device model** (MobileNetV3-Small or similar) on the
   expanded photo dataset. The student model inherits the VLM's
   generalization without needing the VLM at runtime.
5. **For family classification, the existing video-crop dataset is
   fine** — keep it as auxiliary training data.

## Anti-recommendations (don't bother)

- More architecture iteration with current data: every variant lands
  in the same `25–38% Full / 38–62% Family / 25–75% Gender` band on
  the 8-image held-out. Ceiling is real.
- Geometric measurement on the existing wood-bench shots: hex / body
  segmentation just doesn't work without ArUco or cleaner backgrounds.
- VLM-at-inference: can't deploy on phone. VLM-at-development for
  distillation IS the play.

## Data layout reference

```
training/data/
  videos/                      ← 3 source .MOV files
  labeled/embedder/<CLASS>/    ← cleaned training crops (gitignored)
    <name>.jpg                 ← original Hough crop
    <name>_z70.jpg, _z50.jpg   ← center-cropped zoom variants
    <name>_mask.jpg            ← gray-masked variant (legacy)
    <name>_clean.jpg           ← rembg-cleaned (background→neutral gray)
    <name>_bg{0..4}.jpg        ← procedural-background composites
    <name>_central.jpg         ← tight central crop (50% face radius)
    <name>_centralv2.jpg       ← rembg-cleaned tight central (gender)
  test_holdout/<CLASS>/        ← 8 hand-verified phone photos
  distill_pool/<FAMILY>/       ← 1002 fresh crops awaiting VLM labeling
  archive/                     ← legacy scraped images (don't use)
  reference/                   ← committed vendor reference photos
```
