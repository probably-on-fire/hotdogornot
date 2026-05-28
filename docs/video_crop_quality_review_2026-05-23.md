# Video-crop training data: visual quality review

Date: 2026-05-23
Sample: 5 random `agg_*.jpg` per class × 10 classes = 50 crops drawn from
`/opt/rfcai/repo/training/data/videos/` sidecars.
Manifest: `rfcai-sample-crops/manifest.json`.

## What I'm looking at

Each crop was produced by `scripts/batch_extract_videos.py` at 2 fps from the
29 phone-uploaded training videos. The extractor runs Jerry's YOLO11n
detector, takes the highest-score box, then expands the box to a square crop
with some padding before writing the jpg. These crops are what feed Phase B's
retrain.

## Failure modes seen

### 1. Severely off-center framing (dominant — ~30-40% of crops)
The connector occupies a small corner of the crop and 60-85% is empty
backdrop (gray countertop / paper).

Examples:
- `3.5mm-F/agg_0107.jpg` — connector bottom-right, ~70% empty bg
- `3.5mm-F/agg_0198.jpg` — connector lower-left, ~75% empty bg
- `2.4mm-F/agg_0368.jpg` — connector at very bottom edge, ~85% empty bg
- `2.4mm-F/agg_0373.jpg` — connector lower-right, ~70% empty bg
- `2.92mm-F/agg_0215.jpg` — connector lower-right, ~70% empty bg
- `1.85mm-F/agg_0084.jpg` — connector lower-right corner, ~75% empty bg
- `1.85mm-F/agg_0197.jpg` — connector partially clipped on left edge
- `SMA-F/agg_1062.jpg`    — connector lower-right, ~75% empty bg
- `3.5mm-M/agg_0171.jpg`  — connector upper-right + braided cable in frame

Likely cause: the square-padding step around the YOLO box is overshooting
when the box is small (small connectors at distance), turning a tiny detection
into a mostly-background crop. The trained model then has to learn to find
a small connector in noise instead of learning the connector itself.

### 2. Pure false-positive crops — NO connector at all (rare but toxic)
- `SMA-M/agg_0018.jpg`    — YOLO cropped the cable braid sleeve, no SMA face
- `3.5mm-M/agg_0172.jpg`  — pure gray backdrop + edge of braided cable

These are completely mislabelled training examples — they teach the model
that a piece of cable braid IS a 3.5mm-M / SMA-M connector. Even a small
number of these pollutes the class distribution badly.

### 3. Marginal but usable (~20%)
Off-axis / angled / partial face views where the connector face is still
identifiable. Probably fine for training as natural variation.

### 4. Good clean crops (~40-50%)
Centered, sharp connector face filling most of the crop. These are exactly
what we want for training.

## Sharpness is NOT the bottleneck
The 2026-05-23 sharpness audit (`tmp_sharp_audit_*.md`) already showed
iPhone video has Laplacian variance p10=0.9, p90=2.2 across both full
frames and crops — the connector faces themselves are sharp enough for
the model when they're in the frame. The bottleneck is **framing**, not blur.

## Filter design (Phase B.2b)

`training/scripts/filter_video_crops.py` should reject a crop when **any**
of these conditions hit. All thresholds need calibration against a labelled
subset:

1. **YOLO box confidence < 0.40** (catches the cable-braid false positives;
   they typically score 0.30-0.40).
2. **YOLO box area / crop area < 0.15** (catches the severely off-center
   crops; a tight crop should be ~0.5-0.8).
3. *Optional*: rembg-foreground area < some threshold (would catch crops
   where the foreground extractor finds no metal at all). Cheaper alternative:
   metallic-pixel fraction via HSV V > 0.7 & S < 0.2 over connector bbox.

The Phase A.1 extractor records `extracted_crops` in `*.crops.json` sidecars
but does NOT currently record per-crop YOLO box_score or relative area.
The filter script will need to either:
- Re-run YOLO on each agg_*.jpg crop to get box_score (slow but doable), OR
- Re-extract with `batch_extract_videos.py` patched to store box_score +
  box_area / crop_area in the sidecar (the right answer long-term).

## What this changes downstream

- Phase B.2 (curate balanced subset): apply filter first, then curate.
- Phase B.3 (384x384 retrain): trains on filtered data only.
- Phase B.4 (eval vs Jerry's pretrained): if the filter helped, accuracy
  should jump; if it didn't, we know data isn't the bottleneck and the gap
  to Jerry is the variant-generation pipeline (rembg+bg+rescale).
