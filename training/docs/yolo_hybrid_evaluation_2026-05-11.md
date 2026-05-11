# YOLO Hybrid Evaluation — 2026-05-11

Three-way side-by-side benchmark of current production, the
detector-only hybrid, and the full multi-architecture pipeline the
trextrader/hotdogornot fork is building toward. Plus discovery and
fix of a `labels.json` ordering bug that had been silently inverting
every gender prediction in production for over a week.

## The three pipelines

| | A: production | B: hybrid | C: pure Jerry |
|---|---|---|---|
| FG gate | rembg U²-Net | rembg U²-Net | — |
| Composite | rembg silhouette on white | rembg silhouette on white | — |
| **Crop** | cv2.HoughCircles | YOLOv11n (his) | YOLOv11n (his) |
| **Classifier** | v18 ResNet-18 (ours, single head) | v18 ResNet-18 (ours, single head) | Multi-head ResNet-18 (his scaffolding, trained today from our data) |

The multi-head classifier in (C) was trained on the box GPU this
afternoon: 8 epochs, batch 32, 11,117 train / 1,411 val / 1,322
test, mean val accuracy plateaued at 0.984. Training data came from
running `crop_instances` (whole-image mode) + `build_yolo_dataset`
on our existing `data/labeled/embedder/` tree. Only the `family` and
`side_a_gender` heads carry real labels; every other attribute is
`unknown` or `not_applicable` per the auto-generated manifest.

## Final result on `test_holdout/` (8 phone shots)

```
                                  pred (family, gender)
truth       | A: prod         | B: hybrid       | C: pure Jerry
2.4mm-F     | 2.4mm-F    0.53 | 2.4mm-F    0.84 | (none)
2.4mm-F     | 2.4mm-F    0.77 | 2.4mm-F    0.88 | 2.4mm  male_pin
2.4mm-F     | 2.4mm-F    0.86 | 2.4mm-F    0.88 | 2.4mm  male_pin
2.4mm-M     | 3.5mm-F    0.61 | 2.4mm-F    0.64 | 3.5mm  male_pin
2.92mm-F    | 2.4mm-F    0.53 | 2.4mm-F    0.84 | (none)
2.92mm-M    | 2.92mm-M   0.64 | 2.92mm-F   0.93 | 3.5mm  male_pin
3.5mm-F     | 3.5mm-F    0.74 | 2.4mm-F    0.82 | 3.5mm  male_pin
3.5mm-M     | 3.5mm-M    0.57 | 3.5mm-F    0.69 | 3.5mm  male_pin
```

| Metric | A: prod | B: hybrid | C: pure Jerry |
|---|---|---|---|
| Full class | **6/8 = 75.0%** | 3/8 = 37.5% | 1/8 = 12.5% |
| Family | 6/8 = 75.0% | 6/8 = 75.0% | 4/8 = 50.0% |
| Gender | **7/8 = 87.5%** | 5/8 = 62.5% | 3/8 = 37.5% |

**Production wins decisively across every metric.** The friend's
detector + classifier system on our existing data is the weakest of
the three.

## Why C is worse than B is worse than A

1. **YOLO recall fails on 2/8 holdout photos.** `2_4mm-m.jpeg` and
   `IMG_0274.jpeg` are both auto-classified as `(none)` in pipeline
   C — YOLO returns zero detections above its `0.25` default
   threshold. Pipeline A catches them because the rembg foreground
   gate doesn't require shape-level confidence.
2. **YOLO crops bias the classifier toward male/2.4mm-F.** Every
   pipeline that runs the classifier on a YOLO crop tends to output
   `2.4mm-F` (B) or `male_pin` (C) regardless of true class. YOLO's
   wider boxes carry features that both classifiers latch onto as
   class-confounding cues.
3. **Multi-head adds no new signal on our current data.** The
   `crop_instances` whole-image converter writes every non-family
   non-gender attribute as `unknown` / `not_applicable` because we
   have nothing else labeled. Multi-head architecture is supposed to
   factor predictions across many attribute heads; with only two
   non-trivial heads it reduces to ~the "two-head" architecture we
   already tested in May (25/75/37.5 per
   `classifier_journey.md`) — and pure Jerry lands close to that.

## The `labels.json` ordering bug we found and fixed along the way

Earlier in this session a side-by-side run produced wildly different
numbers (production at 0/75/12.5, hybrid at 37.5/75/37.5). The cause
turned out to have nothing to do with model weights or crops — it
was the `class_names` ordering in
`/home/rfcai/training/models/connector_classifier/labels.json`. The
deployed file listed pairs as M before F per family:

```json
"class_names": [
  "3.5mm-M", "3.5mm-F",
  "2.92mm-M", "2.92mm-F",
  "2.4mm-M", "2.4mm-F"
]
```

The v18 model's actual output indices put F before M (probably from
alphabetical-by-folder enumeration during training). So every gender
prediction was being relabeled to its opposite before reaching the
phone app. The fix:

```json
"class_names": [
  "3.5mm-F", "3.5mm-M",
  "2.92mm-F", "2.92mm-M",
  "2.4mm-F", "2.4mm-M"
]
```

Previous file preserved at
`/home/rfcai/training/models/connector_classifier/labels.before_invert_2026-05-11.json`.

After the fix, production matches its documented v18 baseline
(75/75/87.5) exactly. This explains why the README's headline number
hadn't moved in months even as the deployed model "felt off" in the
field — every phone user was seeing inverted gender predictions.

### Root cause fix in `train.py`

Added in this commit. `train.py` now writes `labels.json` from
`dataset.class_names` (the order the model actually saw) and asserts
that it matches `config.class_names`. Future retrains that
introduce an order mismatch will fail loudly at train time instead
of silently shipping inverted predictions.

## What this means for the friend's track

The "pure Jerry" pipeline is not yet a contender on our data. Two
things would change that:

1. **More attribute labels.** Multi-head wants per-image labels for
   polarity, mount style, side-A/side-B gender, termination, etc.
   We have none of these. Until the dataset is annotated with these
   attributes, multi-head architecture is overkill for what it can
   learn. This is what the user means by "add more data once we get
   the connectors" — physical samples need to be acquired and
   labeled at attribute level.
2. **Either retrain YOLO with more recall** or pair it with the
   rembg gate. The 2/8 auto-miss is killing the holdout numbers.
   Either path needs an experiment.

The friend's overall architecture direction (detector → multi-head
attribute classifier → spec lookup) is sound. The execution is
blocked on data, not code.

## Status of components after this session

| Component | State |
|---|---|
| `labels.json` ordering | Fixed in production. `train.py` patched to fail loud on future mismatches. |
| Production `/predict` | At documented 75/75/87.5 baseline. |
| Friend's YOLO detector | Trained, committed at `models/detector/best.pt`. Not a production win for v18. |
| Friend's multi-head classifier | Trained from our data this session. 0.984 mean val acc. 12.5% Full on holdout — weakest of the three pipelines. |
| Detect-classify CLI | Works end-to-end with both halves loaded. |

## Recommended next moves

1. **Keep production on v18 + Hough.** It's the documented baseline,
   it's now serving correct labels, and nothing we tested today
   beats it.
2. **Acquire physical connector samples and label them at attribute
   level** (polarity, mount, sides, termination, etc.). This is the
   gate on whether the friend's multi-head architecture can produce
   a real lift. Without those labels his architecture is overkill.
3. **Re-evaluate after the next batch of attribute-labeled data
   lands.** Train both single-head v18 and pure-Jerry multi-head on
   the enriched dataset; compare again.
