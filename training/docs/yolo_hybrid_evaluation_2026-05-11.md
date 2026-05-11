# YOLO Hybrid Evaluation — 2026-05-11

Side-by-side benchmark of the friend's YOLO detector (committed in the
trextrader/hotdogornot fork) against current production, plus discovery
and fix of a `labels.json` ordering bug that had been silently inverting
every gender prediction in production for over a week.

## Pipelines

| | Pipeline A (production) | Pipeline B (hybrid) |
|---|---|---|
| FG gate | rembg U²-Net | rembg U²-Net |
| Composite | rembg silhouette on white | rembg silhouette on white |
| **Crop** | **cv2.HoughCircles** | **YOLOv11n** (`models/detector/best.pt`, conf=0.20) |
| Classify | v18 ResNet-18 + 5× TTA | v18 ResNet-18 |
| Endpoint | `127.0.0.1:8503/predict` (live) | local eval script |

## Final result on `test_holdout/` (8 phone shots)

```
truth       | A: prod         | B: hybrid       | filename
2.4mm-F     | 2.4mm-F    0.53 | 2.4mm-F    0.84 | 2_4mm-m.jpeg     (dup-mislabel)
2.4mm-F     | 2.4mm-F    0.77 | 2.4mm-F    0.88 | IMG_0271.jpeg
2.4mm-F     | 2.4mm-F    0.86 | 2.4mm-F    0.88 | IMG_0272.jpeg
2.4mm-M     | 3.5mm-F    0.61 | 2.4mm-F    0.64 | IMG_0270.jpeg
2.92mm-F    | 2.4mm-F    0.53 | 2.4mm-F    0.84 | IMG_0274.jpeg
2.92mm-M    | 2.92mm-M   0.64 | 2.92mm-F   0.93 | IMG_0273.jpeg
3.5mm-F     | 3.5mm-F    0.74 | 2.4mm-F    0.82 | IMG_0276.jpeg
3.5mm-M     | 3.5mm-M    0.57 | 3.5mm-F    0.69 | IMG_0275.jpeg
```

| Metric | A (production v18 + Hough) | B (hybrid v18 + YOLO) | Delta |
|---|---|---|---|
| Full class | **6/8 = 75.0%** | 3/8 = 37.5% | A wins by 37.5 pp |
| Family | 6/8 = 75.0% | 6/8 = 75.0% | tied |
| Gender | **7/8 = 87.5%** | 5/8 = 62.5% | A wins by 25 pp |

**Production wins.** The friend's YOLO detector + v18 hybrid is worse
than production v18 + Hough on Full and Gender, tied on Family.

Why: YOLO crops are wider than Hough's tight-on-face crops. The v18
classifier, trained on tight Hough-style crops, reads the wider YOLO
context as 2.4mm-F-shaped — visible in the prediction column where B
consistently outputs `2.4mm-F` regardless of true class. The wider
boxes hurt this model. YOLO's male/female-blind detector also can't
contribute any gender signal of its own (its class set is family-only:
`{0: '2.4MM', 1: '2.92MM', 2: '3.5MM'}`), so the entire gender
decision is left to v18 on a crop it doesn't recognize.

## The `labels.json` ordering bug we found and fixed along the way

Earlier in this session a side-by-side run produced wildly different
numbers (production at 0/75/12.5, hybrid at 37.5/75/37.5). After
investigating, the cause turned out to have nothing to do with model
weights or crops — it was the `class_names` ordering in
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

The previous file is preserved at
`/home/rfcai/training/models/connector_classifier/labels.before_invert_2026-05-11.json`
for reference.

After the fix, production matches its documented v18 baseline
(75/75/87.5) exactly. This explains why the README's headline number
hadn't moved in months even as the deployed model "felt off" in the
field — every phone user was seeing inverted gender predictions.

The repeated note in `classifier_journey.md` ("After the 2026-05-02
swap they're effectively inverted") was a partial diagnosis of the
same issue but was treating it as a known degradation rather than a
fixable bug.

### Root cause to fix in code

`train.py` must be writing `labels.json` with class names in a
different order than the dataset class indices assigned to the model
during training. The model's `Linear(512 -> 6)` layer outputs indices
0-5 in whatever order `ImageFolder` / our dataset emitted classes (most
likely alphabetical by folder name), but `labels.json` is being written
with a different order (M-first-per-family). Either:

- have `train.py` write `class_names` strictly in the order
  `dataset.classes` returns, OR
- explicitly sort folders into a stable order at training time and
  write the same order to `labels.json`.

Until that's fixed, every future retrain will reintroduce the same
inversion bug. The next retrain's `labels.json` should be inspected.

## What this means for the friend's track

The YOLO detector alone is not a production win for this classifier.
The friend's plan calls for pairing it with a **multi-head classifier**
trained from scratch (not yet committed) that should:

1. Take a fresh look at the gender signal in YOLO-style wider crops
   rather than inheriting v18's tight-Hough-crop priors.
2. Add explicit per-attribute heads (family, gender, polarity,
   sides, mount) instead of a single 6-way softmax.
3. Train end-to-end against the new YOLO bbox distribution.

The hypothesis that "swap Hough for YOLO and keep v18" is now
disproven for this data + model combo. Whether the friend's
multi-head training plan will actually beat v18+Hough is an open
question that needs his cloud GPU run to answer.

## Status of components after this session

| Component | State after this session |
|---|---|
| `labels.json` ordering | Fixed in production. `train.py` still has the bug; future retrains need labels inspected. |
| Production `/predict` | At documented 75/75/87.5 baseline again. |
| Friend's YOLO detector | Trained, committed at `models/detector/best.pt`. Doesn't help v18 directly. |
| Friend's multi-head classifier | Scaffold only; needs cloud GPU run before a real comparison. |
| Friend's detect-classify pipeline | Code present; can't run end-to-end without multi-head weights. |
| README's "75/75/87.5" | Was always the right number; deployed model just wasn't actually serving it until today. |

## Recommended next moves

1. **Patch `train.py` to write `labels.json` in the same order as
   `dataset.classes`.** Permanent fix to the bug we just hot-patched
   in production. Otherwise the next retrain re-introduces inversion.
2. **Ping the friend for multi-head classifier weights.** Without
   them his pipeline can't be tested end-to-end. The detect-only
   benchmark says YOLO isn't a win on its own, but multi-head might
   be a different story.
3. **Hold off on `/predict-v2` for now.** The motivation was a
   gender-accuracy lift that didn't actually exist once labels were
   correct. Re-evaluate after multi-head weights land.
