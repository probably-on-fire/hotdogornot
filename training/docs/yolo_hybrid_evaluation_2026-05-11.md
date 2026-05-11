# YOLO Hybrid Evaluation — 2026-05-11

Side-by-side benchmark of the friend's YOLO detector (committed in the
trextrader/hotdogornot fork) against current production. Both pipelines
use the same v18 ResNet-18 classifier; only the crop step differs.

## Pipelines

| | Pipeline A (production) | Pipeline B (hybrid) |
|---|---|---|
| FG gate | rembg U²-Net | rembg U²-Net |
| Composite | rembg silhouette on white | rembg silhouette on white |
| **Crop** | **cv2.HoughCircles** | **YOLOv11n** (`models/detector/best.pt`, conf=0.20) |
| Classify | v18 ResNet-18 + 5× TTA | v18 ResNet-18 |
| Endpoint | `127.0.0.1:8503/predict` (live) | local eval script |

## Result (`test_holdout/`, 8 phone shots)

```
truth       | A: prod         | B: hybrid       | filename
2.4mm-F     | 2.4mm-M    0.53 | 2.4mm-M    0.84 | 2_4mm-m.jpeg     (dup-mislabel)
2.4mm-F     | 2.4mm-M    0.77 | 2.4mm-M    0.88 | IMG_0271.jpeg
2.4mm-F     | 2.4mm-M    0.86 | 2.4mm-M    0.88 | IMG_0272.jpeg
2.4mm-M     | 3.5mm-M    0.61 | 2.4mm-M    0.64 | IMG_0270.jpeg   ← B fixes family
2.92mm-F    | 2.4mm-M    0.53 | 2.4mm-M    0.84 | IMG_0274.jpeg
2.92mm-M    | 2.92mm-F   0.64 | 2.92mm-M   0.93 | IMG_0273.jpeg   ← B fixes gender
3.5mm-F     | 3.5mm-M    0.74 | 2.4mm-M    0.82 | IMG_0276.jpeg
3.5mm-M     | 3.5mm-F    0.57 | 3.5mm-M    0.69 | IMG_0275.jpeg   ← B fixes gender
```

| Metric | A (production) | B (hybrid YOLO) | Delta |
|---|---|---|---|
| Full class | 0/8 = 0.0% | 3/8 = 37.5% | **+37.5 pp** |
| Family | 6/8 = 75.0% | 6/8 = 75.0% | 0 |
| Gender | 1/8 = 12.5% | 3/8 = 37.5% | **+25.0 pp** |

## Important caveat — the README's "75/75/87.5" is stale

The README and architecture.md cite v18 at **75% Full / 75% Family / 87.5%
Gender**. Those numbers were measured **before the 2026-05-02 M/F
relabeling** that flipped from plumbing convention (external thread = M)
to RF convention (pin = M). The model was never retrained against
post-swap labels, so its gender predictions are systematically inverted
on the current holdout. `classifier_journey.md` notes this inversion
explicitly:

> "After the 2026-05-02 swap they're effectively inverted (75% → 25% for
> the same physical predictions)."

The 0/8 / 6/8 / 1/8 numbers we see for Pipeline A on the current holdout
are consistent with that inversion plus a couple of family mistakes —
not a recent regression. Both `weights.synth_20ep.pt` and
`weights.synth_20ep_s1.pt` (the two existing v18 variants) produce
nearly identical post-swap predictions, confirming this isn't a
weights-file drift issue.

**Headline updates pending:**

- README + architecture.md should reflect actual post-swap baseline,
  not the pre-swap 75/75/87.5 number.
- The fix for gender accuracy isn't a model architecture change — it's
  either a retrain with post-swap labels, or relabel the holdout to
  pre-swap convention to undo the inversion.

## Why YOLO beats Hough (same classifier, different crops)

Inspection of the per-image predictions shows YOLO consistently helps on
male connectors. Hough Circles centers on the connector face and crops
tight around it; the male pin protrudes BELOW the face circle, so a
Hough crop frequently truncates the pin and the classifier loses the
gender cue. YOLO learns from full-connector bboxes and consistently
captures the pin/socket region. Concretely:

- All three male-truth holdout images were classified correctly by
  Pipeline B (2.4mm-M, 2.92mm-M, 3.5mm-M) and wrong by Pipeline A.
- For female-truth images, both pipelines defaulted to M (the gender
  inversion above + a model bias).

## Status of friend's track

| Component | State | Notes |
|---|---|---|
| YOLO detector | **Trained, weights committed** | `models/detector/best.pt` — 3-class family detector (2.4MM / 2.92MM / 3.5MM), mAP50=0.979 on his eval |
| Multi-head classifier | Scaffold only, no weights | Real training loop landed; needs cloud GPU run |
| Detect-classify CLI | Code present, can't run end-to-end | Blocked on the missing multi-head weights |
| Mobile/server export | Scaffold only | |
| `/predict` server change | None yet | Production unchanged |

## Recommended next moves

1. **Cheapest win — wire YOLO into a `/predict-v2` endpoint.** Mount
   YOLO before the current Hough call, keep v18 as the classifier,
   keep the rembg fg gate. Flutter app can hit `/predict-v2` while
   `/predict` stays as the rollback. If field testing on the phone
   confirms the 25 pp gender lift, promote v2 to `/predict` and retire
   the Hough path.

2. **Address the label-inversion separately.** Either retrain the
   classifier with post-swap labels (~30 min on the P40s after a small
   train.py tweak) or invert the labels.json mapping in production
   (instant, free, identical effect). The doc-cite "75/75/87.5" only
   becomes accurate again after this is fixed.

3. **Multi-head classifier training.** Friend's scaffolding is ready
   but needs a cloud run; once those weights exist the full
   detect-classify pipeline can be tested.
