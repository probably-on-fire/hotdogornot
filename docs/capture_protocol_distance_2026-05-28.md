# Realistic-distance capture protocol — fine-pitch RF connectors

**Date:** 2026-05-28
**Goal:** Collect realistic-distance training shots for the six fine-pitch classes so the classifier stops being confidently-wrong on phone-app usage. The 2026-05-27 phone-realistic eval established this is a data gap, not a calibration problem — inference-time tricks (ensemble disagreement, reticle-region forcing, TTA) cannot recover these cases because the two ensemble members independently agree on the wrong answer at high confidence.

## What to capture

**Six classes, ~15-20 shots each (90-120 shots total, ~2h):**

| Class | Today's holdout count | Why |
|---|---|---|
| 3.5mm-M | 5 (was failing 4/24 on user phone-test) | Worst-failing class |
| 3.5mm-F | 5 (was failing 8/18 on user phone-test) | Second-worst |
| 2.92mm-M | 7 | Confused with 3.5mm-M (the 2 CW cases) |
| 2.92mm-F | 7 | Family-mate, prevents boundary collapse |
| 2.4mm-M | 5 | Family-mate, prevents 2.4↔2.92 confusion |
| 2.4mm-F | 0 (was relabeled away) | Highest gap |

## Per-shot variation (target spread, not exhaustive)

Each class needs coverage across three axes. Mix freely — don't shoot 5 in a row identical.

**Distance / framing** (the axis we're fixing — the biggest miss-driver):
- ~5 shots: connector tip fills the reticle ring (close)
- ~5 shots: connector tip fills ~70% of reticle ring (medium — natural holding distance)
- ~5 shots: connector tip fills ~50% of reticle ring (further — typical first-frame distance)

**Angle:**
- Mostly head-on (looking straight at the connector face)
- Include 2-3 shots at ±15° tilt and 1-2 at ±30° tilt — RF connectors are cylindrical, slight rotation provides 3D context. (Per the [[slight-angle-videos-useful]] memory — these are not noise.)

**Lighting / background:**
- Vary across ambient room light, direct lamp, side-lit, soft shadow
- Vary backgrounds: hand, neutral table, dark surface, cluttered surface
- Avoid pure-white backgrounds (the model was already biased toward those)

## How to capture

1. Build current Flutter APK, install on phone (`https://aired.com/app.apk?v=5` or fresh build with `?v=6` if pushed).
2. Open app → **Contribute** tab → sign in (`chris / Elad9651!`).
3. Tap the class chip (e.g., `3.5mm-M`) — the chip persists across shots.
4. For each shot:
   - Position connector inside the reticle ring at the target distance
   - Hold steady
   - Tap shutter
   - Wait for the green toast (`✓ #N 3.5mm-M`) — confirms upload landed
5. Tap the counter pill to see live progress per class.

Files land on the box as `labeled/embedder/<cls>/photo_2026-05-28_<session>_<n>.jpg` (and a backup hardlink in `source_backup/`).

## Verification (when done)

```bash
# On the box, count today's new shots per class
ssh chris@192.168.20.235 'echo Elad9651! | sudo -S -p "" bash -c "
  for c in 3.5mm-M 3.5mm-F 2.92mm-M 2.92mm-F 2.4mm-M 2.4mm-F; do
    n=\$(ls /opt/rfcai/repo/training/data/labeled/embedder/\$c/photo_2026-05-28_* 2>/dev/null | wc -l)
    printf \"%-10s %d\\n\" \$c \$n
  done
"'
```

Target: ≥15 per class. If short, fill the gap.

## What happens after capture

1. Carve ~3 shots per class into a new phone-realistic holdout (`test_holdout_phone_2026-05-28/`). Don't let them touch training data.
2. Build `combined_v6` snapshot = combined_v5 train data + remaining new shots (~12-17 per class).
3. Retrain with same recipe v5 used:
   - `RandomResizedCrop(scale=(0.20, 1.0))`
   - epochs=50, batch=16, lr=1e-4, seed=0
4. Build `jerry_v19_hybrid_combined_v6_2026-05-28/` bundle (mirror tmp_build_hybrid_v4.py).
5. Eval against THREE holdouts: 52-img close-up, 12-img v0 phone-realistic, ~18-img v1 phone-realistic.
6. Ship decision: any confidently-wrong on the new phone-realistic holdout → don't ship, iterate.

Total turnaround: ~2h capture + 2.5h training + 30min eval = ~5h.

## Why this should work (theory)

The combined_v4 retrain (the first attempt with today's 44 shots) regressed close-up holdout because:
- 44 shots concentrated in ONE class (3.5mm-M) shifted the 3.5mm-M/2.4mm-M decision boundary
- Aug recipe (scale floor 0.55) didn't accommodate the tighter-crop distribution
- Other fine-pitch classes had no new data to "pull back" their boundaries

Capturing across ALL fine-pitch classes spreads the boundary shift evenly, and v5's wider augmentation (scale floor 0.20) lets the model generalize across distances. The two interventions together address both root causes.

## Things NOT to do

- **Don't** retrain on just 3.5mm-M shots again — that's what made combined_v4 regress.
- **Don't** capture only at one distance — distance variance is the entire point.
- **Don't** capture against pure white backgrounds — model is already biased toward bench shots.
- **Don't** skip 2.4mm-F just because today's user-phone-test didn't include it — it has 0 holdout samples right now, which means we cannot even measure how broken it is.
