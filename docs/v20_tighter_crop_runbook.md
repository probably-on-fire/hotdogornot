# v20 tighter-crop experiment — runbook

Goal: see whether tightening the inference/training crop padding from
`pad_frac=0.35` to `pad_frac=0.15` improves fine-pitch family accuracy
(2.4 / 2.92 / 3.5mm confusions are the dominant baseline failures).

Status as of 2026-05-17:
- **Baseline (v18, default crop):** 68.6% Full / 68.6% Family / 91.4% Gender
  on 35 freshly-carved holdout images. v18 was trained 2026-05-05; all 35
  carved images are dated 2026-05-14 → 2026-05-16, so this baseline is
  clean (model has never seen them).
- Worst classes: SMA-M (1/5), 2.92mm-M (2/5), 3.5mm-F (2/5).
- Best classes: SMA-F, 2.92mm-F, 1.85mm-M (5/5 each).
- Existing holdout: `data/test_holdout/<cls>/`, now 43 images total
  (8 original IMG_*.jpeg + 35 new `holdout-carve-2026-05-17_*.jpg`).

## Two experiments — run in order

**Phase A (inference-only, ~10 min):** drop pad_frac to 0.15 at inference
only. Doesn't retrain. If accuracy *holds or improves*, the classifier
generalizes to tighter crops and we proceed to Phase B. If accuracy
*drops*, the model needs to be retrained on the tighter distribution.

**Phase B (full v20, ~1-2 hours):** regenerate training crops with the
new pad_frac, retrain, deploy. Only run if Phase A is at least neutral.

## Prereqs

- You're on the LAN (so `192.168.20.235` is reachable, OR `chris@aired.com`
  still works for code pull on the box via git).
- `python tmp_eval.py` works from `E:\anduril` (the eval script we wrote).
- Baseline numbers are recorded in `tmp_baseline_eval.json` and
  `tmp_baseline_eval.md`. Don't overwrite.

## Phase A — inference-only test

### 1. Make the code change locally

```bash
# in E:\anduril
git checkout -b exp/tighter-crop-pad-015
```

Edit `training/rfconnectorai/data_fetch/connector_crops.py`:

```diff
 def detect_connector_crops(
     frame_bgr: np.ndarray,
     min_area_frac: float = 0.0005,
     max_area_frac: float = 0.10,
-    pad_frac: float = 0.35,
+    pad_frac: float = 0.15,
     max_crops: int = 4,
     edge_threshold_std: float = 2.0,
     min_circularity: float = 0.0,
 ) -> list[CropResult]:
```

Commit:

```bash
git add training/rfconnectorai/data_fetch/connector_crops.py
git commit -m "exp: tighten inference pad_frac 0.35 -> 0.15"
git push origin exp/tighter-crop-pad-015
```

### 2. Deploy to the box

```bash
ssh chris@192.168.20.235
# password: Elad9651!

sudo -u rfcai bash -c "cd /opt/rfcai/training && \
  git fetch origin && \
  git checkout exp/tighter-crop-pad-015"
sudo systemctl restart rfcai-predict
# Wait ~60s for service to warm up
until curl -sf http://127.0.0.1:8503/healthz >/dev/null; do sleep 3; done
echo "service up"
```

### 3. Eval Phase A

On the Windows/Mac machine, in `E:\anduril`:

```bash
# Run the same eval script against the same 35-image holdout
python -u tmp_eval.py
# Output: tmp_baseline_eval.md is OVERWRITTEN. Move it first if you want to keep it.
mv tmp_baseline_eval.md tmp_phaseA_eval.md
mv tmp_baseline_eval.json tmp_phaseA_eval.json
```

Compare:

```bash
# Diff per-class. The headline number is the "Full" % at the bottom.
diff tmp_baseline_eval.md tmp_phaseA_eval.md | head -50
```

### Decision gate

- Phase A Full % >= baseline 68.6%: **proceed to Phase B** (the
  classifier handles tighter crops fine; retraining should compound the
  gain).
- Phase A Full % < baseline 68.6% by >3pts: **stop**. The classifier
  was trained on looser crops and can't generalize tighter without
  retraining. Go to Phase B, but expect a bigger gain there. (Or revert
  pad_frac and try a different intervention.)

### Rollback (if needed)

```bash
ssh chris@192.168.20.235
sudo -u rfcai bash -c "cd /opt/rfcai/training && git checkout master"
sudo systemctl restart rfcai-predict
```

## Phase B — full v20 retrain

### 1. Regenerate training crops with tighter pad_frac

On the box:

```bash
ssh chris@192.168.20.235
sudo -u rfcai bash -c "cd /opt/rfcai/training && \
  .venv/bin/python -m scripts.zoom_recrop_training \
    --data-dir data/labeled/embedder \
    --pad-frac 0.15 \
    --min-fg 0.04 \
    --max-size-frac 0.7"
```

This adds new `tight_NNNNNN.jpg` files to each class folder. The OLD
tight_ crops (made at pad_frac=0.35 by previous runs) STAY in the
dataset. To get a clean v20 trained ONLY on the new tight crops, either:

a) **Snapshot then clean** (cleaner, slower): tar the existing
   `data/labeled/embedder` first, then `find ... -name 'tight_*' -newer
   <date>` and keep only the new ones. Slow.

b) **Train on the union** (faster, dirtier): just retrain. The new
   tight_ files will outnumber old ones if you ran zoom_recrop
   recently. Mixing crop scales might even help generalization.

Recommend (b) for the first pass.

### 2. Trigger retrain

```bash
sudo -u rfcai bash /opt/rfcai/training/scripts/_kick_retrain.sh
# Will log to /tmp/rfcai_retrain.log
tail -f /tmp/rfcai_retrain.log
# Look for "epoch 12 val_acc=..." at the end. Should take 30-90 min on GPU.
```

When training finishes, `scripts/auto_retrain.py` writes the new
weights to `/home/rfcai/training/models/connector_classifier/`. The
predict service picks them up on next restart.

### 3. Deploy v20

```bash
sudo systemctl restart rfcai-predict
until curl -sf http://127.0.0.1:8503/healthz >/dev/null; do sleep 3; done
```

Check model version:

```bash
curl -s http://127.0.0.1:8503/healthz
# Should show "model_version": 19 (or whatever incremented to).
```

### 4. Eval v20

```bash
# On Windows/Mac
python -u tmp_eval.py
mv tmp_baseline_eval.md tmp_v20_eval.md
mv tmp_baseline_eval.json tmp_v20_eval.json
```

### Decision gate

- v20 Full % > baseline + 3pts: **ship it.** Merge `exp/tighter-crop-pad-015`
  to master. Update `docs/classifier_journey.md` with the tighter-crop
  finding.
- v20 Full % within ±3pts of baseline: **no clear win**. Revert. Try a
  different pad_frac (0.20 next, or 0.25), or shift focus to data
  imbalance (SMA-M only has 29 training images post-carve — class
  weighting could help more than crop tightening).
- v20 Full % < baseline - 3pts: **regression**. Revert. Investigate the
  per-class numbers — likely some class lost more pixels than it could
  afford (fine-pitch ones with very small connectors in the frame).

## Notes / open questions

- **2.4mm-M holdout is still 1 image** (the labeler /grid hung when
  we tried to enumerate it). After restarting `rfcai-predict` for
  this experiment, re-run `python tmp_carve.py` to see if 2.4mm-M
  enumerates now. If it does, carve 5 photos there and re-eval.
- **3.5mm-M and 2.4mm-F have 0 photo_ files in train** (only `agg_*`
  averaged crops + augmentations). They got no new holdout images this
  round. The holdout is still effectively 1 image each for those classes.
  Future capture rounds should prioritize photo_ uploads for these.
- The 8 original holdout images (IMG_*.jpeg) are NOT in `tmp_baseline_eval`
  — we only evaluated the 35 freshly-carved ones. To include the
  originals, the eval script needs a way to list all holdout files,
  which the labeler API doesn't expose. SSH+ls when next on the box
  would solve this; document the paths in `tmp_full_holdout.json`.
- Latency for v18 with default crop was 5.5–7.5s per image. Tighter
  crops won't change this (same Hough work; the crop step is the
  same algorithm, just smaller padding).
