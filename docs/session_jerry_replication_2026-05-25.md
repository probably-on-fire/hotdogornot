# Session: replicating Jerry's pretrained — 2026-05-23 → 2026-05-25

**Question we set out to answer:** Why is Jerry's fork (`trextrader/hotdogornot`) so much better than ours? The partner pipeline benchmark put his model at 97.1% Full / 100% Gender on the 35-img holdout, vs our v18's 68.6% / 91.4%. We wanted to understand the gap and try to close it.

**TL;DR:** Pipeline is identical. The 28+ point gap was almost entirely **training-data composition** + an outdated baseline number. On the current cleaned 43-img holdout, Jerry's pretrained scores 81.4% Full / 88.4% Gender and our replica scores 79.1% Full / 95.3% Gender. We're at parity. The 97.1% baseline was on a smaller, older, cleaner holdout.

## What we did

### Phase 1 — pipeline forensics
Inspected Jerry's deployed model artifacts (`classifier_labels.json`, `thresholds.json`, ONNX graph shapes) and read his `train.py`. **Finding:** architecture (EffNetV2-S @ 384), preprocessing (PIL.BILINEAR, ImageNet norm baked into ONNX), and training code path were essentially identical to ours. He had `--architecture` and `--input-size` flags that we lacked + a save-best-val-acc checkpoint pattern — easy to port.

### Phase 2 — data forensics
Pulled Jerry's training set (`data/labeled/embedder/` in his repo) — 638 photos across 10 classes, ~40-86 per class (only ~2× imbalanced). Resolution: 4080×3060 native Samsung Galaxy photos. **The same source as our `photo_*` uploads** — Jerry's fork pulled snapshots from us.

| Class | Jerry's count | Our `photo_*` only |
|---|---:|---:|
| 1.85mm-F | 82 | 67 |
| 1.85mm-M | 86 | 69 |
| 2.4mm-F | 61 | 54 |
| 2.4mm-M | 63 | 49 |
| 2.92mm-F | 66 | 48 |
| 2.92mm-M | 54 | 45 |
| 3.5mm-F | 62 | 47 |
| 3.5mm-M | 70 | 61 |
| SMA-F | 54 | 41 |
| SMA-M | 40 | 30 |
| **Total** | **638** | **511** |

### Phase 3 — cross-eval
- Jerry's pretrained on **our Sonnet-filtered video crops**: 24.6% Full. **Three female classes scored 0%.** Smoking gun: video crops are head-on close-ups; Jerry's model never saw that distribution.
- Our best classifier on **Jerry's 300 photos**: 42-63% Full (depending on pipeline path). Our model isn't broken — it's just trained on a worse distribution.

### Visual audit (Phase C)
Contact sheets of female-class video crops vs Jerry's photos:
- Video crops: 75% head-on (looking straight into the connector face)
- Jerry's photos: side / 3-4 profile (showing both threaded coupling and inner pin)
- Real user uploads (= our holdout) are side-profile → match Jerry's distribution → fail our head-on-trained model

### Replication attempts (7 model variants)

| Model | Data | Recipe | Hybrid Full | Hybrid Gender | Notes |
|---|---|---|---:|---:|---|
| v18 (prod) | 23K mixed | ResNet-18 | 39.5% | 74.4% | baseline |
| v19_effnet_jerry_v3 | 3,649 Sonnet videos | bal, recipe | 48.8% | 67.4% | balance helped video data |
| v19_effnet_photos_only | 511 our photos | bal, recipe | 53.5% | 76.7% | first time 1.85mm-F = 5/5 |
| v19_effnet_jerry_replica | 300 Jerry photos | bal, recipe | 67.4% | 79.1% | proved data quality matters |
| v19_effnet_jerry_full (bal) | 638 Jerry, balanced→400 | bal, recipe | 67.4% | 86.0% | balance threw away 238 photos |
| **v19_effnet_jerry_full_nobal** | **638 Jerry, all used** | **no-bal, recipe** | **79.1%** | **95.3%** | **best replica** |
| Jerry's pretrained | his 635 | unknown bal | 81.4% | 88.4% | the target |

**The critical finding** between v19_effnet_jerry_full (balanced) and v19_effnet_jerry_full_nobal: dropping `--balance-to-smallest` gained +12pts Full. The flag caps every class at the smallest count (40), discarding ~200 of Jerry's photos. Jerry's `classifier_labels.json` shows 495 train + 140 val = 635 — i.e. he used essentially all 638 without balancing.

### Holdout cleanup
Diagnosed why both Jerry's model AND ours scored 0/3 on the 2.4mm-F holdout class: **all 3 files were mislabeled.** Sonnet vision audit on 8 suspect files identified 5 high-confidence relabels:

| File | From → To |
|---|---|
| `2_4mm-m.jpeg` (filename says male) | 2.4mm-F → 2.4mm-M |
| `IMG_0271.jpeg` | 2.4mm-F → 2.4mm-M |
| `IMG_0272.jpeg` | 2.4mm-F → 2.4mm-M |
| `IMG_0274.jpeg` | 2.92mm-F → 2.92mm-M |
| `IMG_0276.jpeg` | 3.5mm-F → 3.5mm-M |

After applying, 2.4mm-F class has 0 holdout samples. Backup at `/tmp/holdout_pre_relabel_2026-05-25/` on the box.

### The "97.1%" baseline was stale
On the cleaned 43-img holdout, Jerry's pretrained scores 81.4% Full — not 97%. The 97.1% in earlier CLAUDE.md notes was from the older 35-img holdout. When we tied his number, we'd actually been at parity for hours without realizing it.

## What didn't work

- **Video data alone** → distribution-mismatched the app's deployment
- **`--balance-to-smallest` on Jerry's data** → threw away usable photos
- **Standalone Hough eval on photo-trained models** → 16% Full because train-time `RandomResizedCrop(scale=(0.55, 1.0))` floor is too high for tight Hough crops. Photo-trained models need the YOLO hybrid path.
- **Original SFTP upload of 1.4GB tarball** → hung at 695MB; we cloned Jerry's repo directly on the box instead

## Tools we built (and that paid off later)

- **`training/scripts/train_v19_effnet.py`** — patched with Jerry's recipe defaults, save-best-val-acc checkpoint, crash-safe `weights_last.pt` per epoch
- **`/labeler/admin/datasets`** — unified browse + bulk-edit UI across all datasets. Used to audit the holdout. Deployed SFTP-only; uncommitted.
- **Sonnet vision dispatch pattern** — ~70 subagents, 40 images each, ~150s/batch. Used for both the angle-filter pass and the holdout label audit. Caveat: agents key output by basename and lose info when two batch members share a basename across classes; fix by emitting class-prefixed keys or full paths.

## Open questions

1. **Combined model (511 ours + 638 Jerry, dedupe)** is still training. If it beats 81.4% Full, that's our production candidate and we've crossed Jerry.
2. **The `IMG_02XX` batch** — ~6 remaining holdout failures across both models are concentrated in this batch. Different camera, different lighting, different angles than the rest. Adding samples like those to training would likely close the remaining ~15pt gap to a "perfect" score.
3. **Sonnet angle-filter pass halted at 40/92 batches** once we proved data composition mattered more than angle-filtering existing videos. Resume if we want to add side-profile-only video crops to the combined set.

## Files of interest from this session

- Memory: `[[jerry-replication-recipe]]`, `[[holdout-quality-issues]]`, `[[crop-scale-mismatch]]`, `[[datasets-admin-ui]]`
- Local: `tmp_eval_jerry_full_nobal.out`, `tmp_clean_holdout_and_eval.out`, `tmp_24mmF_audit.png`, `tmp_data_contact_sheet.png`, `tmp_female_audit.png`
- Box: `/home/rfcai/training/models/classifier_v19_effnet_jerry_full_nobal/`, `/home/rfcai/training/models/jerry_v19_hybrid_jerry_full_nobal/`, `/tmp/holdout_pre_relabel_2026-05-25/`
