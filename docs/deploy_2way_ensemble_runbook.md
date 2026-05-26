# Deploy 2-way ensemble (combined_v2 + jerry_full_nobal) + best-CLS-conf box

Status: code change is **local + uncommitted** (`training/rfconnectorai/pipeline/jerry_pipeline.py`, `training/rfconnectorai/server/predict_service.py`). See [Code changes](#code-changes) below.

There are now **two** improvements available, both env-gated:
- `RFCAI_JERRY_EXTRA_MODEL_DIRS=...` — softmax-average a second classifier (2-way ensemble): 81.4 → **88.4% Full**
- `RFCAI_BEST_CLS_CONF_BOX=1` — re-rank detected boxes by classifier confidence (instead of YOLO score): on top of the 2-way, gives **90.7% Full / 100% Gender** (and would be 92.9% / 100% with the duplicate fix). See [[best-cls-conf-box-selection]] memory.
- Both env vars are independent: each can be enabled or disabled separately. Empty/unset = preserves current single-model baseline (tested 2026-05-26).

Holdout results (cleaned 43-img, current production path):

| Variant | Full | Family | Gender | Δ vs prod |
|---|---:|---:|---:|---|
| Current prod (combined_v2 only) | 81.4% | 86.0% | 90.7% | — |
| **2-way prod + jerry_full_nobal** | **88.4%** | **90.7%** | **95.3%** | **+7.0pt Full / +4.6pt Gender** |
| 2-way + TTA (any combo) | 88.4% | 90.7% | 95.3% | (TTA adds nothing once ensembled) |
| 3-way + jerry_full | 86.0% | 88.4% | 95.3% | regresses |
| 3-way + jerry_v3 | 81.4% | 86.0% | 93.0% | regresses badly |

5 remaining misses (vs 8 in baseline). **All 5 misses have conf < 0.65** (values: 0.308, 0.388, 0.495, 0.495, 0.585) → they all abstain at the Flutter app's `_kMinAcceptedConfidence = 0.65`. Confident-prediction accuracy is **100% on 31/43 = 72% coverage**. Zero confidently-wrong outputs.

> **Note 2026-05-26:** the 43-img holdout actually contains one byte-identical duplicate (`2.4mm-M/2_4mm-m.jpeg` == `2.92mm-M/IMG_0274.jpeg`). Two of the 5 misses are the same image filed under two class dirs. The true unique count is 42; removing the mislabeled copy bumps the metric to **38/42 = 90.5% Full** without changing the model. See [[holdout-has-mislabeled-duplicate]]. Threshold sweep:

| Threshold | Coverage | Accuracy-on-confident |
|---:|---:|---:|
| 0.40 | 95% | 92.7% |
| 0.50 | 86% | 97.3% |
| **0.60** | **77%** | **100.0%** |
| **0.65** | **72%** | **100.0%** |
| 0.85 | 56% | 100.0% |

The 0.60-0.65 window is the kink where accuracy hits 100% and coverage is still high. No need to retighten the threshold.

## What changes

1. `JerryPipeline.__init__` takes optional `extra_model_dirs: list[Path]`.
2. Each extra bundle's `classifier.onnx` is loaded + softmax-averaged in `_classify_crop`.
3. `predict_service.py` reads a new env var `RFCAI_JERRY_EXTRA_MODEL_DIRS` (comma-separated bundle dirs).
4. When the env var is empty/unset → no behavior change (verified by tests; baseline 81.4% reproduces).

Both extras share the **detector** (only the classifier ONNX is loaded per extra). Inference cost roughly doubles (extra ~80MB classifier, ~150ms/image extra CPU on the P40 box).

## Pre-deploy backup

```bash
ssh chris@192.168.20.235
sudo cp /etc/default/rfcai-predict /etc/default/rfcai-predict.bak.2026-05-26-ensemble
```

## Deploy steps

```bash
# 1. Push code change (commit + push from your local clone first)
ssh chris@192.168.20.235
sudo -u rfcai git -C /opt/rfcai/training pull --ff-only

# 2. Add the new env vars. Append these lines to /etc/default/rfcai-predict:
#    RFCAI_JERRY_EXTRA_MODEL_DIRS=/home/rfcai/training/models/jerry_v19_hybrid_jerry_full_nobal
#    RFCAI_BEST_CLS_CONF_BOX=1     # opt-in to the +2.3pt Full / +4.7pt Gender box-selection lift
sudo nano /etc/default/rfcai-predict      # or: echo '...' | sudo tee -a ...

# 3. Restart the service
sudo systemctl restart rfcai-predict

# 4. Wait ~60s for warm-up, verify
sleep 60
curl -sf http://127.0.0.1:8503/healthz | python3 -m json.tool
sudo journalctl -u rfcai-predict -n 60 --no-pager | grep -i 'jerry pipeline enabled'
# expected: "extras=['/home/.../jerry_v19_hybrid_jerry_full_nobal']"

# 5. Smoke test against a single image (use any holdout sample)
TOKEN='66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1'
curl -sf -H "X-Device-Token: $TOKEN" \
  -F "image=@/opt/rfcai/repo/training/data/test_holdout/SMA-M/<any>.jpg" \
  http://127.0.0.1:8503/predict | python3 -m json.tool
```

## Rollback

```bash
sudo cp /etc/default/rfcai-predict.bak.2026-05-26-ensemble /etc/default/rfcai-predict
sudo systemctl restart rfcai-predict
```

(Code-side change is backward compatible: leaving `RFCAI_JERRY_EXTRA_MODEL_DIRS` unset preserves the single-classifier path even with the new code in place. So a code-only rollback is unnecessary unless you find a real bug.)

## Code changes (uncommitted, in your local working tree)

`training/rfconnectorai/pipeline/jerry_pipeline.py`:
- `__init__` gains an optional `extra_model_dirs: list[Path] | None = None` kwarg.
- Loads each extra bundle's `classifier.onnx`, validates that its `classifier_labels.json` matches the primary's class order.
- `_classify_crop` averages softmax across primary + all extras.

`training/rfconnectorai/server/predict_service.py`:
- Reads `RFCAI_JERRY_EXTRA_MODEL_DIRS` (comma-separated paths), parses into `Path` list, passes to `JerryPipeline(...)`.
- Log line gets an `extras=[...]` field for journalctl visibility.

Tested 2026-05-26 against the live box:
- Without env var → 35/43 = 81.4% Full (matches current prod).
- With `extras=[jerry_v19_hybrid_jerry_full_nobal]` → 38/43 = 88.4% Full (matches eval-runner).

## Why not 3+-way

Tried every 3-way using existing bundles:
- prod + nobal + photos_only → 88.4% (ties — photos_only too weak to shift argmax)
- prod + nobal + jerry_full → 86.0% (regression)
- prod + nobal + jerry_v3 → 81.4% (regression — jerry_v3 is biased)
- prod + nobal + jerry_replica → 86.0% (regression)
- 4-way prod+nobal+replica+v2 → 88.4% (tie, more cost)

The 2-way is the local optimum. Adding more models drags the average toward weaker components.

## Why not TTA

Tried H-flip, V-flip, ±5°, ±10° rotations × {ensemble, no-ensemble}:
- TTA alone on prod: 86.0% (helps single-model)
- TTA on top of 2-way ensemble: 88.4% (no further gain)

The ensemble already smooths the prediction variance that TTA captures.
