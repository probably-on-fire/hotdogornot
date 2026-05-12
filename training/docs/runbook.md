# RF Connector AI — Deploy / Retrain Runbook

Operational reference. Read `classifier_journey.md` first for *why*
the pipeline is shaped the way it is.

---

## Box facts

- **Host**: `chris@192.168.20.235` (LAN). Public access via reverse SSH
  tunnels through `aired.com`.
- **Repo**: `/opt/rfcai/training` (symlinked to `/opt/rfcai/repo/training`)
- **Models**: `/home/rfcai/training/models/connector_classifier/`
- **Data**: `/opt/rfcai/repo/training/data/labeled/embedder/<class>/`
- **Source videos**: `/opt/rfcai/repo/training/data/videos/`
- **Held-out**: `/opt/rfcai/repo/training/data/test_holdout/<class>/`
- **Quarantine**: `/opt/rfcai/repo/training/data/labeled/_quarantine_*/`

The repo lives under user `rfcai`, so commands that touch repo files
need `sudo -u rfcai`. The systemd service runs as `rfcai` too.

### GPU (training only — predict service is CPU)

- **Hardware**: 2× Tesla P40, 24 GB each, sm_61 Pascal.
- **Driver**: NVIDIA 535.230.02 kernel module / 535.288.01 userspace —
  nvidia-smi will fail with "Driver/library version mismatch" until
  the box reboots, but CUDA runtime works fine because it talks to
  the kernel module directly. Don't reboot to fix `nvidia-smi`
  cosmetically — the box hosts production processes (chris's
  go_live_only.py, ollama, others) that have GPU handles open and
  would die.
- **PyTorch in rfcai venv**: `torch==2.5.1+cu121` /
  `torchvision==0.20.1+cu121` (2026-05-06). The training scripts auto
  `.cuda()` if available, so kicking any `train.py` /
  `exp_*_train.py` job runs on GPU automatically — no flags needed.
- **Smoke-test the install** after any pip churn:
  `sudo -u rfcai /opt/rfcai/repo/training/.venv/bin/python scripts/gpu_smoke.py`
- **If you need CUDA 13 / newer torch in future**: a kernel-module
  reload (`rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` then
  `modprobe nvidia`) requires killing every GPU user first — don't
  attempt without coordinating with chris.

## Deployed services

| Path on aired.com | Backed by | systemd service |
|---|---|---|
| `/rfcai/predict`, `/rfcai/predict-video` | predict_service.py:8503 | `rfcai-predict.service` |
| `/rfcai/labeler/*` | labeler.py (router inside predict_service) | (same) |
| `/rfcai/healthz` | relay app (NOT predict_service) | (relay-side) |
| `/demo/` | streamlit_app.py:8501 | `rfcai-streamlit.service` |

Note: `aired.com/rfcai/healthz` is the relay's healthz, not the
predict service's. The predict service's `/healthz` (with model
version, fg filter status, ensemble size) is only reachable on the
box at `127.0.0.1:8503/healthz`.

## Standard deploy after a `git push`

```bash
echo '<sudo-pwd>' | ssh chris@192.168.20.235 \
  'sudo -S sh -c "sudo -u rfcai git -C /opt/rfcai/training pull --ff-only \
                  && systemctl restart rfcai-predict"'
```

First request post-restart is slow (~30s on a fresh `~/.u2net/`)
because rembg downloads the U²-Net silhouette model on first use.
Subsequent requests are fast.

To verify the service came up clean:

```bash
sudo -u rfcai curl -s 127.0.0.1:8503/healthz | python -m json.tool
```

Should show:
```json
{
  "status": "ok",
  "classifier_loaded": true,
  "ensemble_size": 1,                 // or N if RFCAI_ENSEMBLE_WEIGHTS set
  "max_detections": 4,
  "classify_on_cleaned": true,
  "fg_filter": {"enabled": true, "available": true, ...},
  "yolo_fallback": {"enabled": true, "available": true,
                    "conf_threshold": 0.2},
  "spec_lookup": {"available": true, "families_indexed": 23}
}
```

If `fg_filter.available: false`, rembg failed to load — usually means
the `/opt/rfcai/training/.venv` is missing the `rembg` package.

## Production env knobs (`/etc/default/rfcai-predict`)

These are documented in `deploy/systemd/rfcai-predict.env.example`.
Currently in production:

```
RFCAI_FG_FILTER=1               # rembg pre-filter; rejects no-connector frames
RFCAI_CLASSIFY_ON_CLEANED=1     # feed rembg-cleaned silhouette to classifier
RFCAI_MIN_FG_FRACTION=0.05      # rembg threshold floor
RFCAI_REMBG_MODEL=u2netp        # lighter U^2-Net variant; ~3x faster than u2net
                                # with same silhouette quality on this dataset.
                                # Drops mean /predict latency 9.5s -> 5.0s.
RFCAI_YOLO_FALLBACK=1           # YOLO11n detector fallback when Hough finds 0
                                # crops. Detector trained by trextrader, lives
                                # at models/detector/best.pt — mAP50=0.979 on
                                # the fork's eval set.
RFCAI_YOLO_WEIGHTS=/opt/rfcai/repo/models/detector/best.pt
RFCAI_YOLO_CONF=0.2             # YOLO conf threshold for fallback crops
RFCAI_SPECS_PATH=/opt/rfcai/repo/training/rfconnectorai/specs/connectors.yaml
                                # spec lookup — each prediction enriched
                                # with frequency range, impedance, etc.
                                # connectors.yaml authored by trextrader
                                # (16-family taxonomy) in the fork's Epic 1.
# RFCAI_ENSEMBLE_WEIGHTS=...     # comma-separated extra weight files for averaging
```

## Tracked benchmark history (`training/reports/`)

Every accuracy- or latency-affecting change should be followed by a
run of `scripts/eval_holdout.py`, with the resulting `report.json` +
`report.md` committed under `training/reports/holdout_YYYY-MM-DD_*.md`.
This is the artifact path that catches silent regressions like the
labels.json inversion (which drifted production gender from 87.5% to
12.5% for 9 days unnoticed).

| Report | Latency | Full / Family / Gender | Notes |
|---|---|---|---|
| `holdout_2026-05-11_post_improvements.md` | 9.5 s | 75 / 75 / 87.5 | After u2net + per-crop rembg + structured output |
| `holdout_2026-05-11_baseline_restored.md` | 9.7 s | 75 / 75 / 87.5 | After reverting latency-opt attempts that broke accuracy |
| `holdout_2026-05-11_u2netp.md` | **5.0 s** | 75 / 75 / 87.5 | After rembg u2net → u2netp swap. **Current production.** |

Run:

```bash
python scripts/eval_holdout.py \
  --url http://127.0.0.1:8503/predict \
  --token $RFCAI_DEVICE_TOKEN \
  --holdout /opt/rfcai/repo/training/data/test_holdout \
  --out training/reports/holdout_$(date -u +%Y-%m-%d)_<label>
```

## Model file layout in `/home/rfcai/training/models/connector_classifier/`

| File | What |
|---|---|
| `weights.pt` / `weights.onnx` | The currently-loaded model |
| `labels.json` | Class names + INPUT_SIZE (must match `weights.pt`) |
| `weights.NNNN.pt` | Versioned by training run (auto_retrain bumps NNNN) |
| `weights.synth.pt` | Backup of v15 (synth-augmented, 12 epochs) |
| `weights.synth_20ep.pt` | Backup of v18 (synth-augmented, 20 epochs — current production) |
| `weights.seed1.pt`, `weights.seed2.pt` | v8-style seed variants for ensembles |
| `version.json` | Last training's metadata (NOT trustworthy after a manual revert) |

**Rolling back the model**: `sudo -u rfcai cp weights.synth_20ep.pt weights.pt
&& sudo -u rfcai cp weights.synth_20ep.onnx weights.onnx && bash
/opt/rfcai/training/scripts/_fix_v8_labels.sh` — that script also
restarts the service. The `_fix_v8_labels.sh` rewrites `labels.json`
to the canonical 6-class shape; needed because each retrain
overwrites `labels.json` to whatever class set it just used.

## Standard retrain flow

```bash
# Ensure code on box matches HEAD
sudo -u rfcai git -C /opt/rfcai/training pull --ff-only

# Train using the kick helper. Defaults: seed=0, epochs=12.
# Override with env vars:
RFCAI_SEED=0 RFCAI_EPOCHS=20 bash /opt/rfcai/training/scripts/_kick_retrain.sh

# Watch progress (only the FINAL epoch + completion line is logged):
sudo tail -F /tmp/rfcai_retrain.log

# After completion, restart the predict service to pick up new weights
sudo systemctl restart rfcai-predict
```

The kick script passes `--force` and `--balance-to-smallest`. To
turn off balance-to-smallest, edit the kick script directly.

## Synth data generation

```bash
sudo -u rfcai .venv/bin/python /opt/rfcai/training/scripts/synthesize_from_clean.py \
    --data-dir /opt/rfcai/repo/training/data/labeled/embedder \
    --variants-per-base 6 --seed 42
```

For each base image: rembg → silhouette tight crop → composite onto
random background (white / gray / beige / wood-noise / skin-tone /
real-bg-patch) at 50–90% scale. Saves as `synth_NNNNNN.jpg` per class.

The **v18 production model** was trained on a mix of two synth runs:
the original v15-recipe (no perspective/skin/blur) and the later
v17-recipe (with perspective + skin + motion blur). Total ~8300
synth across 6 classes; the diversity matters more than the recipe
purity.

## Quality auditing existing training data

```bash
sudo -u rfcai .venv/bin/python /opt/rfcai/training/scripts/audit_training_quality.py \
    --data-dir /opt/rfcai/repo/training/data/labeled/embedder \
    --quarantine-dir /opt/rfcai/repo/training/data/labeled/_quarantine_lowq \
    --min-fg 0.02 --apply
```

Drops bases (and their rembg-derived `_clean`/`_bg*`/`_z*`/`_central`
variants) where rembg sees less than `--min-fg` foreground. Default
0.02 is calibrated for training data; the inference filter uses 0.05.

## Held-out benchmark

```bash
TOKEN=66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1
for d in /opt/rfcai/repo/training/data/test_holdout/*-?; do
    truth=$(basename "$d")
    for f in "$d"/*.[jJ][pP]*; do
        echo -n "  truth=$truth img=$(basename $f): "
        curl -s -X POST -H "X-Device-Token: $TOKEN" \
            -F "image=@${f}" https://aired.com/rfcai/predict \
        | python -c "import json,sys; d=json.load(sys.stdin);
ps=sorted(d.get('predictions',[]),key=lambda x:-x['confidence']);
print('NO_DETECT' if not ps else f\"pred={ps[0]['class_name']} conf={ps[0]['confidence']:.2f}\")"
    done
done
```

Counts:
- **Full** = predicted class exactly matches truth
- **Family** = first part of the hyphenated class matches (e.g., `2.4mm`)
- **Gender** = last part matches (`M` / `F`)

Per-prediction noise on N=8: ±12.5pp. Don't over-react to single
trial differences <12.5pp.

## Known data quality issues

- `data/test_holdout/2.4mm-M/2_4mm-m.jpeg` and
  `data/test_holdout/2.92mm-M/IMG_0274.jpeg` are byte-identical
  (md5 `5dd2d7c3…`). One label is wrong by construction — max
  achievable Full is 7/8 = 87.5% until one is removed. Filename
  hint suggests the 2.4mm-M label is correct.
- **SMA-M / SMA-F and 1.85mm-M / 1.85mm-F have zero training samples.**
  Both families are reserved slots — `auto_retrain._populated_classes`
  drops them from the trained head until each class folder has
  ≥`MIN_SAMPLES_PER_CLASS` (default 5) images. Flutter app exposes
  all five families via the chip-correction strip on Identify and
  the capture chips on Contribute (`_kFamilies` in
  `identify_screen.dart` / `contribute_screen.dart`), so contributed
  uploads can grow each family until the next retrain expands the
  head automatically.

## Common deployment gotchas

- **Pull conflicts** when `_kick_*.sh` files have been hand-edited
  on the box outside of git. Easiest fix: `sudo -u rfcai rm
  /opt/rfcai/training/scripts/_kick_*.sh` and re-pull.
- **systemd `ProtectSystem=strict`** makes `/opt` read-only for
  the service. New paths under `/opt` need to be added to
  `ReadWritePaths` in `rfcai-predict.service`. Currently allowed:
  `/home/rfcai`, `/opt/rfcai/repo/training/data`,
  `/opt/rfcai/repo/training/models`.
- **CUDA driver too old** error on the box is benign — the service
  falls back to CPU which is fast enough for inference (one batch
  per request).
