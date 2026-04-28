# Continuous Learning Architecture

End-to-end loop that closes the gap between "user takes a photo of a connector"
and "the model that runs on every phone gets better as more photos come in."

---

## The loop, top to bottom

```
                          ┌──────────────────────────┐
                          │  AR app (Unity / Sentis) │
                          │  - live camera           │
                          │  - ensemble inference    │
                          │  - prompt on low conf    │
                          └─────┬────────────────┬───┘
                  upload frames  │                │ poll /model/version
                  + claimed_class │                │ download if newer
                                 ▼                │
                ┌────────────────────────────────────────┐
                │  Relay server (FastAPI)                │
                │  POST /uploads                          │
                │  GET  /model/version                    │
                │  GET  /model/latest                     │
                │  GET  /model/weights                    │
                │  GET  /model/labels                     │
                └─────┬─────────────────────────────────▲─┘
        writes to     │                   reads          │
        incoming/     │                   manifest.json  │
                      ▼                                  │
                ┌─────────────────────────────────────────┐
                │  Training machine (this PC)             │
                │                                         │
                │  ingestion_daemon (long-running)        │
                │  └─► process_upload                     │
                │      ├─► auto-approve to                │
                │      │   data/labeled/embedder/<CLASS>/ │
                │      └─► quarantine for review          │
                │                                         │
                │  auto_retrain (cron, nightly)           │
                │  └─► classifier.train                   │
                │      └─► bump_version → manifest.json   │
                └─────────────────────────────────────────┘
```

---

## Roles

### AR app (Unity)

- Live camera + ensemble inference (`predict_class` + ResNet-18 ONNX)
- When `ensemble.confidence < 0.6` OR `agreement == "disagree"` OR
  `agreement == "neither"`, surface an unobtrusive banner:
  > *"Not sure — tap to help train"*
- On tap:
  1. Tap-select if multiple connector bounding boxes
  2. Show class picker
  3. Capture last 2-3 seconds (rolling buffer) + record next 2-3 seconds
  4. Extract ~20 frames at 4 fps locally (smaller upload than the .mp4)
  5. POST `/uploads` with `frames[]` + `claimed_class` + `device_id` +
     `capture_reason="low_confidence"`
- On launch:
  - GET `/model/version`
  - If server version > local version: GET `/model/weights` + `/model/labels`,
    swap in the new model on next inference

### Relay server

Built in `rfconnectorai/server/app.py` (FastAPI + uvicorn). Stateless.
Reads model files off disk, writes uploads to a directory.

Endpoints:

| Path | Auth | Purpose |
|------|------|---------|
| `POST /uploads` | yes | Multipart frames + manifest fields, atomic move into `incoming/<upload_id>/` |
| `GET /model/version` | no  | Cheap poll endpoint — `{"version": int}` |
| `GET /model/latest` | yes | Full manifest with weights URL + sha256 |
| `GET /model/weights` | yes | Returns weights binary as octet-stream |
| `GET /model/labels` | yes | Returns labels.json |
| `GET /healthz` | no | Liveness probe |

Auth: shared `X-Device-Token` header against `RFCAI_DEVICE_TOKEN` env var.
Server fails closed (503) if the env var is unset.

### Training machine

Two long-running pieces:

1. **`scripts/ingestion_daemon.py`** — polls `incoming/` every N seconds.
   For each `<upload_id>/` with a `.ready` sentinel and no `_processed.json`:
   - Reads `manifest.json` for the user's `claimed_class`
   - Runs the ensemble averager across the upload's frames
   - Routes to `data/labeled/embedder/<CLASS>/` (auto-approve) or
     `data/quarantine/<CLASS>/<upload_id>/` (engineer review)
   - Writes a `_processed.json` sidecar so re-runs skip handled directories

2. **`scripts/auto_retrain.py`** — cron-scheduled (e.g. nightly). Skips
   if delta-since-last-train is below `--min-new-samples`. Otherwise
   runs the classifier trainer, which auto-bumps the version + refreshes
   `manifest.json`. Relay's `/model/version` endpoint then advertises the
   new version on the next poll.

---

## API contract for the AR app team

### `POST /uploads`

Multipart/form-data:

| Field           | Type             | Required | Notes |
|-----------------|------------------|----------|-------|
| `claimed_class` | string           | yes      | One of: SMA-M, SMA-F, 3.5mm-M, 3.5mm-F, 2.92mm-M, 2.92mm-F, 2.4mm-M, 2.4mm-F |
| `device_id`     | string           | yes      | Stable per-device identifier (e.g. iOS `identifierForVendor`) |
| `capture_reason`| string           | no       | `"manual"` (settings page) or `"low_confidence"` (inline correction); default `"low_confidence"` |
| `frames[]`      | repeated file    | yes (1-200) | JPEG/PNG/WEBP. Recommend 20 frames at 4 fps from a 5s capture. |

Header: `X-Device-Token: <shared secret>`

Response (200):
```json
{"upload_id": "20260428T120033_a8c7e1", "n_frames_received": 20, "claimed_class": "2.4mm-M"}
```

### `GET /model/version` (no auth)

```json
{"version": 47}
```

App caches the latest fetched version locally; if `server.version > local.version`,
fetch new model.

### `GET /model/latest` (auth required)

```json
{
  "version": 47,
  "weights_filename": "weights.0047.pt",
  "labels_filename": "labels.json",
  "weights_sha256": "ab12...",
  "labels_sha256": "cd34...",
  "trained_at": "2026-04-28T02:00:00Z",
  "weights_url": "/model/weights",
  "labels_url": "/model/labels"
}
```

App downloads `weights_url` + `labels_url`, verifies sha256, swaps on
next inference start.

---

## Deploy

### Relay server (your existing server)

```bash
# 1. Sync this repo to the server (the relay only needs rfconnectorai/server/)
# 2. Install deps:
pip install fastapi 'uvicorn[standard]' python-multipart

# 3. Set env vars:
export RFCAI_INCOMING_DIR=/srv/rfcai/incoming
export RFCAI_MODEL_DIR=/srv/rfcai/models/connector_classifier
export RFCAI_DEVICE_TOKEN="$(openssl rand -hex 32)"

# 4. Run:
uvicorn rfconnectorai.server.app:app --host 0.0.0.0 --port 8000
```

Behind nginx with TLS in production. The directories `RFCAI_INCOMING_DIR`
and `RFCAI_MODEL_DIR` need to be on a filesystem the training machine
also has access to (NFS / rsync / shared volume).

### Training machine (this PC)

```bash
# Daemon (always running):
python -m scripts.ingestion_daemon \
    --incoming-dir /shared/rfcai/incoming \
    --labeled-root data/labeled/embedder \
    --quarantine-root data/quarantine \
    --classifier-dir models/connector_classifier \
    --interval 5

# Cron retrain (e.g. /etc/cron.d/rfcai or Windows Task Scheduler):
0 2 * * *  python -m scripts.auto_retrain \
    --data-dir data/labeled/embedder \
    --model-dir models/connector_classifier \
    --min-new-samples 20 \
    --epochs 8
```

After every successful retrain, copy `models/connector_classifier/` to
the relay's `RFCAI_MODEL_DIR` (rsync nightly or symlink-via-NFS).

### AR app (Unity)

- Build a `RelayClient` MonoBehaviour around `UnityWebRequest`
- Cache the current model version in `PlayerPrefs`
- On `Start()`: hit `/model/version`; if newer, download
- On low-confidence ensemble result: enqueue an upload via the same client

The Unity work is separate from this repo — point me at the Unity codebase
when you're ready and I'll sketch the `RelayClient` against this same API.

---

## Operational thresholds

| Threshold                     | Default | Where set                       | Why |
|-------------------------------|--------:|---------------------------------|-----|
| Auto-approve confidence       |    0.70 | `IngestionConfig.approve_confidence` | Conservative; quarantine borderline |
| Auto-approve agree fraction   |    0.50 | `IngestionConfig.approve_agree_fraction` | At least half the frames must "agree" |
| Min new samples for retrain   |      20 | `auto_retrain --min-new-samples` | Skip noise; wait for real data |
| App low-confidence prompt     |    0.60 | App-side (not in repo yet)       | When to show "tap to help train" |
| Max upload size               |  100 MB | `RFCAI_MAX_UPLOAD_BYTES`         | Reject oversized uploads |
| Max frames per upload         |     200 | hard-coded in `app.py`           | Reject batch-of-batches; also prevents memory blow-up |

Bump these per-environment as you observe real failure rates.

---

## What this gives you

- **Continuous improvement**: every time the model is uncertain in the field, that exact case becomes training data
- **Self-balancing dataset**: classes the model already nails generate fewer prompts; weak classes generate more
- **Human-in-the-loop only when it helps**: auto-approve handles the easy cases, quarantine catches the hard ones
- **Tamper-resistant**: the user's claim is just one of three signals (measurement + classifier + claim); a wrong label gets caught by the cross-check
- **OTA updates**: phones automatically pick up improved models without an app store update
- **Versioned**: every model snapshot is preserved (`weights.NNNN.pt`), so rollback is `cp weights.0046.pt weights.latest.pt`

---

## What's NOT built yet

- **Unity-side `RelayClient`** — separate workstream, point me at the Unity codebase when ready
- **TLS / nginx config** — production hardening, your call
- **Quarantine review UI** — engineers review via filesystem + the existing Manage Data Streamlit page right now; could be its own page later
- **Rate limiting per device** — straightforward FastAPI middleware if abuse becomes a concern
- **Rollback CLI** — `python -m rfconnectorai.classifier.rollback --to-version 46` could be a one-screen script if/when needed
- **Cloud GPU retrain** — local CPU is fine for now; if the dataset grows past ~5000 images / 16 classes, rent a T4 for nightly
