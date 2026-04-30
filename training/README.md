# RF Connector AI — Training Pipeline

[![tests](https://github.com/probably-on-fire/hotdogornot/actions/workflows/tests.yml/badge.svg)](https://github.com/probably-on-fire/hotdogornot/actions/workflows/tests.yml)

Identify and measure RF coaxial connectors (SMA, 3.5mm, 2.92mm, 2.4mm — male
& female) from a phone camera. Combines a **geometry-grounded measurement
pipeline** (no training required, sub-millimeter accuracy with an ArUco
scale marker) with a **fine-tuned ResNet-18 classifier** (works on any
image, including non-perpendicular product photos).

The two predictors run independently and cross-check each other:
- **Agreement** → high confidence
- **Disagreement** → app prompts the user to recapture

Both targets the Unity AR app under `unity/RFConnectorAR`.

---

## Architecture

```
                                    ┌─ ArUco marker scale  ─────┐
                                    │                            │
   image ──► hex_detector ──► aperture_detector ──► family/gender ──► class_predictor
                                    │                                    │
                                    └─ thread-pitch FFT (backup scale)   │
                                                                         │
   image ──► ConnectorClassifier (ResNet-18) ──────────────────────────► │
                                                                         │
                                          frames ──► frame_averager ──► AveragedPrediction
```

Two complementary pipelines:

**1. Measurement pipeline** (`rfconnectorai/measurement/`)

  - `hex_detector` — finds the coupling-nut hex contour in pixels
  - `aperture_detector` — finds the inner bore diameter
  - `family_detector` — SMA (PTFE dielectric visible) vs precision (air)
  - `gender_detector` — male (pin protrudes) vs female (socket recessed)
  - `aruco_detector` — finds a 25 mm ArUco marker for absolute scale
  - `thread_pitch_scale` — backup absolute scale via FFT on the threaded
    coupling section (standardized pitch: SMA/3.5/2.92 = 0.706 mm; 2.4 = 0.635 mm)
  - `class_predictor` — combines the above. `require_aruco=True` uses the
    marker as the only scale source; `require_aruco=False` falls back to
    enumerating both standard hex sizes (5/16″ and 1/4″).
  - `frame_averager` — runs `predict_class` across many frames, MAD-filters
    outliers, votes on family/gender/class. Returns one consensus
    `AveragedPrediction` with confidence and per-class vote breakdown.

**2. ResNet-18 classifier** (`rfconnectorai/classifier/`)

  - `train.py` — fine-tune ImageNet-pretrained ResNet-18 on labeled folders
  - `predict.py` — load weights, predict class on any image with full
    per-class softmax. Works on non-perpendicular images that the
    measurement pipeline can't fire on.

**Synthetic data** (`rfconnectorai/synthetic/`)

  - `procedural_connectors.py` — emits parametric GLB meshes for all 8
    classes from `configs/datasheet_dimensions.yaml`. No vendor STEPs needed.
  - `face_renderer.py` — PIL-based photo-style mating-face renderer with
    smooth metallic shading, specular highlights, depth-cued bore, and
    perspective tilt. Used for synthetic eval data.
  - `angled_renderer.py` — pyrender 3D off-axis renderer for any camera
    elevation/azimuth. Visualizes the GLB meshes in the round; not used by
    the measurement detectors (which assume perpendicular views).

**Data ingestion** (`rfconnectorai/data_fetch/`)

  - `ddg_images.py` — DuckDuckGo image search fetcher (no API key)
  - `google_cse.py` — Google Custom Search JSON API (needs key, see file)
  - `video_frames.py` — extract frames from a capture video at a target fps

---

## Setup

```bash
cd training
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows
.venv/bin/pip install -e ".[dev]"          # macOS/Linux
```

For Google Custom Search image fetching (optional), copy `.env.example`
to `.env` and fill in `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`. Setup steps are
in `rfconnectorai/data_fetch/google_cse.py`.

---

## Streamlit demo + management UI

```bash
.venv/Scripts/python.exe -m streamlit run scripts/demo_app.py
```

Opens a two-page browser app at `http://localhost:8501` (also live at
`https://aired.com/demo/`):

- **Demo (`demo_app.py`)** — take a photo (or upload), POST to the
  `/predict` relay, render bounding boxes with confidence on the captured
  image, then either confirm or correct the class. Confirmed/corrected
  samples flow back into the training-data folders.
- **Training Data (`pages/1_Training_Data.py`)** — three tabs:
  - *Upload + Label* — drop a connector video → auto-detect each connector
    in each frame → pick a class per crop → save into
    `data/labeled/embedder/<CLASS>/`
  - *Review* — walk through a class folder with the current classifier's
    prediction alongside the on-disk label; disagreements bubble to the
    top; bulk keep / delete / move-to-class
  - *Train* — fine-tune ResNet-18 on the current labeled folders, then
    test on a sample image with full per-class probabilities

---

## Data layout

```
data/
  labeled/embedder/          ← labeled crops the classifier trains on (gitignored)
    SMA-M/video_NNNN.jpg       extracted by the Video Labeler page
    SMA-F/video_NNNN.jpg
    3.5mm-M/...
    ...
  test_holdout/              ← hand-verified golden test set, never trained on
    SMA-M/IMG_*.jpg
    ...
  archive/scraped/           ← old DDG/Google/Chrome scraped images (gitignored,
                               kept out of training; ignore unless reviving)
  synthetic_faces/           ← PIL-rendered frontal mating faces (gitignored, regenerable)
  synthetic_angled/          ← pyrender 3D off-axis renders (gitignored, regenerable)
  reference/pasternack/      ← committed: small set of vendor reference photos
  cad/verified/<CLASS>.glb   ← procedural GLB meshes (gitignored, regenerable)
```

---

## Running the pieces

### Generate synthetic data

```bash
.venv/Scripts/python.exe -m rfconnectorai.synthetic.procedural_connectors
.venv/Scripts/python.exe -m rfconnectorai.synthetic.face_renderer --per-class 30
.venv/Scripts/python.exe -m rfconnectorai.synthetic.angled_renderer --per-class 20
```

### Fetch real images

Use the **Fetch Images** Streamlit page, or:

```python
from pathlib import Path
from rfconnectorai.data_fetch.ddg_images import fetch_images
fetch_images("SMA male connector", Path("data/labeled/embedder/SMA-M"), n=30)
```

### Extract frames from a capture video

```python
from pathlib import Path
from rfconnectorai.data_fetch.video_frames import extract_frames
extract_frames(
    video_path=Path("captures/SMA-M.mp4"),
    out_dir=Path("data/labeled/embedder/SMA-M"),
    fps_target=2.0,
)
```

### Train the classifier

```bash
.venv/Scripts/python.exe -m rfconnectorai.classifier.train \
    --data-dir data/labeled/embedder \
    --out-dir models/connector_classifier \
    --epochs 8
```

Or via the Train Classifier Streamlit page.

### Predict on a single image (measurement pipeline)

```python
import cv2
from rfconnectorai.measurement.class_predictor import predict_class

img = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
pred = predict_class(img, require_aruco=True)
print(pred.class_name, pred.aperture_mm)
```

### Predict averaged across video frames

```python
from rfconnectorai.measurement.frame_averager import average_predictions

frames = [...]   # list of HxWx3 RGB arrays
result = average_predictions(frames, require_aruco=True)
print(result.class_name, result.confidence, result.per_class_votes)
```

### Predict with the trained classifier

```python
from rfconnectorai.classifier.predict import ConnectorClassifier

clf = ConnectorClassifier.load("models/connector_classifier")
pred = clf.predict(img)
print(pred.class_name, pred.confidence, pred.probabilities)
```

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

122 tests covering each detector, the class predictor, the frame averager,
the video extractor, the classifier (round-trip train+predict), the
thread-pitch FFT, and the synthetic renderers.

---

## Capture protocol (for real video / phone photos)

See `docs/capture_protocol.md` for the recommended setup: distance, lighting,
ArUco marker placement, frames-per-class targets.

---

## Reference docs

- `docs/superpowers/specs/` — design specs for each major architecture pivot
- `docs/superpowers/plans/` — implementation plans
- `docs/measurement_baseline_findings.md` — first measurement-pipeline accuracy notes
- `docs/hex_measurement_prototype_findings.md` — hex detection R&D
- `docs/printables/aruco_marker_25mm.png` — print-ready 25mm ArUco marker for capture
