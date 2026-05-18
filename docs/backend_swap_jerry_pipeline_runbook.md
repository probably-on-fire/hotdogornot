# Backend swap: Jerry's YOLO+EffNet pipeline → /predict

Goal: make `/rfcai/predict` serve Jerry's YOLO11n + EfficientNetV2-S
pipeline instead of v18 ResNet-18, so the existing Flutter app
automatically gets the ~94% accuracy we measured on our 35-image
holdout (vs v18's 68.6%).

Status as of 2026-05-18:
- Jerry's models live in `../hotdogornot/exports/web/models/` (this
  Windows machine), 91 MB combined.
- Our `predict_service.py` currently uses edge-density Hough detector
  + ResNet-18 classifier. Both will be replaced.
- The Flutter app already sends `bbox` and accepts an empty bbox if
  none returned, so the API contract doesn't change.

## Pre-flight (verify environment)

```bash
# From the LAN box (192.168.20.235), as user chris:
ssh chris@192.168.20.235
# password: Elad9651!

# Confirm onnxruntime is installed in the rfcai venv
sudo -u rfcai /opt/rfcai/training/.venv/bin/python -c \
  "import onnxruntime; print('ort', onnxruntime.__version__)"
# Expected: ort 1.x.x — if ModuleNotFoundError, install:
# sudo -u rfcai /opt/rfcai/training/.venv/bin/pip install onnxruntime
```

## 1. Copy Jerry's models to the box

From your Mac/Windows (where this repo is checked out):

```bash
scp ../hotdogornot/exports/web/models/detector.onnx \
    ../hotdogornot/exports/web/models/classifier.onnx \
    ../hotdogornot/exports/web/models/classifier_labels.json \
    ../hotdogornot/exports/web/thresholds.json \
    chris@192.168.20.235:/tmp/jerry/
```

On the box:

```bash
sudo mkdir -p /opt/rfcai/repo/training/models/jerry
sudo chown -R rfcai:rfcai /opt/rfcai/repo/training/models/jerry
sudo mv /tmp/jerry/*.onnx /tmp/jerry/*.json \
        /opt/rfcai/repo/training/models/jerry/
sudo chown rfcai:rfcai /opt/rfcai/repo/training/models/jerry/*
ls -la /opt/rfcai/repo/training/models/jerry/
# Expect: classifier.onnx (~77 MB), detector.onnx (~10 MB),
#         classifier_labels.json, thresholds.json
```

Also: the predict-service systemd unit hardens `ProtectSystem=strict`
with `ReadWritePaths=/home/rfcai /opt/rfcai/repo/training/data
/opt/rfcai/repo/training/models`. The `models` path is already in
ReadWritePaths, so the service can read the new files without a unit
edit.

## 2. Apply the code patch

We add a new pipeline class that the predict service can opt into via
env var. Keeps the v18 path live as a fallback (set the env var back
to 0 to revert).

Create `training/rfconnectorai/pipeline/jerry_pipeline.py` in this repo
(I'll commit this as a follow-up — for now you can paste it on the
box directly):

```python
"""YOLO11n detector + EfficientNetV2-S classifier — replicates Jerry's
exports/web/app.js pipeline in Python for the FastAPI predict service."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class JerryPipeline:
    def __init__(self, model_dir: Path):
        self.det = ort.InferenceSession(
            str(model_dir / "detector.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.cls = ort.InferenceSession(
            str(model_dir / "classifier.onnx"),
            providers=["CPUExecutionProvider"],
        )
        labels = json.loads((model_dir / "classifier_labels.json").read_text())
        self.class_names = labels["class_names"]
        self.cls_size = labels.get("input_size", 384)
        self.thresholds = json.loads((model_dir / "thresholds.json").read_text())
        self.det_size = 640
        self.box_min = float(self.thresholds.get("box_min", 0.25))

    def _letterbox(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        scale = min(self.det_size / w, self.det_size / h)
        nw, nh = round(w * scale), round(h * scale)
        dx, dy = (self.det_size - nw) // 2, (self.det_size - nh) // 2
        canvas = np.full((self.det_size, self.det_size, 3), 128, dtype=np.uint8)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[dy:dy + nh, dx:dx + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...]), scale, dx, dy

    def detect(self, bgr: np.ndarray):
        tensor, scale, dx, dy = self._letterbox(bgr)
        out = self.det.run(None, {self.det.get_inputs()[0].name: tensor})[0][0]
        # YOLOv8/v11 format: (4+nc, num_boxes), nc=1
        cx, cy, w, h = out[0], out[1], out[2], out[3]
        scores = out[4:].max(axis=0)
        keep = scores >= self.box_min
        if not keep.any():
            return []
        cx, cy, w, h, sc = cx[keep], cy[keep], w[keep], h[keep], scores[keep]
        orig_h, orig_w = bgr.shape[:2]
        boxes = []
        for i in range(len(sc)):
            x1 = max(0, (cx[i] - w[i]/2 - dx) / scale)
            y1 = max(0, (cy[i] - h[i]/2 - dy) / scale)
            x2 = min(orig_w, (cx[i] + w[i]/2 - dx) / scale)
            y2 = min(orig_h, (cy[i] + h[i]/2 - dy) / scale)
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(sc[i])))
        # NMS
        boxes.sort(key=lambda b: -b[4])
        kept = []
        for b in boxes:
            ok = True
            for k in kept:
                xa, ya, xb, yb = max(b[0], k[0]), max(b[1], k[1]), min(b[2], k[2]), min(b[3], k[3])
                inter = max(0, xb - xa) * max(0, yb - ya)
                a = (b[2] - b[0]) * (b[3] - b[1])
                c = (k[2] - k[0]) * (k[3] - k[1])
                union = a + c - inter
                if union > 0 and inter / union > 0.45:
                    ok = False
                    break
            if ok:
                kept.append(b)
        return kept

    def classify_crop(self, bgr_crop: np.ndarray):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.cls_size, self.cls_size),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
        logits = self.cls.run(None, {self.cls.get_inputs()[0].name: tensor})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return probs

    def run(self, bgr: np.ndarray):
        boxes = self.detect(bgr)
        if not boxes:
            return []
        out = []
        for (x1, y1, x2, y2, score) in boxes[:4]:
            crop = bgr[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            probs = self.classify_crop(crop)
            idx = int(probs.argmax())
            out.append({
                "class_name": self.class_names[idx],
                "confidence": float(probs[idx]),
                "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)},
                "bbox": {
                    "x": int(x1), "y": int(y1),
                    "w": int(x2 - x1), "h": int(y2 - y1),
                },
                "box_score": float(score),
            })
        return out
```

Then in `training/rfconnectorai/server/predict_service.py`, add at module
load time:

```python
_JERRY = None
if os.environ.get("RFCAI_USE_JERRY_PIPELINE") == "1":
    from rfconnectorai.pipeline.jerry_pipeline import JerryPipeline
    _JERRY = JerryPipeline(Path(os.environ["RFCAI_JERRY_MODEL_DIR"]))
```

And in `_classify_frame(bgr)`, near the top:

```python
if _JERRY is not None:
    return _JERRY.run(bgr)
```

## 3. Configure the env var + restart

On the box:

```bash
echo "RFCAI_USE_JERRY_PIPELINE=1" | sudo tee -a /etc/default/rfcai-predict
echo "RFCAI_JERRY_MODEL_DIR=/opt/rfcai/repo/training/models/jerry" \
    | sudo tee -a /etc/default/rfcai-predict
sudo systemctl restart rfcai-predict
# Wait for warmup
until curl -sf http://127.0.0.1:8503/healthz >/dev/null; do sleep 3; done
```

## 4. Smoke test

```bash
# From your Mac/Windows
curl -sS -X POST \
  -H "X-Device-Token: 66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1" \
  -F "image=@tmp_holdout_local/SMA-M/holdout-carve-2026-05-17_photo_2026-05-16_CAP1083963664233658672.jpg" \
  https://aired.com/rfcai/predict | python -m json.tool
# Expect: predictions[0].class_name == "SMA-M" with high confidence
```

## 5. Re-run the holdout eval against the new backend

```bash
# Back the v18 baseline before overwriting
cp tmp_baseline_eval.md tmp_baseline_eval_v18.md
cp tmp_baseline_eval.json tmp_baseline_eval_v18.json

# Re-run against the new /predict
python -u tmp_eval.py
mv tmp_baseline_eval.md tmp_baseline_jerry_backend.md
mv tmp_baseline_eval.json tmp_baseline_jerry_backend.json

# Compare
diff tmp_baseline_eval_v18.md tmp_baseline_jerry_backend.md | head -50
```

Expected: ~94% full accuracy, matching the local-Python eval at
`tmp_partner_eval.md`.

## Rollback

```bash
sudo sed -i '/RFCAI_USE_JERRY_PIPELINE/d; /RFCAI_JERRY_MODEL_DIR/d' \
    /etc/default/rfcai-predict
sudo systemctl restart rfcai-predict
```

## After deploy

- Build the new Flutter APK (reticle crop + zoom + tighter threshold)
  and side-load on your phone.
- Run live tests against several connector types. With the backend
  now serving Jerry's pipeline, predictions should match the 94%
  benchmark.
- If accuracy is good live, capture training data via the Contribute
  tab. Every uploaded file is now a reticle-cropped square (filename
  starts with `reticle_crop_`), matching the inference distribution.
  Prioritize SMA-M, 3.5mm-M, 2.4mm-M/F to fill gaps.
