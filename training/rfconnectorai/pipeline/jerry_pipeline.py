"""YOLO11n detector + EfficientNetV2-S classifier pipeline.

Python port of the ONNX inference path that ships in the partner's
on-device Capacitor app (see exports/web/app.js in trextrader/hotdogornot).
Plugs into predict_service.py via the RFCAI_USE_JERRY_PIPELINE env var.

Validated 2026-05-18: this pipeline scored 94.3% Full / 94.3% Family /
100.0% Gender on our 35-image holdout vs v18's 68.6%/68.6%/91.4%. See
tmp_partner_eval.md for the per-image breakdown.

Inference flow (matches app.js semantics exactly):
  1. Letterbox the frame to 640x640, gray padding, normalize to [0,1]
     NCHW. Run YOLO11n; output is (1, 4+nc, num_boxes) with nc=1.
  2. Filter boxes by box_min (default 0.25 from thresholds.json),
     NMS at IoU=0.45, sort by score.
  3. Crop the top box from the ORIGINAL frame (not the letterboxed
     one — we map back through scale/dx/dy).
  4. Resize the crop to 384x384, normalize [0,1] NCHW. The classifier
     ONNX bakes ImageNet mean/std inside the graph, so we do NOT
     normalize again — that's a double-normalization bug.
  5. Softmax the logits; argmax is the prediction. Family/gender
     decomposition matches the rest of predict_service's API contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# Inference-time constants. These match the partner's app.js / thresholds.json
# defaults; treat them as the deploy-time contract.
DET_SIZE = 640
NMS_IOU_THRESHOLD = 0.45
DEFAULT_BOX_MIN = 0.25
CLS_SIZE_DEFAULT = 384
MAX_RETURN = 4


def _letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    # Use PIL.BILINEAR for resampling — matches the partner's training-time
    # preprocessing (their training stack is torchvision/PIL). cv2.INTER_LINEAR
    # is *also* bilinear but differs in sampling-center convention, producing
    # subtly different pixel values that cost ~14pts of accuracy on fine-pitch
    # female connectors. Verified 2026-05-18 against tmp_partner_eval.md.
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(pil, (dx, dy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])
    return tensor, scale, dx, dy


def _nms(boxes: list[tuple], iou_threshold: float) -> list[tuple]:
    """boxes: (x1, y1, x2, y2, score). Sorted by descending score."""
    kept: list[tuple] = []
    for b in boxes:
        ok = True
        for k in kept:
            xa = max(b[0], k[0])
            ya = max(b[1], k[1])
            xb = min(b[2], k[2])
            yb = min(b[3], k[3])
            inter = max(0.0, xb - xa) * max(0.0, yb - ya)
            ba = (b[2] - b[0]) * (b[3] - b[1])
            ka = (k[2] - k[0]) * (k[3] - k[1])
            union = ba + ka - inter
            if union > 0 and inter / union > iou_threshold:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


class JerryPipeline:
    """Detector + classifier wrapper for predict_service.

    Constructed once at module load; `run(bgr)` is called per frame.
    Returns a list of prediction dicts in the same shape predict_service
    emits today (class_name, confidence, probabilities, bbox, family,
    gender, family_confidence, gender_confidence). spec lookup happens
    in the caller.
    """

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        det_path = model_dir / "detector.onnx"
        cls_path = model_dir / "classifier.onnx"
        labels_path = model_dir / "classifier_labels.json"
        thresholds_path = model_dir / "thresholds.json"

        for p in (det_path, cls_path, labels_path):
            if not p.exists():
                raise FileNotFoundError(f"jerry pipeline missing {p}")

        self.det = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"]
        )
        self.cls = ort.InferenceSession(
            str(cls_path), providers=["CPUExecutionProvider"]
        )
        labels = json.loads(labels_path.read_text())
        self.class_names: list[str] = labels["class_names"]
        self.cls_size: int = labels.get("input_size", CLS_SIZE_DEFAULT)

        # thresholds.json is optional — fall back to defaults if absent.
        if thresholds_path.exists():
            self.thresholds = json.loads(thresholds_path.read_text())
        else:
            self.thresholds = {}
        self.box_min: float = float(self.thresholds.get("box_min", DEFAULT_BOX_MIN))

        self._det_input = self.det.get_inputs()[0].name
        self._cls_input = self.cls.get_inputs()[0].name

    def _detect(self, bgr: np.ndarray) -> list[tuple]:
        tensor, scale, dx, dy = _letterbox(bgr, DET_SIZE)
        out = self.det.run(None, {self._det_input: tensor})[0]
        # Output: (1, 4+nc, num_boxes). Single-class detector, nc=1.
        data = out[0]
        cx, cy, w, h = data[0], data[1], data[2], data[3]
        # Take the max class score per box (single class, so this is just
        # row 4 — but write it as a max so it generalizes if Jerry's
        # detector ever moves to multi-class).
        scores = data[4:].max(axis=0)
        keep_mask = scores >= self.box_min
        if not keep_mask.any():
            return []
        cx = cx[keep_mask]
        cy = cy[keep_mask]
        w = w[keep_mask]
        h = h[keep_mask]
        sc = scores[keep_mask]

        orig_h, orig_w = bgr.shape[:2]
        # Map letterbox coords → original-image coords.
        boxes: list[tuple] = []
        for i in range(len(sc)):
            x1 = max(0.0, float((cx[i] - w[i] / 2 - dx) / scale))
            y1 = max(0.0, float((cy[i] - h[i] / 2 - dy) / scale))
            x2 = min(float(orig_w), float((cx[i] + w[i] / 2 - dx) / scale))
            y2 = min(float(orig_h), float((cy[i] + h[i] / 2 - dy) / scale))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append((x1, y1, x2, y2, float(sc[i])))

        boxes.sort(key=lambda b: -b[4])
        return _nms(boxes, NMS_IOU_THRESHOLD)

    def _classify_crop(self, bgr_crop: np.ndarray) -> np.ndarray:
        # PIL.BILINEAR to match training-time preprocessing (see _letterbox).
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (self.cls_size, self.cls_size), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])
        logits = self.cls.run(None, {self._cls_input: tensor})[0][0]
        # Numerically-stable softmax.
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def run(self, bgr: np.ndarray) -> list[dict]:
        boxes = self._detect(bgr)
        if not boxes:
            return []
        results: list[dict] = []
        for (x1, y1, x2, y2, score) in boxes[:MAX_RETURN]:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            crop = bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            probs = self._classify_crop(crop)
            idx = int(probs.argmax())
            cls_name = self.class_names[idx]
            family, gender = (
                cls_name.rsplit("-", 1) if "-" in cls_name else (cls_name, "")
            )
            # Per-axis confidence: sum of probabilities sharing the
            # same family / gender. Mirrors _decompose_probabilities
            # in predict_service.py.
            fam_conf = 0.0
            gen_conf = 0.0
            for name, p in zip(self.class_names, probs):
                if "-" in name:
                    f, g = name.rsplit("-", 1)
                    if f == family:
                        fam_conf += float(p)
                    if g == gender:
                        gen_conf += float(p)
            results.append({
                "class_name": cls_name,
                "confidence": float(probs[idx]),
                "probabilities": {
                    n: float(p) for n, p in zip(self.class_names, probs)
                },
                "bbox": {
                    "x": ix1, "y": iy1,
                    "w": ix2 - ix1, "h": iy2 - iy1,
                },
                "family": family,
                "gender": gender,
                "family_confidence": fam_conf,
                "gender_confidence": gen_conf,
                "box_score": float(score),
                "_diag": {"crop_source": "yolo11n"},
            })
        return results
