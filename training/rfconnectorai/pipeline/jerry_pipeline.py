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

    def __init__(self, model_dir: Path, extra_model_dirs: list[Path] | None = None):
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

        # Ensemble support: optional list of additional hybrid bundle dirs.
        # Each must share the same classifier_labels.json class order — we
        # validate that at load time. Detector + thresholds come from the
        # primary bundle; only the classifier ONNX is loaded from extras
        # and ensembled by averaging softmax across all classifiers.
        # See the overnight eval (2026-05-26): 2-way ensemble of combined_v2
        # + jerry_full_nobal scores 88.4% Full / 95.3% Gender vs single-model
        # 81.4% / 90.7%.
        # Best-CLS-conf box selection: classify all detected boxes (with a
        # looser detector threshold) and re-rank by classifier confidence
        # rather than YOLO score. Enable by setting RFCAI_BEST_CLS_CONF_BOX=1.
        # When enabled, also overrides box_min to the lower 0.05 floor. See
        # [[best-cls-conf-box-selection]] memory for the win.
        import os as _os
        # Accept the same truthy strings the rest of predict_service does
        # ("1", "true", "yes", "on", case-insensitive) so deploys don't
        # silently fall back to off because someone typed "True" or "yes".
        def _truthy(name: str) -> bool:
            return _os.environ.get(name, "").strip().lower() in ("1","true","yes","on")
        self._best_cls_conf: bool = _truthy("RFCAI_BEST_CLS_CONF_BOX")
        if self._best_cls_conf:
            self.box_min = 0.05

        # Reticle-region box filter: drop YOLO boxes whose center is outside
        # the central 60%-min-dim square (the Flutter reticle crop region).
        # Catches sloppy right-edge / corner detections on full-frame uploads;
        # mostly a no-op on already-reticle-cropped phone uploads. If no boxes
        # survive, fall back to the reticle square itself as a synthetic box
        # so the classifier still runs. Off by default (preserves prod inference).
        self._reticle_filter: bool = _truthy("RFCAI_RETICLE_REGION_FILTER")

        # Ensemble-disagreement abstention: when running an ensemble (extras
        # set), skip any candidate box where the per-model top-1 classes
        # disagree. If all boxes are dropped, return [] (the existing
        # "no prediction" path). Catches uncorrelated errors; does NOT help
        # when models are correlated-wrong (measured 2026-05-28 phone-realistic
        # eval: 0 of 2 confidently-wrong cases were caught). No-op for single
        # model. Off by default.
        self._disagree_abstain: bool = _truthy("RFCAI_ENSEMBLE_DISAGREE_ABSTAIN")

        self.extra_cls: list[ort.InferenceSession] = []
        self._extra_cls_input: list[str] = []
        for extra in (extra_model_dirs or []):
            extra = Path(extra)
            ex_cls = extra / "classifier.onnx"
            ex_labels = extra / "classifier_labels.json"
            for p in (ex_cls, ex_labels):
                if not p.exists():
                    raise FileNotFoundError(f"jerry pipeline extra missing {p}")
            ex_label_data = json.loads(ex_labels.read_text())
            if ex_label_data["class_names"] != self.class_names:
                raise RuntimeError(
                    f"extra bundle {extra} class order disagrees with primary: "
                    f"{ex_label_data['class_names']} vs {self.class_names}"
                )
            sess = ort.InferenceSession(
                str(ex_cls), providers=["CPUExecutionProvider"]
            )
            self.extra_cls.append(sess)
            self._extra_cls_input.append(sess.get_inputs()[0].name)

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
        nms_boxes = _nms(boxes, NMS_IOU_THRESHOLD)

        if self._reticle_filter and nms_boxes:
            orig_h, orig_w = bgr.shape[:2]
            side = 0.6 * min(orig_w, orig_h)
            rx1 = (orig_w - side) / 2.0
            ry1 = (orig_h - side) / 2.0
            rx2 = rx1 + side
            ry2 = ry1 + side
            kept = [b for b in nms_boxes
                    if rx1 <= (b[0]+b[2])/2 <= rx2
                    and ry1 <= (b[1]+b[3])/2 <= ry2]
            if not kept:
                # Synthetic fallback: the reticle square itself. Score 0.5
                # so it sorts behind real high-confidence YOLO boxes when
                # they exist, but ahead of the empty-result path.
                kept = [(rx1, ry1, rx2, ry2, 0.5)]
            return kept
        return nms_boxes

    def _classify_crop(self, bgr_crop: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        # Returns (averaged_probs, per_model_probs). per_model_probs has one
        # full softmax array per ensemble member (always >= 1; primary at
        # index 0). The disagreement-abstention path in run() needs the full
        # arrays so each model can pick its own best-CLS-conf box; the
        # regular path only uses the averaged probs.
        # PIL.BILINEAR to match training-time preprocessing (see _letterbox).
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (self.cls_size, self.cls_size), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])

        def _softmax(logits: np.ndarray) -> np.ndarray:
            e = np.exp(logits - logits.max())
            return e / e.sum()

        per_model: list[np.ndarray] = [
            _softmax(self.cls.run(None, {self._cls_input: tensor})[0][0])
        ]
        for sess, in_name in zip(self.extra_cls, self._extra_cls_input):
            per_model.append(_softmax(sess.run(None, {in_name: tensor})[0][0]))
        avg = sum(per_model) / len(per_model)
        return avg, per_model

    def run(self, bgr: np.ndarray) -> list[dict]:
        boxes = self._detect(bgr)
        if not boxes:
            return []

        # Best-CLS-conf strategy: classify each detected box and re-rank by
        # classifier confidence, not YOLO score. YOLO11n's detection-confidence
        # is well-correlated with "is this a connector-shaped region" but not
        # with "is this a good crop for classifying connector TYPE". On some
        # OOD images the top-YOLO box is a background region; lower-score
        # boxes contain the actual connector. Measured 2026-05-26: this lifts
        # holdout Full 88.4→90.7%, Gender 95.3→100%. See [[best-cls-conf-box-selection]].
        # Latency: ~2-3x classifier inference per image vs current. Enable
        # by setting RFCAI_BEST_CLS_CONF_BOX=1.
        best_cls_conf = self._best_cls_conf
        # scored carries the averaged-probs path (what we emit).
        # per_box_per_model carries the per-model arrays for the disagreement
        # check — same length as scored, same order.
        scored: list[tuple[float, float, float, float, float, np.ndarray, int]] = []
        per_box_per_model: list[list[np.ndarray]] = []
        for (x1, y1, x2, y2, score) in boxes[: (8 if best_cls_conf else MAX_RETURN)]:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            crop = bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            probs, per_model_probs = self._classify_crop(crop)
            idx = int(probs.argmax())
            scored.append((x1, y1, x2, y2, score, probs, idx))
            per_box_per_model.append(per_model_probs)
        if not scored:
            return []

        # Image-level ensemble-disagreement abstention. Each model picks its
        # OWN best box (the one with highest top-1 confidence under that
        # model's classifier), then we compare those independent top-1s.
        # This is strictly stronger than per-box agreement: it catches the
        # case where the two models pick different best boxes AND emit
        # different classes (the standalone subclass eval, 2026-05-28,
        # caught 10/10 close-up image-level disagreements vs 4/10 under
        # per-box agreement). No-op for single-model deploys.
        if self._disagree_abstain and per_box_per_model and len(per_box_per_model[0]) > 1:
            num_models = len(per_box_per_model[0])
            per_model_top1: list[int] = []
            for m_idx in range(num_models):
                best_box = max(
                    range(len(per_box_per_model)),
                    key=lambda b: float(per_box_per_model[b][m_idx].max()),
                )
                per_model_top1.append(
                    int(per_box_per_model[best_box][m_idx].argmax())
                )
            if len(set(per_model_top1)) > 1:
                return []
        if best_cls_conf:
            # Sort by classifier top-1 confidence DESC
            scored.sort(key=lambda t: -float(t[5][t[6]]))
            scored = scored[:MAX_RETURN]
        results: list[dict] = []
        for (x1, y1, x2, y2, score, probs, idx) in scored:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
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
