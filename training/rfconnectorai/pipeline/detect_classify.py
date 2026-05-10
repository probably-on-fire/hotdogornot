"""Detect-then-classify pipeline for RF connectors.

Combines the YOLO11n detector and multi-head EfficientNet classifier into a
single inference call:

    image → YOLO detect → crop each bbox → classify each crop → results

Works with PyTorch (.pt) weights for cloud/server, and with ONNX weights for
cross-platform/mobile deployment. The ONNX path uses ``onnxruntime`` and
requires only ``pillow`` and ``numpy`` (no torch).

Usage::

    from rfconnectorai.pipeline.detect_classify import DetectClassifyPipeline

    pipe = DetectClassifyPipeline.from_torch(
        detector_weights="models/detector/best.pt",
        classifier_weights="models/multihead_classifier/best.pt",
        device="cuda:0",
    )
    results = pipe.predict(image)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass
class ConnectorPrediction:
    """One detected connector and its classification."""

    bbox_xyxy: tuple[int, int, int, int]
    detection_confidence: float
    detection_class: str
    attributes: dict[str, str] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    crop: Image.Image | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "detection_confidence": round(self.detection_confidence, 4),
            "detection_class": self.detection_class,
            "attributes": self.attributes,
            "confidences": {k: round(v, 4) for k, v in self.confidences.items()},
        }


@dataclass
class PipelineResult:
    """Full prediction result for one image."""

    predictions: list[ConnectorPrediction]
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "num_detections": len(self.predictions),
        }


# ---------------------------------------------------------------------------
# Normalisation constants (ImageNet for classifier, passthrough for detector)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_crop_numpy(
    crop: Image.Image, imgsz: int = 384
) -> np.ndarray:
    """Resize + normalise a crop to (1, 3, H, W) float32 for ONNX."""
    crop = crop.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(crop, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis]  # (1, 3, H, W)


class DetectClassifyPipeline:
    """Detect connectors with YOLO, classify with multi-head model.

    Use the factory class-methods to construct:

    - :meth:`from_torch` — load ``.pt`` weights (requires torch, ultralytics)
    - :meth:`from_onnx` — load ``.onnx`` weights (requires onnxruntime)
    """

    def __init__(self, *, detector, classifier, head_vocabs, imgsz_cls, device_str,
                 _backend):
        self._detector = detector
        self._classifier = classifier
        self._head_vocabs = head_vocabs
        self._imgsz_cls = imgsz_cls
        self._device_str = device_str
        self._backend = _backend  # "torch" | "onnx"

    # ------------------------------------------------------------------
    # Factory: PyTorch weights
    # ------------------------------------------------------------------
    @classmethod
    def from_torch(
        cls,
        detector_weights: str | Path,
        classifier_weights: str | Path,
        device: str = "cpu",
        imgsz_cls: int = 384,
    ) -> "DetectClassifyPipeline":  # pragma: no cover – cloud-only
        import torch
        from ultralytics import YOLO

        from rfconnectorai.classifier.model_multihead import build_multihead_classifier

        detector = YOLO(str(detector_weights))
        ckpt = torch.load(str(classifier_weights), map_location=device)
        head_sizes = ckpt["head_sizes"]
        backbone = ckpt["backbone"]
        vocabs = ckpt["vocabs"]  # {head_name: [class_labels]}
        model = build_multihead_classifier(backbone, head_sizes)
        model.load_state_dict(ckpt["state_dict"])
        dev = torch.device(device if device != "0" else "cuda:0")
        model = model.to(dev).eval()

        return cls(
            detector=detector,
            classifier=model,
            head_vocabs=vocabs,
            imgsz_cls=imgsz_cls,
            device_str=device,
            _backend="torch",
        )

    # ------------------------------------------------------------------
    # Factory: ONNX weights
    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        detector_onnx: str | Path,
        classifier_onnx: str | Path,
        head_vocabs_json: str | Path,
        imgsz_cls: int = 384,
    ) -> "DetectClassifyPipeline":
        import onnxruntime as ort

        det_session = ort.InferenceSession(
            str(detector_onnx), providers=["CPUExecutionProvider"]
        )
        cls_session = ort.InferenceSession(
            str(classifier_onnx), providers=["CPUExecutionProvider"]
        )
        with open(head_vocabs_json, "r", encoding="utf-8") as f:
            vocabs = json.load(f)

        return cls(
            detector=det_session,
            classifier=cls_session,
            head_vocabs=vocabs,
            imgsz_cls=imgsz_cls,
            device_str="cpu",
            _backend="onnx",
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        image: Image.Image,
        *,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> PipelineResult:
        t0 = time.perf_counter()

        # 1. Detect
        detections = self._run_detector(image, conf_threshold, iou_threshold)
        t_det = time.perf_counter()

        # 2. Crop and classify each detection
        predictions: list[ConnectorPrediction] = []
        for det in detections:
            bbox = det["bbox_xyxy"]
            crop = image.crop(bbox)
            attrs, confs = self._run_classifier(crop)
            predictions.append(ConnectorPrediction(
                bbox_xyxy=bbox,
                detection_confidence=det["confidence"],
                detection_class=det["class_name"],
                attributes=attrs,
                confidences=confs,
                crop=crop,
            ))
        t_cls = time.perf_counter()

        return PipelineResult(
            predictions=predictions,
            latency_ms={
                "detector": (t_det - t0) * 1000,
                "classifier": (t_cls - t_det) * 1000,
                "total": (t_cls - t0) * 1000,
            },
        )

    # ------------------------------------------------------------------
    # Internal: detector dispatch
    # ------------------------------------------------------------------
    def _run_detector(
        self, image: Image.Image, conf: float, iou: float
    ) -> list[dict]:
        if self._backend == "torch":
            return self._detect_torch(image, conf, iou)
        return self._detect_onnx(image, conf, iou)

    def _detect_torch(
        self, image: Image.Image, conf: float, iou: float
    ) -> list[dict]:  # pragma: no cover – cloud-only
        results = self._detector(image, conf=conf, iou=iou, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(box.conf[0].item()),
                    "class_name": r.names[int(box.cls[0].item())],
                })
        return detections

    def _detect_onnx(
        self, image: Image.Image, conf: float, iou: float
    ) -> list[dict]:
        """ONNX detector inference using raw onnxruntime session."""
        # YOLO ONNX expects (1, 3, 640, 640) float32 [0, 1]
        img = image.resize((640, 640), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)

        input_name = self._detector.get_inputs()[0].name
        outputs = self._detector.run(None, {input_name: arr})

        # Ultralytics ONNX output shape: (1, N, 7) for detect
        # [x_center, y_center, w, h, conf, cls0, cls1, ...]
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]  # (N, 7+)
        detections = []
        # Scale from 640×640 back to original size
        orig_w, orig_h = image.size
        sx, sy = orig_w / 640, orig_h / 640

        for row in raw:
            obj_conf = float(row[4])
            if obj_conf < conf:
                continue
            cx, cy, w, h = row[:4]
            x1 = int((cx - w / 2) * sx)
            y1 = int((cy - h / 2) * sy)
            x2 = int((cx + w / 2) * sx)
            y2 = int((cy + h / 2) * sy)
            cls_scores = row[5:]
            cls_idx = int(np.argmax(cls_scores))
            # Simple NMS placeholder — dedupe heavily overlapping boxes
            detections.append({
                "bbox_xyxy": (max(0, x1), max(0, y1), x2, y2),
                "confidence": obj_conf,
                "class_name": f"class_{cls_idx}",
            })
        return detections

    # ------------------------------------------------------------------
    # Internal: classifier dispatch
    # ------------------------------------------------------------------
    def _run_classifier(
        self, crop: Image.Image
    ) -> tuple[dict[str, str], dict[str, float]]:
        if self._backend == "torch":
            return self._classify_torch(crop)
        return self._classify_onnx(crop)

    def _classify_torch(
        self, crop: Image.Image
    ) -> tuple[dict[str, str], dict[str, float]]:  # pragma: no cover
        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize(int(self._imgsz_cls * 1.14)),
            transforms.CenterCrop(self._imgsz_cls),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
        ])
        tensor = tf(crop).unsqueeze(0)
        dev_str = self._device_str
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        tensor = tensor.to(torch.device(dev_str))

        with torch.no_grad():
            logits = self._classifier(tensor)

        attributes: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for head_name, head_logits in logits.items():
            probs = torch.softmax(head_logits[0], dim=0)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            vocab = self._head_vocabs.get(head_name, [])
            pred_label = vocab[pred_idx] if pred_idx < len(vocab) else f"idx_{pred_idx}"
            attributes[head_name] = pred_label
            confidences[head_name] = confidence

        return attributes, confidences

    def _classify_onnx(
        self, crop: Image.Image
    ) -> tuple[dict[str, str], dict[str, float]]:
        arr = _preprocess_crop_numpy(crop, self._imgsz_cls)
        input_name = self._classifier.get_inputs()[0].name
        outputs = self._classifier.run(None, {input_name: arr})

        output_names = [o.name for o in self._classifier.get_outputs()]
        attributes: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for i, head_name in enumerate(output_names):
            logits = outputs[i][0]  # (num_classes,)
            # softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            vocab = self._head_vocabs.get(head_name, [])
            pred_label = vocab[pred_idx] if pred_idx < len(vocab) else f"idx_{pred_idx}"
            attributes[head_name] = pred_label
            confidences[head_name] = confidence

        return attributes, confidences
