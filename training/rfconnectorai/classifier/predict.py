"""
Inference for the trained connector classifier.

Loads weights + labels saved by `rfconnectorai.classifier.train` and predicts
class on a single image. Returns a `ClassifierPrediction` with the top class,
its confidence, and the full per-class softmax for ensemble use cases (e.g.
averaging across video frames or cross-checking against the measurement
pipeline).

Public surface:

    classifier = ConnectorClassifier.load(model_dir)
    pred = classifier.predict(image_rgb)
    # pred.class_name, pred.confidence, pred.probabilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from rfconnectorai.classifier.dataset import INPUT_SIZE, make_eval_transforms


@dataclass
class ClassifierPrediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


def _tta_variants(pil: Image.Image) -> list[Image.Image]:
    """5 cheap augmentations for test-time averaging. Avoids vertical
    flip (breaks pin-vs-hole appearance) and any color jitter (could
    swap the M-vs-F luminance cue), since both would degrade the average
    rather than denoise it."""
    w, h = pil.size
    short = min(w, h)
    # 90% center crop for the "slight zoom" variant.
    crop = int(short * 0.9)
    left = (w - crop) // 2
    top = (h - crop) // 2
    zoomed = pil.crop((left, top, left + crop, top + crop))
    return [
        pil,
        pil.transpose(Image.FLIP_LEFT_RIGHT),
        pil.rotate(10, resample=Image.BILINEAR),
        pil.rotate(-10, resample=Image.BILINEAR),
        zoomed,
    ]


def _build_model(num_classes: int) -> nn.Module:
    # Mirror what train.py does — initialize weights=None since we're loading
    # the fine-tuned state_dict immediately after.
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class ConnectorClassifier:
    """Loaded, ready-to-predict classifier. Stateless after construction."""

    def __init__(self, model: nn.Module, class_names: list[str], device: torch.device):
        self.model = model.to(device).eval()
        self.class_names = list(class_names)
        self.device = device
        self.transform = make_eval_transforms()

    @classmethod
    def load(cls, model_dir: Path, device: torch.device | None = None) -> "ConnectorClassifier":
        model_dir = Path(model_dir)
        labels_path = model_dir / "labels.json"
        weights_path = model_dir / "weights.pt"
        if not labels_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"missing labels.json or weights.pt in {model_dir}. "
                "Train a classifier first via rfconnectorai.classifier.train."
            )

        labels_blob = json.loads(labels_path.read_text())
        class_names = labels_blob["class_names"]
        if labels_blob.get("input_size", INPUT_SIZE) != INPUT_SIZE:
            raise ValueError(
                f"saved input_size {labels_blob['input_size']} != module "
                f"INPUT_SIZE {INPUT_SIZE}; rebuild dataset transforms."
            )

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(num_classes=len(class_names))
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return cls(model=model, class_names=class_names, device=device)

    def predict(self, image: np.ndarray | Image.Image,
                tta: bool = True) -> ClassifierPrediction:
        """Predict class for a single image (RGB uint8 array or PIL Image).

        With tta=True (default), runs inference on 5 augmented variants of
        the input — original, horizontal flip, ±10° rotations, slight
        center zoom — and averages probabilities. Costs ~5x compute on
        the predict path but typically buys 2-5% on noisy held-out data
        with no retraining.
        """
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB image, got shape {image.shape}")
            pil = Image.fromarray(image)
        else:
            pil = image.convert("RGB")

        if tta:
            variants = _tta_variants(pil)
            xs = torch.stack([self.transform(v) for v in variants], dim=0).to(self.device)
            with torch.no_grad():
                logits = self.model(xs)
                probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
        else:
            x = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        return ClassifierPrediction(
            class_name=self.class_names[top_idx],
            confidence=float(probs[top_idx]),
            probabilities={self.class_names[i]: float(probs[i])
                           for i in range(len(self.class_names))},
        )

    @classmethod
    def load_ensemble(cls, weight_paths: list[Path], labels_path: Path,
                      device: torch.device | None = None
                      ) -> "EnsembleClassifier":
        """Load multiple weight files (same arch + class set) into an
        ensemble. All weights must have been trained with the same
        labels.json — ensembling models with mismatched class lists is
        not supported."""
        if not weight_paths:
            raise ValueError("need at least one weight path")
        labels_blob = json.loads(Path(labels_path).read_text())
        class_names = labels_blob["class_names"]
        if labels_blob.get("input_size", INPUT_SIZE) != INPUT_SIZE:
            raise ValueError(
                f"saved input_size {labels_blob['input_size']} != module "
                f"INPUT_SIZE {INPUT_SIZE}.")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models_list = []
        for wp in weight_paths:
            m = _build_model(num_classes=len(class_names))
            m.load_state_dict(torch.load(wp, map_location=device))
            m.to(device).eval()
            models_list.append(m)
        return EnsembleClassifier(models_list, class_names, device)

    def predict_many(self, images: list[np.ndarray | Image.Image]) -> list[ClassifierPrediction]:
        """Batch-predict; faster than calling predict() in a loop."""
        if not images:
            return []
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img)
            else:
                pil = img.convert("RGB")
            tensors.append(self.transform(pil))
        x = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        results = []
        for i in range(len(images)):
            top_idx = int(np.argmax(probs[i]))
            results.append(ClassifierPrediction(
                class_name=self.class_names[top_idx],
                confidence=float(probs[i, top_idx]),
                probabilities={self.class_names[k]: float(probs[i, k])
                               for k in range(len(self.class_names))},
            ))
        return results


class EnsembleClassifier:
    """Multiple ConnectorClassifier-style models, predicting via averaged
    softmax across all models (and 5x TTA on top of that). Same public
    surface as ConnectorClassifier so predict_service can swap.
    """

    def __init__(self, models_list: list[nn.Module],
                 class_names: list[str], device: torch.device):
        self.models = models_list
        self.class_names = list(class_names)
        self.device = device
        self.transform = make_eval_transforms()

    def predict(self, image: np.ndarray | Image.Image,
                tta: bool = True) -> ClassifierPrediction:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB, got {image.shape}")
            pil = Image.fromarray(image)
        else:
            pil = image.convert("RGB")
        variants = _tta_variants(pil) if tta else [pil]
        xs = torch.stack([self.transform(v) for v in variants], dim=0).to(self.device)
        # Average softmax across all (variant, model) pairs.
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for m in self.models:
                logits = m(xs)
                probs_per_variant = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs_per_variant.mean(axis=0))
        avg = np.mean(np.stack(all_probs, axis=0), axis=0)
        top_idx = int(np.argmax(avg))
        return ClassifierPrediction(
            class_name=self.class_names[top_idx],
            confidence=float(avg[top_idx]),
            probabilities={self.class_names[i]: float(avg[i])
                           for i in range(len(self.class_names))},
        )
