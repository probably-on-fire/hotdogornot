"""
Export the trained ResNet-18 classifier to ONNX so it can be loaded by
Unity Sentis on the AR app.

Usage (CLI):
    python -m rfconnectorai.classifier.export_onnx \\
        --model-dir models/connector_classifier \\
        --output models/connector_classifier/weights.onnx

Usage (programmatic):
    from rfconnectorai.classifier.export_onnx import export_to_onnx
    export_to_onnx(model_dir, output_path)

Auto-called from train.py at the end of each successful training run, so
every retrain produces a fresh weights.onnx alongside weights.pt. The
relay's GET /model/weights endpoint serves whichever extension the
manifest points at.

Sentis 2.x can load this ONNX directly via ModelLoader.Load(...) — no
.sentis-specific conversion needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from rfconnectorai.classifier.dataset import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class _NormalizedClassifier(nn.Module):
    """
    Wraps the trained ResNet-18 with input normalization baked in.

    The Unity side passes a raw [0, 1] float NCHW tensor directly. Doing
    normalization in Python at training time and ALSO at inference time
    on Unity would require duplicating the constants (and getting them
    right). Baking it into the ONNX graph means the C# code just feeds
    raw pixel-normalized values and gets logits out, with no per-platform
    normalization to keep in sync.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # Register as buffers so they ride along in state_dict + ONNX export.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) in [0, 1]
        return self.base((x - self.mean) / self.std)


def export_to_onnx(
    model_dir: Path,
    output_path: Path | None = None,
    opset: int = 17,
) -> Path:
    """
    Load weights from `model_dir/weights.pt`, wrap with normalization,
    and emit ONNX to `output_path`. Returns the written path.
    """
    model_dir = Path(model_dir)
    if output_path is None:
        output_path = model_dir / "weights.onnx"

    weights_path = model_dir / "weights.pt"
    labels_path = model_dir / "labels.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing {weights_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing {labels_path}")

    labels_blob = json.loads(labels_path.read_text())
    class_names = labels_blob["class_names"]
    input_size = labels_blob.get("input_size", INPUT_SIZE)

    base = _build_model(num_classes=len(class_names))
    base.load_state_dict(torch.load(weights_path, map_location="cpu"))
    wrapped = _NormalizedClassifier(base).eval()

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        # Static shape is fine — Unity always feeds INPUT_SIZE×INPUT_SIZE crops.
        opset_version=opset,
        dynamo=False,   # legacy TorchScript exporter; matches existing onnx_export pattern
    )
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = export_to_onnx(args.model_dir, args.output, args.opset)
    size_kb = out.stat().st_size // 1024
    print(f"wrote {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
