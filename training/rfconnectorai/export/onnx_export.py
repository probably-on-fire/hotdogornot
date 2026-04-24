from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from rfconnectorai.models.embedder import RGBDEmbedder


def export_embedder(checkpoint: Path, output: Path, image_size: int = 384) -> Path:
    """
    Export the RGBDEmbedder to ONNX.

    Unity Sentis as of 2.x supports ONNX opset 15+. We target opset 17 which is
    broadly compatible and supported by onnxruntime 1.18+.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dim = ckpt["embedding_dim"]
    backbone = ckpt.get("backbone", "mobilevitv2_100")
    model = RGBDEmbedder(embedding_dim=dim, pretrained=False, backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 4, image_size, image_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces the legacy TorchScript-based exporter, which does not
    # require the optional `onnxscript` dependency (the new torch.export path does).
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    return output


def export_detector(weights: Path, output: Path, image_size: int = 640) -> Path:
    """
    Export the YOLO detector to ONNX using Ultralytics' built-in exporter.

    Ultralytics writes an ONNX alongside the .pt file. We move it to `output`
    for consistent artifact naming.
    """
    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=image_size, opset=17)
    produced_path = Path(produced)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced_path), str(output))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder-checkpoint", type=Path)
    ap.add_argument("--embedder-out", type=Path)
    ap.add_argument("--embedder-size", type=int, default=384)
    ap.add_argument("--detector-weights", type=Path)
    ap.add_argument("--detector-out", type=Path)
    ap.add_argument("--detector-size", type=int, default=640)
    args = ap.parse_args()

    if args.embedder_checkpoint and args.embedder_out:
        p = export_embedder(
            checkpoint=args.embedder_checkpoint,
            output=args.embedder_out,
            image_size=args.embedder_size,
        )
        print(f"embedder ONNX: {p}")

    if args.detector_weights and args.detector_out:
        p = export_detector(
            weights=args.detector_weights,
            output=args.detector_out,
            image_size=args.detector_size,
        )
        print(f"detector ONNX: {p}")


if __name__ == "__main__":
    main()
