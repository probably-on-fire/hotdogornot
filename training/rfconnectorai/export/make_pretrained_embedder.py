"""
Export a frozen, pretrained RGBDEmbedder to ONNX without any fine-tuning.

This is the "ship the embedder immediately" path: the on-device enroll
architecture lets the reference database carry the class signal, so the
embedder only has to provide good general visual features. A pretrained
DINOv2 (or MobileViT) backbone with no further training does that out of
the box for the V1 demo.

Usage:
    python -m rfconnectorai.export.make_pretrained_embedder \\
        --output runs/embedder_pretrained.onnx \\
        --backbone vit_small_patch14_dinov2.lvd142m \\
        --quantize

The projection head is randomly initialized (no training data informed it),
which is fine for the on-device-enroll use case: the random projection still
preserves the relative geometry the backbone provides, and the on-device
reference DB will calibrate the metric in production.

If you'd rather have a trained projection head, run the full pipeline with
`train_embedder` and use `onnx_export` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rfconnectorai.export.onnx_export import quantize_int8
from rfconnectorai.models.embedder import RGBDEmbedder, recommended_image_size


def export(
    output: Path,
    backbone: str,
    image_size: int | None,
    quantize: bool,
) -> Path:
    if image_size is None:
        image_size = recommended_image_size(backbone)

    model = RGBDEmbedder(embedding_dim=128, pretrained=True, backbone=backbone)
    model.eval()

    dummy = torch.randn(1, 4, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
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

    if quantize:
        q = output.with_name(output.stem + "_int8" + output.suffix)
        quantize_int8(output, q)
        return q
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="Path to write the ONNX file")
    ap.add_argument(
        "--backbone",
        type=str,
        default="vit_small_patch14_dinov2.lvd142m",
        help="timm model name (default: DINOv2 small — strong fine-grained features)",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution. If unset, uses recommended size for the backbone.",
    )
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce an INT8 quantized variant alongside the float model.",
    )
    args = ap.parse_args()

    out = export(
        output=args.output,
        backbone=args.backbone,
        image_size=args.image_size,
        quantize=args.quantize,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    if args.quantize and out.name.endswith("_int8.onnx"):
        # Note: when --quantize, export() returns the INT8 path; the FP32 also exists.
        fp = out.with_name(out.name.replace("_int8.onnx", ".onnx"))
        if fp.exists():
            print(f"also  {fp}  ({fp.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
