"""CLI entry point for the detect-classify pipeline.

Usage::

    # PyTorch weights (requires torch + ultralytics):
    python -m rfconnectorai.pipeline.predict_cli \
        --image test.jpg \
        --detector models/detector/best.pt \
        --classifier models/multihead_classifier/best.pt \
        --device 0

    # ONNX weights (torch-free, works on any platform):
    python -m rfconnectorai.pipeline.predict_cli \
        --image test.jpg \
        --detector-onnx models/exports/detector.onnx \
        --classifier-onnx models/exports/classifier.onnx \
        --vocabs models/exports/head_vocabs.json

    # Batch mode (directory of images):
    python -m rfconnectorai.pipeline.predict_cli \
        --image-dir test_images/ \
        --detector models/detector/best.pt \
        --classifier models/multihead_classifier/best.pt \
        --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from rfconnectorai.pipeline.detect_classify import DetectClassifyPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF connector detect + classify pipeline CLI"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Single image path")
    input_group.add_argument("--image-dir", type=Path, help="Directory of images")

    # PyTorch weights
    parser.add_argument("--detector", type=Path, help="YOLO .pt weights")
    parser.add_argument("--classifier", type=Path, help="Multi-head .pt weights")
    parser.add_argument("--device", default="cpu", help="PyTorch device")

    # ONNX weights
    parser.add_argument("--detector-onnx", type=Path, help="YOLO .onnx model")
    parser.add_argument("--classifier-onnx", type=Path, help="Classifier .onnx model")
    parser.add_argument("--vocabs", type=Path, help="head_vocabs.json for ONNX")

    # Options
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz-cls", type=int, default=384, help="Classifier input size")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--save-crops", action="store_true", help="Save detected crops")
    parser.add_argument("--crops-dir", type=Path, default=Path("crops"), help="Crops output dir")

    return parser.parse_args(argv)


def build_pipeline(args: argparse.Namespace) -> DetectClassifyPipeline:
    if args.detector_onnx and args.classifier_onnx:
        if not args.vocabs:
            print("ERROR: --vocabs required with ONNX models", file=sys.stderr)
            sys.exit(1)
        return DetectClassifyPipeline.from_onnx(
            detector_onnx=args.detector_onnx,
            classifier_onnx=args.classifier_onnx,
            head_vocabs_json=args.vocabs,
            imgsz_cls=args.imgsz_cls,
        )
    elif args.detector and args.classifier:
        return DetectClassifyPipeline.from_torch(
            detector_weights=args.detector,
            classifier_weights=args.classifier,
            device=args.device,
            imgsz_cls=args.imgsz_cls,
        )
    else:
        print(
            "ERROR: provide either (--detector + --classifier) for PyTorch "
            "or (--detector-onnx + --classifier-onnx + --vocabs) for ONNX",
            file=sys.stderr,
        )
        sys.exit(1)


def iter_images(args: argparse.Namespace):
    if args.image:
        yield args.image
    elif args.image_dir:
        for p in sorted(args.image_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = build_pipeline(args)

    out_file = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(args.out, "w", encoding="utf-8")

    if args.save_crops:
        args.crops_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_detections = 0

    for img_path in iter_images(args):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"WARNING: skipping {img_path}: {exc}", file=sys.stderr)
            continue

        result = pipeline.predict(image, conf_threshold=args.conf)
        total_images += 1
        total_detections += len(result.predictions)

        row = {
            "image": str(img_path),
            **result.to_dict(),
        }

        if out_file:
            out_file.write(json.dumps(row, sort_keys=True) + "\n")

        # Print summary to stdout
        det_ms = result.latency_ms.get("detector", 0)
        cls_ms = result.latency_ms.get("classifier", 0)
        tot_ms = result.latency_ms.get("total", 0)
        print(
            f"{img_path.name}: {len(result.predictions)} detections "
            f"({det_ms:.0f}ms det, {cls_ms:.0f}ms cls, {tot_ms:.0f}ms total)"
        )

        for i, pred in enumerate(result.predictions):
            attrs = pred.attributes
            family = attrs.get("family", "?")
            gender = attrs.get("side_a_gender", "?")
            conf_f = pred.confidences.get("family", 0)
            print(
                f"  [{i}] {pred.detection_class} "
                f"conf={pred.detection_confidence:.2f} → "
                f"family={family} ({conf_f:.1%}), gender={gender}"
            )

            if args.save_crops and pred.crop is not None:
                crop_name = f"{img_path.stem}_det{i}.jpg"
                pred.crop.save(args.crops_dir / crop_name)

    if out_file:
        out_file.close()

    print(f"\nDone: {total_images} images, {total_detections} detections")
    if args.out:
        print(f"Results: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
