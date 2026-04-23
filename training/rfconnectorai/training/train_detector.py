from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: Path,
    output_dir: Path,
    image_size: int,
    epochs: int,
    base_weights: str,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)

    # Ultralytics writes to runs/detect/<name>/weights/best.pt by default.
    # We point it at a subdir under output_dir and then copy the best weights out.
    project = output_dir / "ultralytics"
    name = "connector"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        project=str(project),
        name=name,
        exist_ok=True,
        device=device,
        verbose=False,
    )

    best = project / name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training completed but best weights missing at {best}")

    final = output_dir / "detector.pt"
    shutil.copy2(best, final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-weights", type=str, default="yolo11n.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.image_size = 128
        args.epochs = 1

    weights = train(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        base_weights=args.base_weights,
        device=args.device,
    )
    print(f"weights written to {weights}")


if __name__ == "__main__":
    main()
