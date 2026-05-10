"""Connector instance catalog and crop workflow.

Reads source images and emits one row per visible connector instance to
``datasets/rfconnectors/instances.jsonl``. Originals are never modified or
moved. This module is data-prep tooling: it does not train models and does
not require a GPU.

Two operating modes:

- ``--mode whole-image``: produce one weak-labeled instance per source image,
  using the full image as a placeholder bbox. Folder-derived labels are
  marked ``weak_folder_label``.
- ``--mode bbox-jsonl``: ingest a pre-generated JSONL of detected/curated
  bboxes (from a detector run, Label Studio, CVAT, or a manual review pass)
  and write properly cropped instance rows.

In both modes the manifest validates against
``training/rfconnectorai/schemas/instance.py``.

Usage::

    # Bootstrap weak instance rows from existing folder structure:
    python -m rfconnectorai.data.crop_instances \\
        --input training/data/labeled/embedder \\
        --manifest datasets/rfconnectors/instances.jsonl \\
        --out datasets/rfconnectors/crops \\
        --mode whole-image \\
        --dry-run

    # Promote curated bboxes into the manifest:
    python -m rfconnectorai.data.crop_instances \\
        --input training/Images \\
        --bboxes datasets/rfconnectors/curated_bboxes.jsonl \\
        --manifest datasets/rfconnectors/instances.jsonl \\
        --out datasets/rfconnectors/crops \\
        --mode bbox-jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from rfconnectorai.schemas.instance import (
    ConnectorInstance,
    ConnectorSide,
    GeometryLabel,
    LabelConfidence,
    SourceType,
    instance_from_dict,
    validate_instance,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CropRecord:
    """In-memory record returned by :func:`build_instances`."""

    instance: ConnectorInstance
    write_crop_from: Path | None  # set when an actual crop should be saved
    crop_bbox: tuple[int, int, int, int] | None


def stable_instance_id(source: Path, bbox: tuple[int, int, int, int]) -> str:
    """Deterministic instance_id derived from source path + bbox.

    Stable across runs so a re-import of the same source/bbox produces the
    same id; useful for dedup-on-re-run.
    """
    payload = f"{source.as_posix()}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"inst_{digest}"


def parse_family_from_folder(name: str) -> tuple[str, str | None]:
    """Best-effort family + gender parse from a labeled-folder name like 'SMA-M'."""
    upper = name.upper()
    family = upper.split("-")[0] if "-" in upper else upper
    gender = None
    if upper.endswith("-M"):
        gender = "male_pin"
    elif upper.endswith("-F"):
        gender = "female_socket"
    elif upper.endswith("-RPM"):
        gender = "rp_male_body_female_contact"
    elif upper.endswith("-RPF"):
        gender = "rp_female_body_male_contact"
    return family, gender


def _infer_source_type(source: Path) -> SourceType:
    parts_lower = [p.lower() for p in source.parts]
    synthetic_hints = ("synthetic", "synth", "render", "cad")
    if any(hint in part for part in parts_lower for hint in synthetic_hints):
        return SourceType.SYNTHETIC_RENDER
    video_hints = ("videos", "video_frames")
    if any(hint in part for part in parts_lower for hint in video_hints):
        return SourceType.REAL_VIDEO_FRAME
    return SourceType.REAL_PHOTO


def _read_image_size(path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def iter_images(root: Path) -> Iterator[Path]:
    if not root.exists():
        return iter(())
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_whole_image_instances(
    *,
    input_root: Path,
    out_crops_dir: Path,
    base_dir: Path | None = None,
) -> list[CropRecord]:
    """One instance per source image using the full frame as the placeholder bbox."""
    base = base_dir or input_root
    records: list[CropRecord] = []
    skipped_unreadable = 0
    scanned = 0
    for image_path in iter_images(input_root):
        scanned += 1
        size = _read_image_size(image_path)
        if size is None:
            skipped_unreadable += 1
            continue
        w, h = size
        bbox = (0, 0, w, h)
        try:
            rel = image_path.relative_to(base)
        except ValueError:
            rel = image_path
        family_folder = image_path.parent.name
        family, gender = parse_family_from_folder(family_folder)
        instance = ConnectorInstance(
            instance_id=stable_instance_id(image_path, bbox),
            source_image=str(rel).replace("\\", "/"),
            crop_path=(out_crops_dir / rel.with_suffix(".jpg").name).as_posix(),
            bbox_xyxy=bbox,
            label_confidence=LabelConfidence.WEAK_FOLDER_LABEL,
            source_type=_infer_source_type(image_path),
            family=family,
            side_a_gender=gender or "unknown",
            geometry=GeometryLabel(),
        )
        validate_instance(instance)
        records.append(CropRecord(instance=instance, write_crop_from=None, crop_bbox=None))
    print(
        f"whole-image: scanned {scanned} images under {input_root}, "
        f"skipped {skipped_unreadable} unreadable, kept {len(records)}",
        file=sys.stderr,
    )
    return records


def build_bbox_instances(
    *,
    input_root: Path,
    bbox_jsonl: Path,
    out_crops_dir: Path,
    base_dir: Path | None = None,
) -> list[CropRecord]:
    """Read curated bbox rows and emit instance rows that crop the source.

    Each input row must contain at minimum::

        {
          "source_image": "training/Images/example.webp",
          "bbox_xyxy": [120, 80, 420, 360],
          "label_confidence": "human_verified",
          "family": "SMA",
          ...
        }

    Anything else from the instance schema (mount_style, side_b, geometry,
    etc.) is forwarded into the row.
    """
    base = base_dir or input_root
    records: list[CropRecord] = []
    with open(bbox_jsonl, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{bbox_jsonl}:{line_no} invalid JSON: {exc}"
                ) from exc

            source = Path(row["source_image"])
            full_source = (
                source if source.is_absolute() else (base / source).resolve()
            )
            bbox = tuple(int(v) for v in row["bbox_xyxy"])
            if len(bbox) != 4:
                raise ValueError(f"{bbox_jsonl}:{line_no} bbox_xyxy must have 4 ints")

            try:
                rel = full_source.relative_to(base)
            except ValueError:
                rel = full_source

            payload = dict(row)
            payload.setdefault("instance_id", stable_instance_id(full_source, bbox))  # type: ignore[arg-type]
            payload.setdefault("crop_path", (out_crops_dir / f"{payload['instance_id']}.jpg").as_posix())
            payload.setdefault("source_type", _infer_source_type(full_source).value)
            payload["bbox_xyxy"] = list(bbox)
            payload["source_image"] = str(rel).replace("\\", "/")

            instance = instance_from_dict(payload)
            validate_instance(instance)
            records.append(
                CropRecord(
                    instance=instance,
                    write_crop_from=full_source if full_source.exists() else None,
                    crop_bbox=bbox,  # type: ignore[arg-type]
                )
            )
    return records


def write_manifest(records: Iterable[CropRecord], manifest_path: Path) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.instance.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def write_crops(records: Iterable[CropRecord], dry_run: bool = False) -> int:
    if dry_run or Image is None:
        return 0
    count = 0
    for record in records:
        if record.write_crop_from is None or record.crop_bbox is None:
            continue
        out_path = Path(record.instance.crop_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(record.write_crop_from) as im:
                im.crop(record.crop_bbox).save(out_path)
            count += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"warning: failed to crop {record.write_crop_from} -> {out_path}: {exc}",
                file=sys.stderr,
            )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build connector instance manifest")
    parser.add_argument("--input", type=Path, required=True, help="Input image root.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output instances.jsonl path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for crop files.",
    )
    parser.add_argument(
        "--mode",
        choices=["whole-image", "bbox-jsonl"],
        default="whole-image",
    )
    parser.add_argument(
        "--bboxes",
        type=Path,
        default=None,
        help="Curated bbox JSONL input (required when --mode bbox-jsonl).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory used to resolve relative paths in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records and validate, but do not write crop files.",
    )
    args = parser.parse_args(argv)

    if args.mode == "bbox-jsonl" and args.bboxes is None:
        parser.error("--bboxes is required when --mode bbox-jsonl")

    if args.mode == "whole-image":
        records = build_whole_image_instances(
            input_root=args.input,
            out_crops_dir=args.out,
            base_dir=args.base_dir,
        )
    else:
        records = build_bbox_instances(
            input_root=args.input,
            bbox_jsonl=args.bboxes,
            out_crops_dir=args.out,
            base_dir=args.base_dir,
        )

    n = write_manifest(records, args.manifest)
    cropped = write_crops(records, dry_run=args.dry_run)
    print(f"manifest: {args.manifest} ({n} rows)")
    if args.dry_run:
        print("crops: dry-run, nothing written")
    else:
        print(f"crops: {args.out} ({cropped} files written)")

    if n == 0:
        print(
            f"\nERROR: 0 instances were emitted from input {args.input}.",
            file=sys.stderr,
        )
        print(f"  --input resolved to: {args.input.resolve()}", file=sys.stderr)
        print(f"  exists: {args.input.exists()}", file=sys.stderr)
        if args.input.exists() and args.input.is_dir():
            all_files = list(args.input.rglob("*"))
            n_files = sum(1 for p in all_files if p.is_file())
            n_images = sum(
                1 for p in all_files
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            print(
                f"  total files: {n_files}, image-extension files: {n_images}",
                file=sys.stderr,
            )
            top_level = sorted(
                p.name for p in args.input.iterdir() if p.is_dir()
            )[:10]
            print(f"  top-level subdirs (first 10): {top_level}", file=sys.stderr)
        print(
            "  Common causes: wrong cwd, --input path typo, dataset not "
            "checked out, or images with unsupported extensions.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
