"""Build a YOLO-formatted dataset from the connector instance manifest.

Reads ``datasets/rfconnectors/instances.jsonl`` (validated by
``rfconnectorai.schemas.instance``) and emits the standard tree:

    datasets/rfconnectors/
      images/{train,val,test}/
      labels/{train,val,test}/
      attributes.csv
      data.yaml
      dataset.lock.json

Splits are deterministic for a given ``--seed``. Splitting is
*specimen-aware* whenever a ``specimen_group`` field is set on the
instance row (or can be inferred from the source image stem) so that the
same physical connector never appears in multiple splits.

This module is data-prep only. It does not train models and does not
require a GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rfconnectorai.schemas.instance import (
    ConnectorInstance,
    instance_from_dict,
    validate_instance,
)


DEFAULT_SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


@dataclass(frozen=True)
class SplitPlan:
    """Plan describing which group ids go to which split."""

    train: list[str]
    val: list[str]
    test: list[str]

    def split_for_group(self, group: str) -> str | None:
        if group in self.train:
            return "train"
        if group in self.val:
            return "val"
        if group in self.test:
            return "test"
        return None


def read_manifest(path: Path) -> list[ConnectorInstance]:
    instances: list[ConnectorInstance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} invalid JSON: {exc}"
                ) from exc
            instance = instance_from_dict(payload)
            validate_instance(instance)
            instances.append(instance)
    return instances


def specimen_group_for(instance: ConnectorInstance) -> str:
    """Infer a specimen-group id from the source image so duplicates of the
    same physical connector never split across train/val/test.
    """
    extra = getattr(instance, "specimen_group", None)
    if extra:
        return str(extra)
    stem = Path(instance.source_image).stem.split("_")[0]
    return stem or instance.source_image


def plan_splits(
    instances: Iterable[ConnectorInstance],
    *,
    splits: tuple[tuple[str, float], ...] = DEFAULT_SPLITS,
    seed: int = 1337,
    holdout_groups: Iterable[str] = (),
) -> SplitPlan:
    """Group-aware deterministic split plan.

    All instances that share a specimen group go to the same split.
    Groups in ``holdout_groups`` are forced into ``test``.
    """
    if not splits:
        raise ValueError("splits must be non-empty")
    fractions = sum(fraction for _, fraction in splits)
    if not (0.999 <= fractions <= 1.001):
        raise ValueError(f"split fractions must sum to 1.0; got {fractions}")

    forced = {str(g) for g in holdout_groups}
    groups = sorted({specimen_group_for(inst) for inst in instances})
    free_groups = [g for g in groups if g not in forced]

    rng = random.Random(seed)
    rng.shuffle(free_groups)

    n = len(free_groups)
    split_groups: dict[str, list[str]] = {name: [] for name, _ in splits}
    cursor = 0
    for name, frac in splits:
        take = int(round(n * frac))
        split_groups[name].extend(free_groups[cursor : cursor + take])
        cursor += take
    # Assign any rounding remainder to the first split.
    if cursor < n:
        split_groups[splits[0][0]].extend(free_groups[cursor:])

    if forced:
        split_groups.setdefault("test", []).extend(sorted(forced))

    return SplitPlan(
        train=split_groups.get("train", []),
        val=split_groups.get("val", []),
        test=split_groups.get("test", []),
    )


def class_index_for(instance: ConnectorInstance, family_to_idx: dict[str, int]) -> int:
    if instance.family not in family_to_idx:
        raise ValueError(f"family {instance.family!r} not in family_to_idx")
    return family_to_idx[instance.family]


def normalize_bbox(
    bbox: tuple[int, int, int, int], image_w: int, image_h: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    bw = (x2 - x1) / image_w
    bh = (y2 - y1) / image_h
    return cx, cy, bw, bh


def get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file_at(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def family_index(instances: Iterable[ConnectorInstance]) -> dict[str, int]:
    families = sorted({inst.family for inst in instances})
    return {family: idx for idx, family in enumerate(families)}


def build_dataset(
    *,
    manifest: Path,
    out_dir: Path,
    base_dir: Path,
    splits: tuple[tuple[str, float], ...] = DEFAULT_SPLITS,
    seed: int = 1337,
    dry_run: bool = False,
    taxonomy_path: Path | None = None,
    holdout_groups: Iterable[str] = (),
    dataset_id: str | None = None,
) -> dict:
    """Materialize the standard dataset tree from the instance manifest.

    Returns a summary dict (also persisted as ``dataset.lock.json``).
    """
    instances = read_manifest(manifest)
    if not instances:
        raise ValueError(f"no instances loaded from {manifest}")

    family_to_idx = family_index(instances)
    plan = plan_splits(
        instances, splits=splits, seed=seed, holdout_groups=holdout_groups
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"
    labels_root = out_dir / "labels"
    for split_name, _ in splits:
        (images_root / split_name).mkdir(parents=True, exist_ok=True)
        (labels_root / split_name).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {name: 0 for name, _ in splits}
    skipped = 0

    attributes_csv_path = out_dir / "attributes.csv"
    attributes_rows: list[dict[str, object]] = []

    for instance in instances:
        group = specimen_group_for(instance)
        split = plan.split_for_group(group)
        if split is None:
            skipped += 1
            continue

        source = (base_dir / instance.source_image).resolve()
        size = get_image_size(source) if not dry_run else (0, 0)
        if size is None or size == (0, 0):
            # In dry-run we don't require source images to exist; in real
            # mode we must read dimensions.
            if not dry_run:
                skipped += 1
                continue
            image_w, image_h = 1, 1
        else:
            image_w, image_h = size

        cx, cy, bw, bh = normalize_bbox(instance.bbox_xyxy, image_w, image_h)
        cls_idx = class_index_for(instance, family_to_idx)

        label_line = f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        label_path = labels_root / split / f"{instance.instance_id}.txt"

        attributes_rows.append(
            {
                "instance_id": instance.instance_id,
                "split": split,
                "source_image": instance.source_image,
                "family": instance.family,
                "precision_family": instance.precision_family,
                "side_a_gender": instance.side_a_gender,
                "side_b_gender": instance.side_b_gender,
                "polarity": instance.polarity,
                "mount_style": instance.mount_style,
                "orientation": instance.orientation,
                "termination": instance.termination,
                "finish_material_cue": instance.finish_material_cue,
                "label_confidence": instance.label_confidence.value,
                "source_type": instance.source_type.value,
            }
        )

        if dry_run:
            counts[split] += 1
            continue

        image_dst = images_root / split / source.name
        if not image_dst.exists():
            shutil.copy2(source, image_dst)
        label_path.write_text(label_line, encoding="utf-8")
        counts[split] += 1

    if not dry_run:
        with open(attributes_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(attributes_rows[0].keys()))
            writer.writeheader()
            for row in attributes_rows:
                writer.writerow(row)

        data_yaml = out_dir / "data.yaml"
        names = [family for family, _ in sorted(family_to_idx.items(), key=lambda x: x[1])]
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write("path: .\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n")
            f.write("nc: " + str(len(names)) + "\n")
            f.write("names:\n")
            for n in names:
                f.write(f"  - {n}\n")

    instances_sha = hash_file_at(manifest) or ""
    taxonomy_sha = (
        hash_file_at(taxonomy_path) if taxonomy_path else hash_text("no_taxonomy")
    )
    summary = {
        "dataset_id": dataset_id
        or f"rfconnectors_{datetime.now(timezone.utc).strftime('%Y_%m_%d_001')}",
        "taxonomy_sha256": taxonomy_sha,
        "instances_sha256": instances_sha,
        "split_seed": seed,
        "train_count": counts.get("train", 0),
        "val_count": counts.get("val", 0),
        "test_count": counts.get("test", 0),
        "holdout_excluded": True,
        "skipped_count": skipped,
        "family_to_idx": family_to_idx,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
    }

    if not dry_run:
        lock_path = out_dir / "dataset.lock.json"
        lock_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from instance manifest")
    parser.add_argument("--input", type=Path, required=True, help="instances.jsonl path.")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base dir used to resolve `source_image` relative paths.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Optional path to connectors.yaml for taxonomy hashing.",
    )
    parser.add_argument(
        "--holdout-group",
        action="append",
        default=[],
        help="Force this specimen group into the test split. Can be repeated.",
    )
    args = parser.parse_args(argv)

    summary = build_dataset(
        manifest=args.input,
        out_dir=args.out,
        base_dir=args.base_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        taxonomy_path=args.taxonomy,
        holdout_groups=args.holdout_group,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
