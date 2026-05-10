from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfconnectorai.data.build_yolo_dataset import (
    DEFAULT_SPLITS,
    build_dataset,
    family_index,
    main,
    normalize_bbox,
    plan_splits,
    read_manifest,
    specimen_group_for,
)
from rfconnectorai.schemas.instance import (
    ConnectorInstance,
    LabelConfidence,
    SourceType,
)


def _save_image(path: Path, seed: int, size: tuple[int, int] = (200, 100)) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _make_instance(
    instance_id: str,
    source: str,
    family: str = "SMA",
    bbox: tuple[int, int, int, int] = (10, 10, 60, 60),
) -> ConnectorInstance:
    return ConnectorInstance(
        instance_id=instance_id,
        source_image=source,
        crop_path=f"crops/{instance_id}.jpg",
        bbox_xyxy=bbox,
        label_confidence=LabelConfidence.HUMAN_VERIFIED,
        source_type=SourceType.REAL_PHOTO,
        family=family,
    )


def _write_manifest(path: Path, instances: list[ConnectorInstance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict(), sort_keys=True) + "\n")


def test_normalize_bbox_centered():
    cx, cy, bw, bh = normalize_bbox((0, 0, 100, 50), 100, 50)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert bw == pytest.approx(1.0)
    assert bh == pytest.approx(1.0)


def test_specimen_group_uses_stem_prefix():
    inst = _make_instance("inst_a", "Images/SMA-M/agg_0001.jpg")
    assert specimen_group_for(inst) == "agg"


def test_plan_splits_is_specimen_aware():
    instances = [
        _make_instance(f"i_{i}", f"Images/SMA-M/specA_{i}.jpg") for i in range(5)
    ] + [
        _make_instance(f"j_{i}", f"Images/SMA-F/specB_{i}.jpg") for i in range(5)
    ] + [
        _make_instance(f"k_{i}", f"Images/SMA-RPM/specC_{i}.jpg") for i in range(5)
    ]
    plan = plan_splits(instances, splits=DEFAULT_SPLITS, seed=1337)
    all_groups = set(plan.train) | set(plan.val) | set(plan.test)
    assert all_groups <= {"specA", "specB", "specC"}
    # No group should appear in more than one split.
    seen: set[str] = set()
    for split in (plan.train, plan.val, plan.test):
        for g in split:
            assert g not in seen
            seen.add(g)


def test_plan_splits_forces_holdout_groups_to_test():
    instances = [
        _make_instance(f"i_{i}", f"Images/SMA-M/specA_{i}.jpg") for i in range(3)
    ]
    plan = plan_splits(instances, holdout_groups=["specA"], seed=1)
    assert "specA" in plan.test
    assert "specA" not in plan.train


def test_family_index_is_sorted():
    instances = [
        _make_instance("a", "x/a.jpg", family="SMA"),
        _make_instance("b", "x/b.jpg", family="BNC"),
        _make_instance("c", "x/c.jpg", family="TNC"),
    ]
    assert family_index(instances) == {"BNC": 0, "SMA": 1, "TNC": 2}


def test_build_dataset_dry_run_does_not_touch_disk(tmp_path: Path):
    instances = [_make_instance(f"i_{i}", f"Images/SMA-M/specA_{i}.jpg") for i in range(3)]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, instances)

    summary = build_dataset(
        manifest=manifest,
        out_dir=tmp_path / "out",
        base_dir=tmp_path,
        dry_run=True,
        seed=1337,
    )
    assert summary["dry_run"] is True
    # Dry-run still creates the directory tree but should not write data.yaml.
    assert not (tmp_path / "out" / "data.yaml").exists()
    assert not (tmp_path / "out" / "dataset.lock.json").exists()


def test_build_dataset_writes_full_tree(tmp_path: Path):
    base = tmp_path / "src"
    images_dir = base / "Images" / "SMA-M"
    instances: list[ConnectorInstance] = []
    for i in range(4):
        rel = f"Images/SMA-M/specA_{i}.jpg"
        _save_image(base / rel, seed=i, size=(120, 80))
        instances.append(_make_instance(f"i_{i}", rel))
    for i in range(4):
        rel = f"Images/SMA-F/specB_{i}.jpg"
        _save_image(base / rel, seed=i + 10, size=(120, 80))
        instances.append(_make_instance(f"j_{i}", rel, family="SMA"))

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, instances)

    out = tmp_path / "out"
    summary = build_dataset(
        manifest=manifest,
        out_dir=out,
        base_dir=base,
        dry_run=False,
        seed=1337,
    )
    assert (out / "data.yaml").exists()
    assert (out / "attributes.csv").exists()
    assert (out / "dataset.lock.json").exists()
    assert summary["train_count"] + summary["val_count"] + summary["test_count"] == 8
    assert summary["family_to_idx"] == {"SMA": 0}

    # Each split that has rows should have matching label files.
    for split in ("train", "val", "test"):
        labels = list((out / "labels" / split).glob("*.txt"))
        images = list((out / "images" / split).glob("*.jpg"))
        assert len(labels) == len(images)


def test_main_cli_dry_run(tmp_path: Path):
    instances = [_make_instance(f"i_{i}", f"Images/SMA-M/specA_{i}.jpg") for i in range(3)]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, instances)
    rc = main([
        "--input", str(manifest),
        "--out", str(tmp_path / "out"),
        "--base-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0


def test_read_manifest_rejects_invalid_json(tmp_path: Path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(manifest)
