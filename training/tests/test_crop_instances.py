from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfconnectorai.data.crop_instances import (
    build_bbox_instances,
    build_whole_image_instances,
    main,
    parse_family_from_folder,
    stable_instance_id,
    write_crops,
    write_manifest,
)
from rfconnectorai.schemas.instance import LabelConfidence, SourceType, instance_from_dict


def _save_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def labeled_root(tmp_path: Path) -> Path:
    root = tmp_path / "labeled" / "embedder"
    _save_image(root / "SMA-M" / "001.jpg", seed=1)
    _save_image(root / "SMA-M" / "002.jpg", seed=2)
    _save_image(root / "SMA-F" / "001.jpg", seed=3)
    _save_image(root / "RP-SMA-RPM" / "001.jpg", seed=4)
    return root


def test_parse_family_from_folder():
    assert parse_family_from_folder("SMA-M") == ("SMA", "male_pin")
    assert parse_family_from_folder("SMA-F") == ("SMA", "female_socket")
    assert parse_family_from_folder("RP-SMA-RPM") == ("RP", "rp_male_body_female_contact")
    assert parse_family_from_folder("BNC") == ("BNC", None)


def test_stable_instance_id_is_deterministic(tmp_path: Path):
    p = tmp_path / "img.jpg"
    p.write_text("placeholder")
    a = stable_instance_id(p, (1, 2, 3, 4))
    b = stable_instance_id(p, (1, 2, 3, 4))
    c = stable_instance_id(p, (1, 2, 3, 5))
    assert a == b
    assert a != c
    assert a.startswith("inst_")


def test_build_whole_image_instances(labeled_root: Path, tmp_path: Path):
    crops_dir = tmp_path / "crops"
    records = build_whole_image_instances(
        input_root=labeled_root,
        out_crops_dir=crops_dir,
        base_dir=labeled_root,
    )
    assert len(records) == 4
    families = {r.instance.family for r in records}
    assert "SMA" in families
    for record in records:
        assert record.instance.label_confidence == LabelConfidence.WEAK_FOLDER_LABEL
        assert record.instance.source_type == SourceType.REAL_PHOTO
        x1, y1, x2, y2 = record.instance.bbox_xyxy
        assert x2 > x1 and y2 > y1


def test_build_whole_image_instances_marks_synthetic(tmp_path: Path):
    root = tmp_path / "data" / "synthetic_faces"
    _save_image(root / "SMA-M" / "render_001.jpg", seed=1)
    records = build_whole_image_instances(
        input_root=root,
        out_crops_dir=tmp_path / "crops",
        base_dir=tmp_path,
    )
    assert len(records) == 1
    assert records[0].instance.source_type == SourceType.SYNTHETIC_RENDER


def test_build_bbox_instances_validates_and_promotes(tmp_path: Path):
    src = tmp_path / "images" / "panel.jpg"
    _save_image(src, seed=10, size=(128, 96))
    bboxes = tmp_path / "bboxes.jsonl"
    rows = [
        {
            "source_image": "images/panel.jpg",
            "bbox_xyxy": [10, 10, 60, 60],
            "label_confidence": "human_verified",
            "family": "SMA",
            "side_a_gender": "male_pin",
            "polarity": "standard",
            "mount_style": "cable_mount",
        },
        {
            "source_image": "images/panel.jpg",
            "bbox_xyxy": [70, 10, 120, 80],
            "label_confidence": "human_verified",
            "family": "BNC",
            "side_a_gender": "female_socket",
            "mount_style": "adapter",
            "side_a": {
                "family": "BNC",
                "gender": "female_socket",
                "polarity": "not_applicable",
                "coupling": "bayonet",
            },
            "side_b": {
                "family": "SMA",
                "gender": "male_pin",
                "polarity": "standard",
                "threaded": True,
            },
        },
    ]
    bboxes.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    records = build_bbox_instances(
        input_root=tmp_path,
        bbox_jsonl=bboxes,
        out_crops_dir=tmp_path / "crops",
        base_dir=tmp_path,
    )
    assert len(records) == 2
    cable = records[0].instance
    adapter = records[1].instance
    assert cable.family == "SMA"
    assert adapter.mount_style == "adapter"
    assert adapter.side_b is not None
    assert adapter.side_b.family == "SMA"


def test_write_manifest_round_trips(tmp_path: Path):
    src = tmp_path / "img.jpg"
    _save_image(src, seed=1, size=(40, 40))
    records = build_whole_image_instances(
        input_root=tmp_path,
        out_crops_dir=tmp_path / "crops",
        base_dir=tmp_path,
    )
    manifest = tmp_path / "instances.jsonl"
    n = write_manifest(records, manifest)
    assert n >= 1
    with open(manifest, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    parsed = instance_from_dict(first)
    assert parsed.label_confidence == LabelConfidence.WEAK_FOLDER_LABEL


def test_write_crops_skips_when_no_source(tmp_path: Path):
    records = build_whole_image_instances(
        input_root=tmp_path,
        out_crops_dir=tmp_path / "crops",
        base_dir=tmp_path,
    )
    # whole-image mode does not stage actual crops — write_crop_from is None.
    written = write_crops(records, dry_run=False)
    assert written == 0


def test_write_crops_creates_files_in_bbox_mode(tmp_path: Path):
    src = tmp_path / "img.jpg"
    _save_image(src, seed=1, size=(80, 60))
    bboxes = tmp_path / "bboxes.jsonl"
    bboxes.write_text(json.dumps({
        "source_image": "img.jpg",
        "bbox_xyxy": [5, 5, 30, 30],
        "label_confidence": "human_verified",
        "family": "SMA",
    }) + "\n", encoding="utf-8")
    crops = tmp_path / "crops"
    records = build_bbox_instances(
        input_root=tmp_path,
        bbox_jsonl=bboxes,
        out_crops_dir=crops,
        base_dir=tmp_path,
    )
    n = write_crops(records, dry_run=False)
    assert n == 1
    out_files = list(crops.rglob("*.jpg"))
    assert len(out_files) == 1
    with Image.open(out_files[0]) as im:
        w, h = im.size
        assert w == 25 and h == 25


def test_main_cli_whole_image(tmp_path: Path, labeled_root: Path):
    manifest = tmp_path / "instances.jsonl"
    crops = tmp_path / "crops"
    rc = main([
        "--input", str(labeled_root),
        "--manifest", str(manifest),
        "--out", str(crops),
        "--mode", "whole-image",
        "--base-dir", str(labeled_root),
        "--dry-run",
    ])
    assert rc == 0
    assert manifest.exists()
    assert manifest.read_text(encoding="utf-8").strip()
