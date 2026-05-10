from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfconnectorai.data.audit import (
    AuditReport,
    build_report,
    classify_path,
    detect_multi_connector_hint,
    find_duplicate_groups,
    find_leakage_groups,
    infer_class_label,
    main,
    render_markdown,
    write_report,
)


def _save_image(path: Path, seed: int, size: tuple[int, int] = (32, 32)) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _save_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


@pytest.fixture
def audit_layout(tmp_path: Path) -> Path:
    """Fake layout matching the Implementation Plan §6.1 root list."""
    images = tmp_path / "data" / "labeled" / "embedder"
    holdout = tmp_path / "data" / "test_holdout"
    reference = tmp_path / "data" / "reference"
    synthetic = tmp_path / "data" / "synthetic_faces" / "SMA-M"

    _save_image(images / "SMA-M" / "001.jpg", seed=1)
    _save_image(images / "SMA-M" / "002.jpg", seed=2)
    _save_image(images / "SMA-F" / "001.jpg", seed=3)
    _save_image(holdout / "SMA-M" / "phone_shot.jpg", seed=4)
    _save_image(reference / "vendor" / "vendor.jpg", seed=5)
    _save_image(synthetic / "render_001.jpg", seed=6)
    return tmp_path


def test_classify_path_synthetic_holdout_reference():
    syn = classify_path(Path("data/synthetic_faces/SMA-M/render_001.jpg"))
    assert syn["synthetic"] and not syn["holdout"] and not syn["reference"]

    hold = classify_path(Path("data/test_holdout/SMA-M/phone_shot.jpg"))
    assert hold["holdout"] and not hold["synthetic"]

    ref = classify_path(Path("data/reference/vendor/vendor.jpg"))
    assert ref["reference"]


def test_infer_class_label_skips_known_directory_names():
    assert infer_class_label(Path("Images/SMA-M/001.jpg")) == "SMA-M"
    assert infer_class_label(Path("data/labeled/embedder/SMA-F/001.jpg")) == "SMA-F"
    assert infer_class_label(Path("images/001.jpg")) is None


def test_detect_multi_connector_hint_uses_filename():
    assert detect_multi_connector_hint(Path("a/b/group_shot.jpg"))
    assert detect_multi_connector_hint(Path("a/b/connector_panel.jpg"))
    assert not detect_multi_connector_hint(Path("a/b/single.jpg"))


def test_build_report_counts_and_classes(audit_layout: Path):
    roots = [
        audit_layout / "data" / "labeled",
        audit_layout / "data" / "test_holdout",
        audit_layout / "data" / "reference",
        audit_layout / "data" / "synthetic_faces",
    ]
    report = build_report(
        data_dir=audit_layout,
        roots=roots,
        taxonomy_ids=["sma", "rp_sma", "bnc"],
        min_per_class=2,
    )
    assert isinstance(report, AuditReport)
    assert sum(r.image_count for r in report.roots) == 6
    labeled = next(r for r in report.roots if r.root.endswith("labeled"))
    assert labeled.by_class == {"SMA-F": 1, "SMA-M": 2}
    holdout = next(r for r in report.roots if r.root.endswith("test_holdout"))
    assert holdout.holdout_count == 1
    synthetic = next(r for r in report.roots if r.root.endswith("synthetic_faces"))
    assert synthetic.synthetic_count == 1


def test_build_report_detects_duplicate_and_leakage(audit_layout: Path, tmp_path: Path):
    src = audit_layout / "data" / "labeled" / "embedder" / "SMA-M" / "001.jpg"
    dup_in_train = audit_layout / "data" / "labeled" / "embedder" / "SMA-M" / "001_dup.jpg"
    dup_in_holdout = audit_layout / "data" / "test_holdout" / "SMA-M" / "001_dup.jpg"
    _save_copy(src, dup_in_train)
    _save_copy(src, dup_in_holdout)

    roots = [
        audit_layout / "data" / "labeled",
        audit_layout / "data" / "test_holdout",
    ]
    report = build_report(data_dir=audit_layout, roots=roots, min_per_class=1)
    assert report.duplicate_groups, "expected at least one duplicate group"
    assert report.leakage_groups, "expected at least one holdout/train leakage entry"
    leak = report.leakage_groups[0]
    assert leak["kind"] == "holdout_train_overlap"
    assert any("test_holdout" in p for p in leak["paths"])


def test_build_report_flags_missing_taxonomy_classes(audit_layout: Path):
    roots = [audit_layout / "data" / "labeled"]
    report = build_report(
        data_dir=audit_layout,
        roots=roots,
        taxonomy_ids=["sma-m", "sma-f", "bnc", "tnc"],
        min_per_class=1,
    )
    assert "bnc" in report.missing_taxonomy_classes
    assert "tnc" in report.missing_taxonomy_classes


def test_render_markdown_contains_section_headers(audit_layout: Path):
    roots = [audit_layout / "data" / "labeled"]
    report = build_report(data_dir=audit_layout, roots=roots, min_per_class=10)
    markdown = render_markdown(report)
    assert "# Dataset Audit" in markdown
    assert "## Summary By Root" in markdown
    assert "## Class Distribution" in markdown


def test_write_report_emits_markdown_and_json(audit_layout: Path, tmp_path: Path):
    roots = [audit_layout / "data" / "labeled"]
    report = build_report(data_dir=audit_layout, roots=roots, min_per_class=1)
    out = tmp_path / "out" / "AUDIT.md"
    write_report(report, out)
    assert out.exists()
    json_path = out.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "roots" in payload
    assert "duplicate_groups" in payload


def test_main_cli_runs_against_fixture(audit_layout: Path, tmp_path: Path):
    out = tmp_path / "out" / "AUDIT.md"
    rc = main(
        [
            "--data-dir", str(audit_layout),
            "--root", str(audit_layout / "data" / "labeled"),
            "--root", str(audit_layout / "data" / "test_holdout"),
            "--out", str(out),
            "--min-per-class", "1",
            "--skip-taxonomy",
        ]
    )
    assert rc == 0
    assert out.exists()


def test_unreadable_file_is_flagged(tmp_path: Path):
    bad = tmp_path / "labeled" / "SMA-M" / "broken.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"this is not an image")
    report = build_report(data_dir=tmp_path, roots=[tmp_path / "labeled"], min_per_class=1)
    root = report.roots[0]
    assert root.unreadable_count == 1
