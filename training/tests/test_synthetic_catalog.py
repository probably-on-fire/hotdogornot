from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfconnectorai.synthetic.model_catalog import (
    ParametricConnectorModel,
    builtin_models,
    iter_models,
    model_by_id,
)
from rfconnectorai.synthetic.render_suite import (
    main,
    plan_renders,
    render_task_label,
    write_render_manifest,
)


def test_builtin_catalog_has_required_families():
    families = {m.family for m in builtin_models()}
    for required in ("SMA", "RP-SMA", "BNC", "3.5mm", "2.92mm", "2.4mm", "1.85mm", "1.0mm"):
        assert required in families, f"missing family {required}"


def test_builtin_catalog_includes_adapters_and_negatives():
    descriptions = " ".join(m.description for m in builtin_models())
    assert "adapter" in descriptions.lower()
    assert any(m.is_confusing_negative for m in builtin_models())


def test_iter_models_can_filter_families():
    sma_only = list(iter_models(families={"SMA"}))
    assert sma_only
    assert all(m.family == "SMA" for m in sma_only)


def test_iter_models_can_skip_confusing_negatives():
    out = list(iter_models(include_confusing_negatives=False))
    assert all(not m.is_confusing_negative for m in out)


def test_model_by_id_round_trip():
    model = model_by_id("sma_male_straight")
    assert model.family == "SMA"
    assert model.side_a.gender == "male_pin"


def test_model_by_id_unknown_raises():
    with pytest.raises(KeyError):
        model_by_id("does_not_exist")


def test_plan_renders_creates_per_model_tasks():
    models = list(iter_models(families={"SMA"}, include_confusing_negatives=False))
    plan = plan_renders(models, per_model=3, seed=7, include_multi_connector=False)
    assert len(plan) == 3 * len(models)
    for task in plan:
        assert task.output_image.startswith("images/")
        assert task.output_label.startswith("labels/")


def test_plan_renders_emits_multi_connector_when_enabled():
    models = list(iter_models(families={"SMA", "BNC"}, include_confusing_negatives=False))
    plan = plan_renders(models, per_model=2, seed=7, include_multi_connector=True)
    assert any(t.is_multi_connector for t in plan)


def test_render_task_label_compatible_with_instance_schema():
    models = list(iter_models(families={"SMA"}, include_confusing_negatives=False))
    plan = plan_renders(models, per_model=1, seed=1, include_multi_connector=False)
    task = plan[0]
    label = render_task_label(task)
    assert label["label_confidence"] == "synthetic_verified"
    assert label["source_type"] == "synthetic_render"
    assert label["family"] == "SMA"
    assert "side_a" in label
    assert label["render"]["model_id"] == task.model_id


def test_write_render_manifest_jsonl_round_trip(tmp_path: Path):
    models = list(iter_models(families={"SMA"}, include_confusing_negatives=False))
    plan = plan_renders(models, per_model=2, seed=2, include_multi_connector=False)
    manifest = tmp_path / "manifest.jsonl"
    n = write_render_manifest(plan, manifest)
    assert n == len(plan)
    with open(manifest, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines[0]["model_id"]
    assert "variation" in lines[0]


def test_main_cli(tmp_path: Path):
    rc = main([
        "--out", str(tmp_path),
        "--per-model", "1",
        "--seed", "7",
        "--no-multi-connector",
        "--no-confusing-negatives",
    ])
    assert rc == 0
    assert (tmp_path / "render_manifest.jsonl").exists()
