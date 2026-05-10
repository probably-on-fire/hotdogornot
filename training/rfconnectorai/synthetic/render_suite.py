"""Synthetic render suite scaffold.

Defines the *render plan* for the cloud render pipeline. Actual rendering
runs on Kaggle/Colab where Trimesh/Open3D/Blender are available; this
module is responsible for:

- enumerating render variations (angle/pose, lighting, focal length, blur,
  occlusion, scale marker, multi-connector scene),
- emitting per-render *labels* that validate against the instance schema,
- writing a render manifest that the cloud worker consumes.

Tests cover the planning logic and the label emission only — no images
are rendered.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from rfconnectorai.synthetic.model_catalog import (
    ParametricConnectorModel,
    builtin_models,
    iter_models,
    model_by_id,
)


@dataclass(frozen=True)
class CameraPose:
    azimuth_deg: float
    elevation_deg: float
    distance_mm: float
    focal_length_mm: float


@dataclass(frozen=True)
class RenderVariation:
    pose: CameraPose
    lighting: str
    background: str
    blur_sigma: float
    noise_sigma: float
    occlusion: bool
    scale_marker: bool
    seed: int


@dataclass(frozen=True)
class RenderTask:
    """One render-to-be: model + variation + output paths."""

    model_id: str
    variation: RenderVariation
    output_image: str  # relative to render root
    output_label: str  # relative to render root, json
    is_multi_connector: bool = False
    extra_models: tuple[str, ...] = field(default_factory=tuple)


def default_camera_poses(seed: int) -> list[CameraPose]:
    rng = random.Random(seed)
    poses: list[CameraPose] = []
    for azimuth in (0, 45, 90, 135, 180, 225, 270, 315):
        for elevation in (-15, 0, 30, 60):
            distance = 60 + rng.uniform(-15, 25)
            focal = rng.choice((24, 35, 50, 80))
            poses.append(
                CameraPose(
                    azimuth_deg=float(azimuth),
                    elevation_deg=float(elevation),
                    distance_mm=float(distance),
                    focal_length_mm=float(focal),
                )
            )
    return poses


def default_variations(*, per_model: int, seed: int) -> Iterator[RenderVariation]:
    rng = random.Random(seed)
    poses = default_camera_poses(seed)
    backgrounds = ("studio_white", "lab_bench", "outdoor_diffuse", "tabletop_dark")
    lights = ("softbox", "topdown", "ringlight", "harsh_directional")
    for i in range(per_model):
        pose = rng.choice(poses)
        lighting = rng.choice(lights)
        background = rng.choice(backgrounds)
        blur = rng.choice((0.0, 0.0, 0.5, 1.5, 3.0))
        noise = rng.choice((0.0, 5.0, 10.0))
        occlusion = rng.random() < 0.15
        marker = rng.random() < 0.20
        yield RenderVariation(
            pose=pose,
            lighting=lighting,
            background=background,
            blur_sigma=blur,
            noise_sigma=noise,
            occlusion=occlusion,
            scale_marker=marker,
            seed=rng.randrange(0, 2**31),
        )


def plan_renders(
    models: Iterable[ParametricConnectorModel],
    *,
    per_model: int,
    seed: int = 1337,
    include_multi_connector: bool = True,
) -> list[RenderTask]:
    plan: list[RenderTask] = []
    models = list(models)
    for model in models:
        for i, variation in enumerate(default_variations(per_model=per_model, seed=seed + hash(model.model_id) % 9999)):
            base = f"{model.model_id}_{i:04d}"
            plan.append(
                RenderTask(
                    model_id=model.model_id,
                    variation=variation,
                    output_image=f"images/{base}.png",
                    output_label=f"labels/{base}.json",
                )
            )
    if include_multi_connector and len(models) >= 2:
        rng = random.Random(seed + 9999)
        for i in range(min(per_model, len(models))):
            primary = rng.choice(models)
            extra = rng.choice([m for m in models if m.model_id != primary.model_id])
            variation = next(default_variations(per_model=1, seed=seed + i + 7777))
            base = f"multi_{primary.model_id}_{extra.model_id}_{i:04d}"
            plan.append(
                RenderTask(
                    model_id=primary.model_id,
                    variation=variation,
                    output_image=f"images/{base}.png",
                    output_label=f"labels/{base}.json",
                    is_multi_connector=True,
                    extra_models=(extra.model_id,),
                )
            )
    return plan


def render_task_label(
    task: RenderTask,
    *,
    image_size: tuple[int, int] = (1024, 1024),
) -> dict:
    """Emit the perfect-label JSON for one render task.

    The schema is intentionally compatible with
    ``rfconnectorai.schemas.instance.ConnectorInstance`` so synthetic data
    can be ingested by the same crop_instances/build_yolo pipeline as real
    data, with ``label_confidence: synthetic_verified``.
    """
    primary = model_by_id(task.model_id)
    image_w, image_h = image_size
    cx, cy = image_w // 2, image_h // 2
    half = min(image_w, image_h) // 4
    label = {
        "instance_id": f"syn_{task.model_id}_{task.variation.seed}",
        "source_image": task.output_image,
        "crop_path": task.output_image,
        "bbox_xyxy": [cx - half, cy - half, cx + half, cy + half],
        "label_confidence": "synthetic_verified",
        "source_type": "synthetic_render",
        "family": primary.family,
        "side_a_gender": primary.side_a.gender,
        "side_b_gender": primary.side_b.gender if primary.side_b else "not_applicable",
        "polarity": primary.side_a.polarity,
        "mount_style": primary.mount_style,
        "orientation": primary.orientation,
        "termination": primary.termination,
        "finish_material_cue": primary.finish_material_cue,
        "side_a": {
            "family": primary.side_a.family,
            "gender": primary.side_a.gender,
            "polarity": primary.side_a.polarity,
            "threaded": primary.side_a.threaded,
            "coupling": primary.side_a.coupling,
        },
        "side_b": (
            {
                "family": primary.side_b.family,
                "gender": primary.side_b.gender,
                "polarity": primary.side_b.polarity,
                "threaded": primary.side_b.threaded,
                "coupling": primary.side_b.coupling,
            }
            if primary.side_b is not None
            else None
        ),
        "geometry": {
            "thread_diameter_mm": primary.geometry.thread_diameter_mm,
            "thread_pitch_or_count": primary.geometry.thread_pitch_or_count,
            "body_length_mm": primary.geometry.body_length_mm,
            "hex_size_mm": primary.geometry.hex_size_mm,
            "aperture_mm": primary.geometry.aperture_mm,
            "requires_calibrated_reference": False,
        },
        "render": {
            "model_id": primary.model_id,
            "extra_models": list(task.extra_models),
            "is_multi_connector": task.is_multi_connector,
            "azimuth_deg": task.variation.pose.azimuth_deg,
            "elevation_deg": task.variation.pose.elevation_deg,
            "distance_mm": task.variation.pose.distance_mm,
            "focal_length_mm": task.variation.pose.focal_length_mm,
            "lighting": task.variation.lighting,
            "background": task.variation.background,
            "blur_sigma": task.variation.blur_sigma,
            "noise_sigma": task.variation.noise_sigma,
            "occlusion": task.variation.occlusion,
            "scale_marker": task.variation.scale_marker,
            "seed": task.variation.seed,
        },
    }
    return label


def write_render_manifest(plan: list[RenderTask], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in plan:
            f.write(
                json.dumps(
                    {
                        "model_id": task.model_id,
                        "output_image": task.output_image,
                        "output_label": task.output_label,
                        "is_multi_connector": task.is_multi_connector,
                        "extra_models": list(task.extra_models),
                        "variation": {
                            "azimuth_deg": task.variation.pose.azimuth_deg,
                            "elevation_deg": task.variation.pose.elevation_deg,
                            "distance_mm": task.variation.pose.distance_mm,
                            "focal_length_mm": task.variation.pose.focal_length_mm,
                            "lighting": task.variation.lighting,
                            "background": task.variation.background,
                            "blur_sigma": task.variation.blur_sigma,
                            "noise_sigma": task.variation.noise_sigma,
                            "occlusion": task.variation.occlusion,
                            "scale_marker": task.variation.scale_marker,
                            "seed": task.variation.seed,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan synthetic renders")
    parser.add_argument("--out", type=Path, required=True, help="Render plan output dir")
    parser.add_argument("--per-model", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--no-confusing-negatives",
        action="store_true",
        help="Exclude confusing-negative models from the plan.",
    )
    parser.add_argument(
        "--no-multi-connector",
        action="store_true",
        help="Skip multi-connector scenes.",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    models = list(iter_models(include_confusing_negatives=not args.no_confusing_negatives))
    plan = plan_renders(
        models,
        per_model=args.per_model,
        seed=args.seed,
        include_multi_connector=not args.no_multi_connector,
    )
    manifest = args.out / "render_manifest.jsonl"
    n = write_render_manifest(plan, manifest)
    print(json.dumps({"models": len(models), "tasks": n, "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
