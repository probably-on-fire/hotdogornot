"""
Orchestrator for the synthetic-data pipeline. Given a class config and a
budget, produces N renders per class with randomized scene parameters.
Output layout matches what the existing RGBDConnectorDataset expects
(per-class directories) so the trainer needs no modification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from rfconnectorai.synthetic.render import render_single
from rfconnectorai.synthetic.scene import RenderConfig, DomainRandomizationConfig


@dataclass
class ClassSpec:
    name: str
    family: str
    gender: str
    mesh_path: Path


def load_class_specs(config_path: Path, cad_root: Path) -> list[ClassSpec]:
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    out = []
    for entry in doc["classes"]:
        mesh_path = cad_root / entry["mesh_path"]
        out.append(ClassSpec(
            name=entry["name"],
            family=entry["family"],
            gender=entry["gender"],
            mesh_path=mesh_path,
        ))
    return out


def render_class(
    spec: ClassSpec,
    n_samples: int,
    output_root: Path,
    render_config: RenderConfig,
    dr_config: DomainRandomizationConfig | None,
    seed_offset: int = 0,
) -> int:
    """Render n_samples images for one class into output_root/<class_name>/."""
    class_dir = output_root / spec.name
    class_dir.mkdir(parents=True, exist_ok=True)

    if not spec.mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh not found for class {spec.name}: {spec.mesh_path}"
        )

    produced = 0
    for i in range(n_samples):
        seed = seed_offset + i
        try:
            render_single(
                mesh_path=spec.mesh_path,
                out_dir=class_dir,
                config=render_config,
                dr_config=dr_config,
                seed=seed,
            )
            produced += 1
        except Exception as e:
            print(f"[synth] skip {spec.name} seed={seed}: {e}")
    return produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/cad_sources.yaml"))
    ap.add_argument("--cad-root", type=Path, default=Path("data/cad"))
    ap.add_argument("--output", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--hdri-dir", type=Path, default=None)
    ap.add_argument("--only-class", type=str, default=None,
                    help="Render only this class (debug / partial reruns)")
    args = ap.parse_args()

    specs = load_class_specs(args.config, args.cad_root)
    if args.only_class:
        specs = [s for s in specs if s.name == args.only_class]

    rcfg = RenderConfig(image_size=args.image_size, samples=args.samples)
    dr = DomainRandomizationConfig(hdri_dir=args.hdri_dir) if args.hdri_dir else None

    for i, spec in enumerate(specs):
        try:
            n = render_class(
                spec=spec,
                n_samples=args.per_class,
                output_root=args.output,
                render_config=rcfg,
                dr_config=dr,
                seed_offset=i * args.per_class,
            )
            print(f"[synth] {spec.name}: {n}/{args.per_class} rendered")
        except FileNotFoundError as e:
            print(f"[synth] skip class {spec.name}: {e}")


if __name__ == "__main__":
    main()
