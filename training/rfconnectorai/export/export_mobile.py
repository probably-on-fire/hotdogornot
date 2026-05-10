"""Mobile/server export entry.

Exports trained detector and classifier artifacts to ONNX (always),
TFLite/LiteRT (where supported), and Core ML (where supported). Real
exports run in the cloud; the local CPU PC should only invoke ``--dry-run``
to validate config and emit a manifest.

The manifest captures the source ``ModelRecord`` for every output so
clients (mobile, server, demo) can identify exactly which trained model
produced an export.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rfconnectorai.models.registry import read_record


SUPPORTED_FORMATS = ("onnx", "tflite", "coreml")


@dataclass(frozen=True)
class ExportTarget:
    name: str  # e.g. "detector" or "classifier"
    artifact: Path  # e.g. models/detector/best.pt
    record: Path  # corresponding model_record.json
    formats: tuple[str, ...]


@dataclass
class ExportManifest:
    generated_at: str
    out_dir: str
    entries: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "out_dir": self.out_dir,
            "entries": self.entries,
            "notes": self.notes,
        }


def validate_targets(targets: Iterable[ExportTarget]) -> None:
    targets = list(targets)
    if not targets:
        raise ValueError("at least one --target is required")
    for target in targets:
        unknown = set(target.formats) - set(SUPPORTED_FORMATS)
        if unknown:
            raise ValueError(
                f"target {target.name!r} requested unsupported formats {sorted(unknown)}; "
                f"supported: {SUPPORTED_FORMATS}"
            )
        if not target.record.exists():
            raise FileNotFoundError(
                f"target {target.name!r} model_record.json missing: {target.record}"
            )


def plan_exports(
    targets: Iterable[ExportTarget],
    *,
    out_dir: Path,
) -> ExportManifest:
    manifest = ExportManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        out_dir=str(out_dir),
    )
    for target in targets:
        record = read_record(target.record)
        for fmt in target.formats:
            output_path = out_dir / f"{target.name}_{record.architecture}_{record.model_id}.{fmt}"
            manifest.entries.append(
                {
                    "target": target.name,
                    "format": fmt,
                    "source_artifact": str(target.artifact),
                    "model_record": target.record.as_posix(),
                    "model_id": record.model_id,
                    "architecture": record.architecture,
                    "trained_on": record.trained_on,
                    "taxonomy_version": record.taxonomy_version,
                    "output": str(output_path),
                }
            )
    return manifest


def run_exports(manifest: ExportManifest) -> ExportManifest:  # pragma: no cover - cloud only
    """Cloud-only export executor.

    The local PC should never call this. The cloud pipeline replaces this
    with a real implementation that imports torch / onnx / coremltools /
    tensorflow as needed for each format.
    """
    raise NotImplementedError(
        "run_exports() is intentionally a stub on the local PC. Run the "
        "cloud notebook to produce real ONNX/TFLite/CoreML artifacts."
    )


def write_manifest(manifest: ExportManifest, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "exports_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_target(spec: str) -> ExportTarget:
    """Parse ``name:artifact:record:fmt1,fmt2`` from CLI."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--target must be name:artifact:record:fmts; got {spec!r}"
        )
    name, artifact, record, fmts = parts
    return ExportTarget(
        name=name,
        artifact=Path(artifact),
        record=Path(record),
        formats=tuple(f.strip() for f in fmts.split(",") if f.strip()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export models for mobile/server use")
    parser.add_argument(
        "--target",
        type=parse_target,
        action="append",
        required=True,
        help="name:artifact_path:model_record_path:fmt1,fmt2 (repeat for each model)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output dir for exports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    validate_targets(args.target)
    manifest = plan_exports(args.target, out_dir=args.out)
    manifest_path = write_manifest(manifest, args.out)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest": str(manifest_path)}, indent=2))
        return 0
    run_exports(manifest)
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
