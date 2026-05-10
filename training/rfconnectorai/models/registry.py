"""Model and dataset version registry.

Every detector/classifier/embedder run must produce a `ModelRecord` so that
metrics, dataset hashes, and exported artifacts can be matched back to the
exact training configuration. Without this, experiment results drift apart
once multiple architectures and dataset revisions are in play.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_MODEL_TYPES = {
    "detector",
    "classifier",
    "multihead_classifier",
    "embedder",
    "geometry_verifier",
}


@dataclass(frozen=True)
class ModelRecord:
    """Metadata for one trained model artifact."""

    model_id: str
    model_type: str
    architecture: str
    trained_on: str  # e.g. "datasets/rfconnectors@<sha256>"
    taxonomy_version: str  # sha256 of connectors.yaml at training time
    metrics_path: str
    artifact_path: str
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_model_record(
    *,
    model_type: str,
    architecture: str,
    trained_on: str,
    taxonomy_version: str,
    metrics_path: str,
    artifact_path: str,
    seq: int = 1,
    now: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> ModelRecord:
    """Build a `ModelRecord` with a deterministic `model_id`.

    The model_id format is `<model_type>_<architecture>_<YYYY-MM-DD>_<seq:03d>`
    so that registry rows sort chronologically and stay readable in logs.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"model_type {model_type!r} not in VALID_MODEL_TYPES {sorted(VALID_MODEL_TYPES)}"
        )
    if seq < 0 or seq > 999:
        raise ValueError(f"seq must be in [0, 999]; got {seq}")

    timestamp = now or datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y-%m-%d")
    model_id = f"{model_type}_{architecture}_{date_str}_{seq:03d}"
    return ModelRecord(
        model_id=model_id,
        model_type=model_type,
        architecture=architecture,
        trained_on=trained_on,
        taxonomy_version=taxonomy_version,
        metrics_path=metrics_path,
        artifact_path=artifact_path,
        created_at=timestamp.isoformat(),
        extra=dict(extra or {}),
    )


def write_record(record: ModelRecord, path: Path | str) -> None:
    """Persist `record` to `path` as pretty-printed JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, sort_keys=True)


def read_record(path: Path | str) -> ModelRecord:
    """Load a `ModelRecord` previously written by `write_record`."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelRecord(
        model_id=data["model_id"],
        model_type=data["model_type"],
        architecture=data["architecture"],
        trained_on=data["trained_on"],
        taxonomy_version=data["taxonomy_version"],
        metrics_path=data["metrics_path"],
        artifact_path=data["artifact_path"],
        created_at=data["created_at"],
        extra=data.get("extra", {}),
    )
