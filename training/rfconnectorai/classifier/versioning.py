"""
Model versioning + manifest for the OTA model-update path.

Each successful retrain bumps an integer version, snapshots the trained
weights as `weights.<version>.pt`, points `weights.latest.pt` at the
newest snapshot, and writes a manifest:

    models/connector_classifier/
      weights.latest.pt          ← always points to current
      weights.0001.pt
      weights.0002.pt
      labels.json                ← class names, input size
      version.json               ← {"version": 2, "trained_at": "...",
                                     "n_train_samples": ..., "val_acc": ...}
      manifest.json              ← what the server publishes:
                                   {"version": 2,
                                    "weights_filename": "weights.0002.pt",
                                    "labels_filename": "labels.json",
                                    "sha256": "..."}

The relay server reads `manifest.json`, hosts the referenced files behind a
stable URL pattern, and bumps its `/model/latest` endpoint when it sees a
new manifest version. Clients (the AR app) poll `/model/version`, compare
to local, and fetch the new files when they differ.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


VERSION_FILENAME = "version.json"
MANIFEST_FILENAME = "manifest.json"
LABELS_FILENAME = "labels.json"
LATEST_WEIGHTS = "weights.latest.pt"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_version(model_dir: Path) -> int:
    """Return the current integer version, or 0 if none exists yet."""
    p = model_dir / VERSION_FILENAME
    if not p.exists():
        return 0
    try:
        return int(_read_json(p).get("version", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def bump_version(
    model_dir: Path,
    weights_path: Path | None = None,
    val_acc: float | None = None,
    n_train_samples: int | None = None,
) -> int:
    """
    Snapshot the current `weights.pt` under a versioned filename, point
    `weights.latest.pt` at it, and refresh the manifest. Returns the new
    version number.

    `weights_path` is an explicit path to the just-trained `.pt` file.
    Defaults to `model_dir / "weights.pt"` (which is what train.py writes).
    """
    model_dir = Path(model_dir)
    if weights_path is None:
        weights_path = model_dir / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if not (model_dir / LABELS_FILENAME).exists():
        raise FileNotFoundError(f"labels.json not found in {model_dir}")

    new_version = current_version(model_dir) + 1
    versioned = model_dir / f"weights.{new_version:04d}.pt"
    shutil.copyfile(weights_path, versioned)

    # Repoint "latest". On Windows we can't rely on real symlinks without
    # admin rights; just copy to keep the deploy path simple. (The versioned
    # file is the canonical artifact; "latest" is a convenience pointer.)
    latest = model_dir / LATEST_WEIGHTS
    if latest.exists():
        latest.unlink()
    shutil.copyfile(versioned, latest)

    version_blob = {
        "version": new_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "weights_filename": versioned.name,
        "n_train_samples": n_train_samples,
        "val_acc": val_acc,
    }
    _write_json(model_dir / VERSION_FILENAME, version_blob)

    manifest = {
        "version": new_version,
        "weights_filename": versioned.name,
        "labels_filename": LABELS_FILENAME,
        "weights_sha256": _sha256(versioned),
        "labels_sha256": _sha256(model_dir / LABELS_FILENAME),
        "trained_at": version_blob["trained_at"],
    }
    _write_json(model_dir / MANIFEST_FILENAME, manifest)

    return new_version


def read_manifest(model_dir: Path) -> dict:
    """Return the published manifest dict (raises if not yet versioned)."""
    p = model_dir / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p} — run bump_version first")
    return _read_json(p)
