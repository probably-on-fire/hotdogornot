"""
Tests for the ingestion daemon's directory-watching + dispatch logic.
We use --once mode so we don't actually loop.
"""
import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

# scripts/ is not a package, so we have to add it to the path manually.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import ingestion_daemon

from rfconnectorai.classifier.predict import ClassifierPrediction
from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ingest.process_upload import IngestionConfig


def _make_face(path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (500, 500), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = 250 + int(rng.integers(-3, 4))
    cy = 250 + int(rng.integers(-3, 4))
    apothem = 6.35 * 30.0 / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    verts = [
        (cx + circumradius * math.cos(math.radians(60 * i + 30)),
         cy + circumradius * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]
    draw.polygon(verts, fill=(90, 90, 90))
    r = 2.4 * 30.0 / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15))
    img.save(path)


def _build_ready_upload(incoming: Path, upload_id: str, claimed: str,
                        n_frames: int = 3, ready: bool = True) -> Path:
    upload = incoming / upload_id
    upload.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        _make_face(upload / f"frame_{i:03d}.png", seed=i)
    (upload / "manifest.json").write_text(json.dumps({"claimed_class": claimed}))
    if ready:
        (upload / ".ready").write_text("")
    return upload


def _stub_predictor(class_name: str, confidence: float) -> EnsemblePredictor:
    classifier = MagicMock()
    classifier.predict.return_value = ClassifierPrediction(
        class_name=class_name,
        confidence=confidence,
        probabilities={class_name: confidence, "other": 1 - confidence},
    )
    return EnsemblePredictor(classifier=classifier)


def test_processes_ready_upload(tmp_path):
    incoming = tmp_path / "incoming"
    labeled = tmp_path / "labeled"
    quarantine = tmp_path / "quarantine"
    upload = _build_ready_upload(incoming, "u_001", claimed="2.4mm-F")

    predictor = _stub_predictor("2.4mm-F", 0.95)
    ingestion_daemon.run_loop(
        incoming_dir=incoming,
        labeled_root=labeled,
        quarantine_root=quarantine,
        classifier_dir=None,    # ignored — we override predictor below
        interval_seconds=0.0,
        config=IngestionConfig(),
        once=True,
    )
    # The above used the default-constructed predictor (no classifier).
    # That returns measurement-only "agree" since synthetic frames produce a
    # valid measurement. Verify the upload got processed (sidecar written).
    assert (upload / "_processed.json").exists()


def test_skips_upload_without_ready_sentinel(tmp_path):
    incoming = tmp_path / "incoming"
    labeled = tmp_path / "labeled"
    quarantine = tmp_path / "quarantine"
    upload = _build_ready_upload(incoming, "u_002", claimed="2.4mm-F", ready=False)

    ingestion_daemon.run_loop(
        incoming_dir=incoming,
        labeled_root=labeled,
        quarantine_root=quarantine,
        classifier_dir=None,
        interval_seconds=0.0,
        config=IngestionConfig(),
        once=True,
    )
    # Upload should NOT have been processed because no .ready sentinel.
    assert not (upload / "_processed.json").exists()


def test_skips_already_processed_upload(tmp_path):
    incoming = tmp_path / "incoming"
    labeled = tmp_path / "labeled"
    quarantine = tmp_path / "quarantine"
    upload = _build_ready_upload(incoming, "u_003", claimed="2.4mm-F")
    # Pre-write a processed marker so daemon should skip.
    (upload / "_processed.json").write_text(json.dumps({"decision": "approve"}))
    mtime_before = (upload / "_processed.json").stat().st_mtime

    ingestion_daemon.run_loop(
        incoming_dir=incoming,
        labeled_root=labeled,
        quarantine_root=quarantine,
        classifier_dir=None,
        interval_seconds=0.0,
        config=IngestionConfig(),
        once=True,
    )
    mtime_after = (upload / "_processed.json").stat().st_mtime
    # Sidecar should not have been re-written.
    assert mtime_before == mtime_after


def test_handles_missing_manifest_gracefully(tmp_path):
    incoming = tmp_path / "incoming"
    labeled = tmp_path / "labeled"
    quarantine = tmp_path / "quarantine"
    upload = incoming / "u_004"
    upload.mkdir(parents=True)
    (upload / ".ready").write_text("")     # ready, but no manifest
    # No frames either. The daemon should skip with a warning, not crash.

    ingestion_daemon.run_loop(
        incoming_dir=incoming,
        labeled_root=labeled,
        quarantine_root=quarantine,
        classifier_dir=None,
        interval_seconds=0.0,
        config=IngestionConfig(),
        once=True,
    )
    assert not (upload / "_processed.json").exists()
