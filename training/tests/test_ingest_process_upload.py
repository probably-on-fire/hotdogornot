"""
Tests for the auto-trust ingestion. Build small frame uploads on disk,
mock the classifier so we can pin exact ensemble decisions, and verify
each upload routes to the correct destination.
"""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.classifier.predict import ClassifierPrediction
from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ingest.process_upload import (
    IngestionConfig,
    process_upload,
)


def _make_face(path: Path, hex_mm: float, ap_mm: float, ppm: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (500, 500), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = 250 + int(rng.integers(-3, 4))
    cy = 250 + int(rng.integers(-3, 4))
    hex_ff_px = hex_mm * ppm
    apothem = hex_ff_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    verts = [
        (cx + circumradius * math.cos(math.radians(60 * i + 30)),
         cy + circumradius * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]
    draw.polygon(verts, fill=(90, 90, 90))
    r = ap_mm * ppm / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15))
    img.save(path)


def _build_upload_dir(root: Path, n_frames: int = 5) -> Path:
    upload = root / "upload_001"
    upload.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        _make_face(upload / f"frame_{i:03d}.png", 6.35, 2.4, 30.0, seed=i)
    return upload


def _stub_classifier(class_name: str, confidence: float) -> MagicMock:
    classifier = MagicMock()
    classifier.predict.return_value = ClassifierPrediction(
        class_name=class_name,
        confidence=confidence,
        probabilities={class_name: confidence, "other": 1 - confidence},
    )
    return classifier


def test_approve_when_user_label_matches_high_confidence(tmp_path):
    upload = _build_upload_dir(tmp_path)
    classifier = _stub_classifier("2.4mm-F", 0.95)
    predictor = EnsemblePredictor(classifier=classifier)
    labeled_root = tmp_path / "labeled"
    quarantine_root = tmp_path / "quarantine"
    decision = process_upload(
        upload, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=labeled_root, quarantine_root=quarantine_root,
    )
    assert decision.decision == "approve"
    assert decision.destination == labeled_root / "2.4mm-F"
    files = list((labeled_root / "2.4mm-F").glob("upload_*.png"))
    assert len(files) == 5
    # Sidecar metadata written.
    assert decision.metadata_written is not None
    meta = json.loads(decision.metadata_written.read_text())
    assert meta["decision"] == "approve"
    assert meta["claimed_class"] == "2.4mm-F"


def test_quarantine_when_user_label_disagrees_with_ensemble(tmp_path):
    upload = _build_upload_dir(tmp_path)
    classifier = _stub_classifier("2.4mm-F", 0.95)
    predictor = EnsemblePredictor(classifier=classifier)
    labeled_root = tmp_path / "labeled"
    quarantine_root = tmp_path / "quarantine"
    # User claimed something different — ensemble strongly disagrees.
    decision = process_upload(
        upload, claimed_class="3.5mm-M", predictor=predictor,
        labeled_root=labeled_root, quarantine_root=quarantine_root,
    )
    assert decision.decision == "quarantine"
    assert decision.destination is not None
    assert "3.5mm-M" in str(decision.destination)
    # Original labeled folder stays empty.
    assert not (labeled_root / "3.5mm-M").exists() or not list((labeled_root / "3.5mm-M").iterdir())


def test_quarantine_when_confidence_below_threshold(tmp_path):
    upload = _build_upload_dir(tmp_path)
    # Low classifier confidence → ensemble confidence drops below 0.7.
    classifier = _stub_classifier("2.4mm-F", 0.30)
    predictor = EnsemblePredictor(classifier=classifier)
    labeled_root = tmp_path / "labeled"
    quarantine_root = tmp_path / "quarantine"
    decision = process_upload(
        upload, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=labeled_root, quarantine_root=quarantine_root,
        config=IngestionConfig(approve_confidence=0.85),
    )
    assert decision.decision == "quarantine"


def test_drop_when_no_frames_present(tmp_path):
    empty = tmp_path / "empty_upload"
    empty.mkdir()
    classifier = _stub_classifier("2.4mm-F", 0.95)
    predictor = EnsemblePredictor(classifier=classifier)
    decision = process_upload(
        empty, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=tmp_path / "labeled",
        quarantine_root=tmp_path / "quarantine",
    )
    assert decision.decision == "drop"
    assert decision.destination is None


def test_drop_when_all_frames_unknown(tmp_path):
    # Blank uniform frames → measurement returns Unknown, no classifier.
    upload = tmp_path / "blank_upload"
    upload.mkdir()
    for i in range(3):
        Image.fromarray(np.full((400, 400, 3), 240, dtype=np.uint8)).save(
            upload / f"frame_{i:03d}.png"
        )
    predictor = EnsemblePredictor(classifier=None)
    decision = process_upload(
        upload, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=tmp_path / "labeled",
        quarantine_root=tmp_path / "quarantine",
    )
    assert decision.decision == "drop"


def test_idempotent_re_run_dedups(tmp_path):
    upload = _build_upload_dir(tmp_path)
    classifier = _stub_classifier("2.4mm-F", 0.95)
    predictor = EnsemblePredictor(classifier=classifier)
    labeled_root = tmp_path / "labeled"
    quarantine_root = tmp_path / "quarantine"
    process_upload(
        upload, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=labeled_root, quarantine_root=quarantine_root,
    )
    n_after_first = len(list((labeled_root / "2.4mm-F").glob("upload_*.png")))
    # Re-run: same frames, same md5s → should not duplicate.
    process_upload(
        upload, claimed_class="2.4mm-F", predictor=predictor,
        labeled_root=labeled_root, quarantine_root=quarantine_root,
    )
    n_after_second = len(list((labeled_root / "2.4mm-F").glob("upload_*.png")))
    assert n_after_second == n_after_first
