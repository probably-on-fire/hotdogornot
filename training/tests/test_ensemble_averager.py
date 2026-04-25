"""
Tests for the multi-frame ensemble averager.

We use a stub classifier (MagicMock) to avoid loading torch weights, and
real synthetic frames so the measurement pipeline actually fires.
"""
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.classifier.predict import ClassifierPrediction
from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ensemble_averager import AveragedEnsembleResult, average_ensemble


def _make_connector_face(
    image_size: int,
    hex_flat_to_flat_mm: float,
    aperture_diameter_mm: float,
    pixels_per_mm: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = image_size // 2 + int(rng.integers(-3, 4))
    cy = image_size // 2 + int(rng.integers(-3, 4))
    hex_ff_px = hex_flat_to_flat_mm * pixels_per_mm
    apothem = hex_ff_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    rotation_offset = float(rng.uniform(-0.05, 0.05))
    verts = []
    for i in range(6):
        a = math.radians(60 * i + 30) + rotation_offset
        x = cx + circumradius * math.cos(a)
        y = cy + circumradius * math.sin(a)
        verts.append((x, y))
    draw.polygon(verts, fill=(90, 90, 90))
    ap_d_px = aperture_diameter_mm * pixels_per_mm
    r = ap_d_px / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15))
    return np.array(img)


def _stub_classifier(class_name: str, confidence: float) -> MagicMock:
    classifier = MagicMock()
    classifier.predict.return_value = ClassifierPrediction(
        class_name=class_name,
        confidence=confidence,
        probabilities={class_name: confidence, "other": 1 - confidence},
    )
    return classifier


def test_empty_input_returns_unknown():
    predictor = EnsemblePredictor(classifier=None)
    result = average_ensemble([], predictor, require_aruco=False)
    assert result.class_name == "Unknown"
    assert result.n_frames_total == 0


def test_consistent_frames_with_classifier_agreement():
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(5)
    ]
    classifier = _stub_classifier("2.4mm-F", confidence=0.9)
    predictor = EnsemblePredictor(classifier=classifier)
    result = average_ensemble(frames, predictor, require_aruco=False)
    assert result.class_name == "2.4mm-F"
    assert result.n_frames_used == 5
    assert result.aperture_mm is not None
    assert abs(result.aperture_mm - 2.4) < 0.3
    assert result.confidence > 0.7


def test_classifier_softmax_averaging():
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(4)
    ]
    classifier = _stub_classifier("2.4mm-F", confidence=0.85)
    predictor = EnsemblePredictor(classifier=classifier)
    result = average_ensemble(frames, predictor, require_aruco=False)
    # Averaged classifier probabilities should reflect the stub's value.
    assert "2.4mm-F" in result.classifier_probabilities
    assert abs(result.classifier_probabilities["2.4mm-F"] - 0.85) < 0.01


def test_measurement_only_path():
    """No classifier loaded — should still return a class via measurement vote."""
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(4)
    ]
    predictor = EnsemblePredictor(classifier=None)
    result = average_ensemble(frames, predictor, require_aruco=False)
    assert result.class_name == "2.4mm-F"
    assert result.classifier_probabilities == {}


def test_per_frame_agreement_counts_recorded():
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(3)
    ]
    classifier = _stub_classifier("2.4mm-F", confidence=0.9)
    predictor = EnsemblePredictor(classifier=classifier)
    result = average_ensemble(frames, predictor, require_aruco=False)
    # All 3 frames should land in "agree" agreement category.
    assert result.per_frame_agreement.get("agree", 0) == 3
