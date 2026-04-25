import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.measurement.frame_averager import (
    AveragedPrediction,
    average_predictions,
    _mad_filter,
)


def _make_connector_face(
    image_size: int,
    hex_flat_to_flat_mm: float,
    aperture_diameter_mm: float,
    pixels_per_mm: float,
    seed: int = 0,
) -> np.ndarray:
    """Same renderer used in test_class_predictor — small jitter via seed so
    repeated calls produce slightly different frames (simulating real video)."""
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


def test_mad_filter_drops_obvious_outliers():
    samples = [3.5, 3.5, 3.6, 3.4, 3.5, 9.0]   # 9.0 is the outlier
    kept = _mad_filter(samples)
    assert 9.0 not in kept
    assert all(3.0 <= v <= 4.0 for v in kept)


def test_mad_filter_keeps_all_when_no_outliers():
    samples = [3.5, 3.5, 3.51, 3.49, 3.5, 3.51]
    kept = _mad_filter(samples)
    assert len(kept) == len(samples)


def test_mad_filter_handles_small_samples():
    # Fewer than 4 values: should not filter (insufficient data for robust MAD).
    samples = [1.0, 100.0]
    assert _mad_filter(samples) == [1.0, 100.0]


def test_average_returns_unknown_for_empty_input():
    result = average_predictions([])
    assert isinstance(result, AveragedPrediction)
    assert result.class_name == "Unknown"
    assert result.n_frames_total == 0
    assert result.n_frames_used == 0


def test_average_agrees_across_consistent_frames():
    # 5 consistent frames of a 2.4mm female; require_aruco off so the hex
    # hypothesis works (no real ArUco in these synthetic frames).
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(5)
    ]
    result = average_predictions(frames, require_aruco=False)
    assert result.class_name == "2.4mm-F"
    assert result.confidence == 1.0
    assert result.n_frames_used == 5
    assert result.aperture_mm is not None
    assert abs(result.aperture_mm - 2.4) < 0.3


def test_average_majority_wins_with_one_outlier():
    # 4 consistent 2.4mm-F frames + 1 frame with garbage geometry.
    consistent = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(4)
    ]
    garbage = np.full((500, 500, 3), 240, dtype=np.uint8)
    frames = consistent + [garbage]
    result = average_predictions(frames, require_aruco=False)
    assert result.class_name == "2.4mm-F"
    assert result.n_frames_total == 5
    assert result.n_frames_used == 4
    assert result.confidence == 1.0  # all 4 valid frames agreed


def test_require_aruco_returns_unknown_when_no_marker():
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(3)
    ]
    result = average_predictions(frames, require_aruco=True)
    # No ArUco in synthetic frames → all individual predictions are Unknown.
    assert result.class_name == "Unknown"
    assert result.n_frames_total == 3
    assert result.n_frames_used == 0
    assert "aruco" in result.reason.lower() or "marker" in result.reason.lower()


def test_aperture_stddev_computed_for_multiple_frames():
    frames = [
        _make_connector_face(500, 6.35, 2.4, 30.0, seed=i)
        for i in range(6)
    ]
    result = average_predictions(frames, require_aruco=False)
    assert result.class_name == "2.4mm-F"
    assert result.aperture_mm_stddev is not None
    # Should be small but non-zero given the per-frame jitter we added.
    assert 0.0 <= result.aperture_mm_stddev < 0.5
