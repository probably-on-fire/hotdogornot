"""
Tests for the thread-pitch FFT scale recovery.

We synthesize images of a periodic stripe pattern at known pixels-per-mm,
run the detector, and check it recovers the right ppm within a few percent.
"""
import numpy as np
import pytest

from rfconnectorai.measurement.thread_pitch_scale import (
    KNOWN_PITCHES_MM,
    detect_thread_pitch,
)


def _make_threaded_strip(
    height: int,
    width: int,
    pixels_per_mm: float,
    pitch_mm: float = 0.7056,
    contrast: int = 80,
    noise: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    Synthesize an image where a thread-like sinusoidal pattern runs along
    the row direction (Y axis), at the spatial frequency a real connector
    would produce given pixels_per_mm and pitch_mm.
    """
    rng = np.random.default_rng(seed)
    pitch_px = pitch_mm * pixels_per_mm  # pixels per cycle
    yy = np.arange(height, dtype=np.float32)
    pattern = 128 + (contrast / 2) * np.sin(2 * np.pi * yy / pitch_px)
    img = np.tile(pattern[:, None], (1, width))
    noise_arr = rng.integers(-noise, noise + 1, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    return img


def test_recovers_pixels_per_mm_within_5_percent():
    target_ppm = 30.0
    pitch_mm = KNOWN_PITCHES_MM["SMA"]
    img = _make_threaded_strip(height=300, width=40, pixels_per_mm=target_ppm,
                               pitch_mm=pitch_mm)
    est = detect_thread_pitch(img, roi=(0, 0, 40, 300), pitch_mm=pitch_mm)
    assert est is not None
    assert abs(est.pixels_per_mm - target_ppm) / target_ppm < 0.05


def test_recovers_higher_resolution_correctly():
    target_ppm = 60.0
    pitch_mm = KNOWN_PITCHES_MM["2.4mm"]
    img = _make_threaded_strip(height=400, width=40, pixels_per_mm=target_ppm,
                               pitch_mm=pitch_mm)
    est = detect_thread_pitch(img, roi=(0, 0, 40, 400), pitch_mm=pitch_mm)
    assert est is not None
    assert abs(est.pixels_per_mm - target_ppm) / target_ppm < 0.05


def test_returns_none_on_uniform_image():
    img = np.full((300, 40), 128, dtype=np.uint8)
    est = detect_thread_pitch(img, roi=(0, 0, 40, 300), pitch_mm=0.7)
    assert est is None


def test_returns_none_on_pure_noise():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(300, 40), dtype=np.uint8)
    est = detect_thread_pitch(img, roi=(0, 0, 40, 300), pitch_mm=0.7)
    # Pure noise has no clean periodic peak — detector should refuse.
    assert est is None


def test_returns_none_on_too_small_roi():
    img = np.full((300, 40), 128, dtype=np.uint8)
    est = detect_thread_pitch(img, roi=(0, 0, 3, 5), pitch_mm=0.7)
    assert est is None


def test_works_on_rgb_input():
    target_ppm = 30.0
    gray = _make_threaded_strip(height=300, width=40, pixels_per_mm=target_ppm)
    rgb = np.stack([gray, gray, gray], axis=-1)
    est = detect_thread_pitch(rgb, roi=(0, 0, 40, 300), pitch_mm=KNOWN_PITCHES_MM["SMA"])
    assert est is not None
    assert abs(est.pixels_per_mm - target_ppm) / target_ppm < 0.05


def test_known_pitches_table_has_all_families():
    assert "SMA" in KNOWN_PITCHES_MM
    assert "3.5mm" in KNOWN_PITCHES_MM
    assert "2.92mm" in KNOWN_PITCHES_MM
    assert "2.4mm" in KNOWN_PITCHES_MM
