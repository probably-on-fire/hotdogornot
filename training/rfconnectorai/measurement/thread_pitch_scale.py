"""
Thread-pitch FFT for absolute scale recovery.

Connector coupling threads are standardized: SMA / 3.5mm / 2.92mm all use
1/4-36 UNS-2A (pitch ≈ 0.706 mm), and 2.4mm uses #5-40 UNF-2A (pitch ≈ 0.635
mm). On a frame where the threaded post is visible (any female connector and
the inside of any male coupling nut at a slight angle), the threads form a
bright/dark periodic pattern along the post axis — its spatial frequency is
*directly* a pixels-per-mm reference.

This is a backup scale source for when an ArUco marker isn't in the frame.
The FFT approach:

  1. Caller provides a roughly-rectangular ROI across the threaded section,
     oriented so the connector axis runs vertically (along the rows).
  2. Average each row to a 1D intensity profile (works even when the post
     occupies only part of the rectangle, because lateral variation cancels).
  3. Take the magnitude FFT of the profile.
  4. Find the dominant non-DC peak — its frequency is the thread crests per
     pixel (cycles/pixel).
  5. peak_cycles_per_pixel × known_pitch_mm = mm/pixel → invert for ppm.

Public surface:

    detect_thread_pitch(image, roi=(x, y, w, h), pitch_mm) -> ScaleEstimate

Returns None if no clean periodic pattern is detected (low SNR, motion
blur, occluded threads). Caller decides whether to trust it as scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Standardized thread pitches by connector family. Sources: industry datasheets.
# SMA / 3.5mm / 2.92mm — 1/4-36 UNS-2A:  pitch = 1/36 in = 0.7056 mm
# 2.4mm — #5-40 UNF-2A:                  pitch = 1/40 in = 0.6350 mm
KNOWN_PITCHES_MM = {
    "SMA":     0.7056,
    "3.5mm":   0.7056,
    "2.92mm":  0.7056,
    "2.4mm":   0.6350,
}


@dataclass
class ScaleEstimate:
    pixels_per_mm: float
    peak_cycles_per_pixel: float
    snr: float
    pitch_mm_used: float
    reason: str = ""


def _row_mean_profile(roi_gray: np.ndarray) -> np.ndarray:
    """Average each row → 1D intensity vector along the connector axis."""
    return roi_gray.mean(axis=1).astype(np.float32)


def _detrend(profile: np.ndarray) -> np.ndarray:
    """Subtract a low-frequency component so the FFT isn't dominated by DC + slow gradients."""
    # Subtract a long-window moving average; window length scales with profile size.
    win = max(9, len(profile) // 8)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(profile, pad, mode="reflect")
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(padded, kernel, mode="valid")
    return profile - smooth


def detect_thread_pitch(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    pitch_mm: float,
    min_cycles: int = 4,
) -> ScaleEstimate | None:
    """
    Recover pixels-per-mm by FFT-detecting the thread spatial frequency.

    `image` is the full frame (RGB or grayscale uint8).
    `roi` is (x, y, w, h) of a rectangle covering the threaded post, oriented
    so the connector axis runs along the long (vertical, h) dimension.
    `pitch_mm` is the known thread pitch — caller picks based on suspected
    family, or runs detection multiple times and trusts the highest-SNR result.

    Returns None when:
      - The ROI is too small (need at least min_cycles*pitch_in_pixels)
      - No non-DC FFT peak rises clearly above the noise floor
    """
    x, y, w, h = roi
    if w <= 4 or h <= 4 * min_cycles:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    img_h, img_w = gray.shape
    x = max(0, x); y = max(0, y)
    x1 = min(img_w, x + w); y1 = min(img_h, y + h)
    if x1 - x < 4 or y1 - y < 4 * min_cycles:
        return None
    roi_gray = gray[y:y1, x:x1]

    profile = _row_mean_profile(roi_gray)
    profile = _detrend(profile)

    # Hann window reduces spectral leakage so the dominant peak is sharper.
    n = len(profile)
    window = np.hanning(n).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(profile * window))

    # Skip the lowest frequencies (still partially DC after detrending).
    low_skip = max(2, min_cycles)
    if low_skip >= len(spectrum):
        return None
    search = spectrum[low_skip:]
    peak_offset = int(np.argmax(search))
    peak_idx = peak_offset + low_skip
    peak_value = float(spectrum[peak_idx])
    # SNR vs the median magnitude of the rest of the spectrum.
    others = np.concatenate([spectrum[1:peak_idx], spectrum[peak_idx + 1:]])
    noise_floor = float(np.median(others)) if len(others) else 1e-9
    snr = peak_value / max(noise_floor, 1e-9)

    # Real thread patterns produce SNR in the hundreds; pure noise tops out
    # around 3-4. Threshold at 5 to comfortably accept the former and reject
    # the latter without trimming the long tail of borderline real captures.
    if snr < 5.0:
        return None

    cycles_per_pixel = peak_idx / float(n)
    if cycles_per_pixel <= 0:
        return None
    mm_per_pixel = pitch_mm * cycles_per_pixel
    if mm_per_pixel <= 0:
        return None
    pixels_per_mm = 1.0 / mm_per_pixel

    return ScaleEstimate(
        pixels_per_mm=float(pixels_per_mm),
        peak_cycles_per_pixel=float(cycles_per_pixel),
        snr=float(snr),
        pitch_mm_used=float(pitch_mm),
    )
