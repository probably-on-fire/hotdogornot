"""
Multi-frame averaging for the measurement pipeline.

Single-frame measurements have ~0.3mm noise on aperture diameter — borderline
for distinguishing 2.92mm vs 3.5mm even with an ArUco marker. Averaging over
30+ frames at 10fps brings noise down to <0.05mm while staying fully online
(each frame's prediction is interpretable on its own).

The averager:
  - Runs predict_class on each frame (with require_aruco=True by default —
    averaging is for accuracy-mode capture, not loose mode)
  - Collects per-frame aperture_mm, hex_flat_to_flat_mm, pixels_per_mm samples
  - Drops outlier frames using median absolute deviation (MAD) — robust to
    occasional bad detections without needing a frame count threshold
  - Votes on family (sma vs precision) and gender (male vs female)
  - Returns one AveragedPrediction with the consensus class plus confidence
    (fraction of frames that agreed) and the spread of measurements

Public surface:

    average_predictions(frames, **predict_kwargs) -> AveragedPrediction

`frames` is any iterable of HxWx3 RGB uint8 arrays.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from rfconnectorai.measurement.class_predictor import predict_class, Prediction


@dataclass
class AveragedPrediction:
    class_name: str
    confidence: float                       # 0..1, fraction of frames that voted for the winning class
    n_frames_total: int
    n_frames_used: int                      # frames that produced a non-Unknown prediction
    aperture_mm: float | None = None
    aperture_mm_stddev: float | None = None
    hex_flat_to_flat_mm: float | None = None
    pixels_per_mm: float | None = None
    family: str | None = None
    gender: str | None = None
    per_class_votes: dict[str, int] = field(default_factory=dict)
    reason: str = ""


def _mad_filter(values: list[float], threshold: float = 3.0) -> list[float]:
    """
    Median absolute deviation outlier filter. Returns the subset of `values`
    within `threshold` MADs of the median. Falls back to all values when the
    sample is too small to estimate MAD reliably.
    """
    if len(values) < 4:
        return list(values)
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    abs_dev = np.abs(arr - median)
    mad = float(np.median(abs_dev))
    if mad == 0:
        # All values identical (or close to it) — nothing to filter.
        return list(values)
    # Scale so MAD is comparable to a stddev for a normal distribution.
    scaled_dev = abs_dev / (mad * 1.4826)
    keep_mask = scaled_dev <= threshold
    return arr[keep_mask].tolist()


def average_predictions(
    frames: Iterable[np.ndarray],
    require_aruco: bool = True,
    aruco_marker_size_mm: float | None = 25.0,
    assumed_pixels_per_mm: float | None = None,
) -> AveragedPrediction:
    """
    Aggregate predictions across a sequence of frames into one AveragedPrediction.

    Defaults to require_aruco=True because frame-averaging only makes sense
    when scale calibration is consistent across frames — the hex-hypothesis
    fallback would let one bad frame swing the vote into the wrong size class.
    """
    per_frame: list[Prediction] = []
    for frame in frames:
        pred = predict_class(
            frame,
            assumed_pixels_per_mm=assumed_pixels_per_mm,
            aruco_marker_size_mm=aruco_marker_size_mm,
            require_aruco=require_aruco,
        )
        per_frame.append(pred)

    n_total = len(per_frame)
    if n_total == 0:
        return AveragedPrediction(
            class_name="Unknown",
            confidence=0.0,
            n_frames_total=0,
            n_frames_used=0,
            reason="no frames provided",
        )

    valid = [p for p in per_frame if p.class_name != "Unknown"]
    n_used = len(valid)
    if n_used == 0:
        # All frames came back Unknown; surface the most common reason.
        reasons = Counter(p.reason for p in per_frame if p.reason)
        top_reason = reasons.most_common(1)[0][0] if reasons else "all frames produced Unknown"
        return AveragedPrediction(
            class_name="Unknown",
            confidence=0.0,
            n_frames_total=n_total,
            n_frames_used=0,
            per_class_votes={"Unknown": n_total},
            reason=top_reason,
        )

    # Vote on class — winner is the most-frequent non-Unknown prediction.
    class_votes = Counter(p.class_name for p in valid)
    winning_class, winning_count = class_votes.most_common(1)[0]
    confidence = winning_count / n_used

    # Average measurements across frames that voted for the winning class only —
    # mixing measurements from disagreeing frames would smear the result.
    consensus_frames = [p for p in valid if p.class_name == winning_class]

    aperture_samples = [p.aperture_mm for p in consensus_frames if p.aperture_mm is not None]
    hex_samples = [p.hex_flat_to_flat_mm for p in consensus_frames if p.hex_flat_to_flat_mm is not None]
    ppm_samples = [p.pixels_per_mm for p in consensus_frames if p.pixels_per_mm is not None]

    aperture_filtered = _mad_filter(aperture_samples)
    hex_filtered = _mad_filter(hex_samples)
    ppm_filtered = _mad_filter(ppm_samples)

    family_votes = Counter(p.family for p in consensus_frames if p.family is not None)
    gender_votes = Counter(p.gender for p in consensus_frames if p.gender is not None)

    return AveragedPrediction(
        class_name=winning_class,
        confidence=confidence,
        n_frames_total=n_total,
        n_frames_used=n_used,
        aperture_mm=float(np.mean(aperture_filtered)) if aperture_filtered else None,
        aperture_mm_stddev=float(np.std(aperture_filtered)) if len(aperture_filtered) > 1 else None,
        hex_flat_to_flat_mm=float(np.mean(hex_filtered)) if hex_filtered else None,
        pixels_per_mm=float(np.mean(ppm_filtered)) if ppm_filtered else None,
        family=family_votes.most_common(1)[0][0] if family_votes else None,
        gender=gender_votes.most_common(1)[0][0] if gender_votes else None,
        per_class_votes=dict(class_votes),
    )
