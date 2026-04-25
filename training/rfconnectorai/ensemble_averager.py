"""
Multi-frame ensemble averaging — runs the EnsemblePredictor across many
frames (e.g. all frames extracted from a capture video) and consolidates the
results into one decision.

The single-frame averager (`measurement.frame_averager`) only votes on the
measurement-pipeline output. This module additionally averages the
classifier's softmax probabilities across frames before picking the winner —
a soft vote that's more robust than per-frame argmax voting because it
preserves uncertainty information.

Final decision logic (per frame):
  - If ensemble.agreement == "agree" or "measurement_only" or "classifier_only",
    contribute the predicted class to votes
  - Aggregate classifier probabilities across all frames where classifier fired
  - Aggregate measurement aperture/hex/ppm samples across consensus frames
  - Final class = highest-probability class from aggregated classifier softmax
                  (falls back to measurement vote winner if no classifier)
  - Final confidence = blended (avg_agree_score, mean classifier prob, frame fraction)

Public surface:

    avg = average_ensemble(frames, predictor, **kwargs)
    # avg.class_name, avg.confidence, avg.aperture_mm, avg.aperture_mm_stddev,
    # avg.classifier_probabilities (averaged), avg.per_frame_agreement
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from rfconnectorai.ensemble import EnsemblePredictor, EnsembleResult
from rfconnectorai.measurement.frame_averager import _mad_filter


@dataclass
class AveragedEnsembleResult:
    class_name: str
    confidence: float
    n_frames_total: int
    n_frames_used: int                          # frames where at least one pipeline fired
    aperture_mm: float | None = None
    aperture_mm_stddev: float | None = None
    hex_flat_to_flat_mm: float | None = None
    pixels_per_mm: float | None = None
    classifier_probabilities: dict[str, float] = field(default_factory=dict)
    per_class_votes: dict[str, int] = field(default_factory=dict)
    per_frame_agreement: dict[str, int] = field(default_factory=dict)
    reason: str = ""


def average_ensemble(
    frames: Iterable[np.ndarray],
    predictor: EnsemblePredictor,
    require_aruco: bool = True,
    aruco_marker_size_mm: float | None = 25.0,
) -> AveragedEnsembleResult:
    """
    Aggregate EnsemblePredictor results across a sequence of frames.

    Defaults to require_aruco=True since multi-frame averaging is intended
    for capture-mode video (where the ArUco marker should be in frame); the
    Process Video page lets the caller flip it off for one-off review of
    captures without a marker.
    """
    per_frame: list[EnsembleResult] = []
    for frame in frames:
        per_frame.append(predictor.predict(
            frame,
            require_aruco=require_aruco,
            aruco_marker_size_mm=aruco_marker_size_mm,
        ))

    n_total = len(per_frame)
    if n_total == 0:
        return AveragedEnsembleResult(
            class_name="Unknown",
            confidence=0.0,
            n_frames_total=0,
            n_frames_used=0,
            reason="no frames provided",
        )

    agreement_counts = Counter(r.agreement for r in per_frame)
    used = [r for r in per_frame if r.class_name != "Unknown"]
    n_used = len(used)

    if n_used == 0:
        reasons = Counter(r.reason for r in per_frame if r.reason)
        return AveragedEnsembleResult(
            class_name="Unknown",
            confidence=0.0,
            n_frames_total=n_total,
            n_frames_used=0,
            per_frame_agreement=dict(agreement_counts),
            reason=reasons.most_common(1)[0][0] if reasons else "all frames unknown",
        )

    # Hard-vote across the ensemble class predictions.
    class_votes = Counter(r.class_name for r in used)

    # Soft-vote: average classifier softmax probabilities across all frames
    # where the classifier fired (regardless of measurement-pipeline outcome).
    clf_results = [r.classifier for r in per_frame if r.classifier is not None]
    softmax_avg: dict[str, float] = {}
    if clf_results:
        all_classes = set()
        for c in clf_results:
            all_classes.update(c.probabilities.keys())
        sums = {k: 0.0 for k in all_classes}
        for c in clf_results:
            for k, v in c.probabilities.items():
                sums[k] += v
        n_clf = len(clf_results)
        softmax_avg = {k: v / n_clf for k, v in sums.items()}

    # Final class: prefer the soft-voted classifier winner if available;
    # otherwise the hard-voted ensemble winner.
    if softmax_avg:
        winning_class = max(softmax_avg, key=softmax_avg.get)
        soft_conf = softmax_avg[winning_class]
    else:
        winning_class, _ = class_votes.most_common(1)[0]
        soft_conf = class_votes[winning_class] / n_used

    # Confidence: blend soft probability with frame coverage.
    coverage = n_used / n_total
    confidence = 0.6 * soft_conf + 0.4 * coverage

    # Average measurements across frames where the measurement pipeline fired
    # AND the measurement agreed with the winning class (consistency).
    consensus_frames = [
        r for r in used
        if r.measurement is not None
        and r.measurement.class_name == winning_class
    ]
    aperture_samples = [
        r.measurement.aperture_mm for r in consensus_frames
        if r.measurement.aperture_mm is not None
    ]
    hex_samples = [
        r.measurement.hex_flat_to_flat_mm for r in consensus_frames
        if r.measurement.hex_flat_to_flat_mm is not None
    ]
    ppm_samples = [
        r.measurement.pixels_per_mm for r in consensus_frames
        if r.measurement.pixels_per_mm is not None
    ]

    aperture_filtered = _mad_filter(aperture_samples)
    hex_filtered = _mad_filter(hex_samples)
    ppm_filtered = _mad_filter(ppm_samples)

    return AveragedEnsembleResult(
        class_name=winning_class,
        confidence=float(confidence),
        n_frames_total=n_total,
        n_frames_used=n_used,
        aperture_mm=float(np.mean(aperture_filtered)) if aperture_filtered else None,
        aperture_mm_stddev=float(np.std(aperture_filtered)) if len(aperture_filtered) > 1 else None,
        hex_flat_to_flat_mm=float(np.mean(hex_filtered)) if hex_filtered else None,
        pixels_per_mm=float(np.mean(ppm_filtered)) if ppm_filtered else None,
        classifier_probabilities=softmax_avg,
        per_class_votes=dict(class_votes),
        per_frame_agreement=dict(agreement_counts),
    )
