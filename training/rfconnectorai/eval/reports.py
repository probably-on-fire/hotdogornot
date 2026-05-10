"""Eval-harness reporting primitives.

Stdlib-only metrics and report writers so the evaluation harness can run on
a Kaggle notebook without dragging in scikit-learn for the basics. Heavier
metrics (calibration curves, etc.) are defined here as pure functions and
left for the cloud entry to call against real model outputs.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Detection / IoU primitives -----------------------------------------------


def iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def precision_recall(
    *, true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float]:
    p_denom = true_positives + false_positives
    r_denom = true_positives + false_negatives
    precision = true_positives / p_denom if p_denom else 0.0
    recall = true_positives / r_denom if r_denom else 0.0
    return precision, recall


def macro_f1(per_class_pr: Mapping[str, tuple[float, float]]) -> float:
    if not per_class_pr:
        return 0.0
    f1s = []
    for precision, recall in per_class_pr.values():
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s)


# Per-attribute classification ---------------------------------------------


def per_class_accuracy(
    truths: Sequence[str], predictions: Sequence[str]
) -> dict[str, float]:
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must align")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for truth, pred in zip(truths, predictions):
        counts[truth] += 1
        if truth == pred:
            correct[truth] += 1
    return {
        cls: (correct[cls] / counts[cls]) if counts[cls] else 0.0
        for cls in sorted(counts)
    }


def confusion_matrix(
    truths: Sequence[str], predictions: Sequence[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in zip(truths, predictions):
        matrix[truth][pred] += 1
    return {k: dict(v) for k, v in matrix.items()}


# Calibration / abstention --------------------------------------------------


def expected_calibration_error(
    *,
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Standard ECE: weighted mean of |confidence - accuracy| per bin."""
    if not confidences:
        return 0.0
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must align")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((float(conf), int(ok)))

    total = float(len(confidences))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bucket_conf = sum(c for c, _ in bucket) / len(bucket)
        bucket_acc = sum(o for _, o in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(bucket_conf - bucket_acc)
    return ece


def abstention_aware_correctness(
    *,
    truths: Sequence[str],
    predictions: Sequence[str | None],
) -> dict[str, float]:
    """Score correctness while treating ``None`` predictions as abstentions.

    Abstentions are not counted as wrong; they reduce coverage but not
    accuracy. ``coverage`` is the fraction of items with a non-None
    prediction, ``correct_when_predicting`` is accuracy on those items, and
    ``selective_score`` is coverage * correct_when_predicting.
    """
    n = len(truths)
    if n == 0:
        return {"coverage": 0.0, "correct_when_predicting": 0.0, "selective_score": 0.0}
    predicted = [(t, p) for t, p in zip(truths, predictions) if p is not None]
    coverage = len(predicted) / n
    if not predicted:
        return {"coverage": 0.0, "correct_when_predicting": 0.0, "selective_score": 0.0}
    correct = sum(1 for t, p in predicted if t == p)
    correct_when_predicting = correct / len(predicted)
    return {
        "coverage": coverage,
        "correct_when_predicting": correct_when_predicting,
        "selective_score": coverage * correct_when_predicting,
    }


# Aggregated report ---------------------------------------------------------


@dataclass
class EvalReport:
    generated_at: str
    detector_metrics: dict[str, Any] = field(default_factory=dict)
    classifier_metrics: dict[str, Any] = field(default_factory=dict)
    abstention_metrics: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    model_size_mb: dict[str, float] = field(default_factory=dict)
    dataset_lock_path: str | None = None
    model_record_paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "detector": self.detector_metrics,
            "classifier": self.classifier_metrics,
            "abstention": self.abstention_metrics,
            "latency_ms": self.latency_ms,
            "model_size_mb": self.model_size_mb,
            "dataset_lock_path": self.dataset_lock_path,
            "model_record_paths": self.model_record_paths,
            "notes": self.notes,
        }


def write_report(report: EvalReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_card_path = out_dir / "model_card.md"
    model_card_path.write_text(render_model_card(report), encoding="utf-8")
    return {"metrics": metrics_path, "model_card": model_card_path}


def render_model_card(report: EvalReport) -> str:
    lines = ["# Model Card", "", f"- Generated: `{report.generated_at}`"]
    if report.dataset_lock_path:
        lines.append(f"- Dataset lock: `{report.dataset_lock_path}`")
    if report.model_record_paths:
        lines.append("- Model records:")
        for kind, path in report.model_record_paths.items():
            lines.append(f"  - {kind}: `{path}`")
    lines.append("")
    if report.detector_metrics:
        lines.append("## Detector")
        for key, value in sorted(report.detector_metrics.items()):
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if report.classifier_metrics:
        lines.append("## Classifier (per head)")
        for head, head_metrics in sorted(report.classifier_metrics.items()):
            lines.append(f"### {head}")
            for key, value in sorted(head_metrics.items()):
                lines.append(f"- {key}: `{value}`")
            lines.append("")
    if report.abstention_metrics:
        lines.append("## Abstention")
        for key, value in sorted(report.abstention_metrics.items()):
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if report.latency_ms:
        lines.append("## Latency (ms)")
        for key, value in sorted(report.latency_ms.items()):
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if report.notes:
        lines.append("## Notes")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
