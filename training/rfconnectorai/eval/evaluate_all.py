"""End-to-end evaluation harness.

Reads a predictions JSONL produced by a cloud run, joins it against the
test split's ground-truth labels, and writes an :class:`EvalReport` plus
its rendered model card.

This module is *I/O-light*: detector inference, classifier inference, and
the actual model loading happen in the cloud and produce a predictions
JSONL on disk. The harness itself is pure Python so it stays unit-testable
on the local CPU PC.

Predictions JSONL row schema::

    {
      "instance_id": "i_001",
      "split": "test",
      "ground_truth": {
        "family": "SMA", "polarity": "standard",
        "side_a_gender": "male_pin", ...
      },
      "prediction": {
        "family": "SMA", "polarity": "standard",
        "side_a_gender": "male_pin", ...
      },
      "confidences": {"family": 0.96, "polarity": 0.92, ...},
      "abstain": {"polarity": false, "side_b_gender": true},
      "detection_iou": 0.82,
      "latency_ms": {"detector": 31, "classifier": 18, "total": 74}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from rfconnectorai.classifier.label_encoding import HEAD_ORDER
from rfconnectorai.eval.reports import (
    EvalReport,
    abstention_aware_correctness,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    now_iso,
    per_class_accuracy,
    precision_recall,
    write_report,
)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} invalid JSON: {exc}"
                ) from exc
    return rows


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_classifier_heads(
    rows: Iterable[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Per-head metrics joined on detection IoU.

    Rows whose ``detection_iou < iou_threshold`` are dropped from
    classifier metrics (the detection failed, so the classifier choice is
    moot for accuracy purposes).
    """
    by_head_truth: dict[str, list[str]] = {h: [] for h in HEAD_ORDER}
    by_head_pred: dict[str, list[str]] = {h: [] for h in HEAD_ORDER}
    by_head_pred_with_abstain: dict[str, list[str | None]] = {h: [] for h in HEAD_ORDER}
    by_head_conf: dict[str, list[float]] = {h: [] for h in HEAD_ORDER}
    by_head_correct: dict[str, list[int]] = {h: [] for h in HEAD_ORDER}

    for row in rows:
        if _coerce_float(row.get("detection_iou", 1.0)) < iou_threshold:
            continue
        truth = row.get("ground_truth", {})
        prediction = row.get("prediction", {})
        confidences = row.get("confidences", {})
        abstain = row.get("abstain", {})
        for head in HEAD_ORDER:
            t = truth.get(head)
            p = prediction.get(head)
            if t is None:
                continue
            by_head_truth[head].append(str(t))
            pred_value = str(p) if p is not None else ""
            by_head_pred[head].append(pred_value)
            by_head_pred_with_abstain[head].append(
                None if abstain.get(head) else (str(p) if p is not None else None)
            )
            if p is not None:
                by_head_conf[head].append(_coerce_float(confidences.get(head, 0.0)))
                by_head_correct[head].append(1 if str(p) == str(t) else 0)

    metrics: dict[str, dict[str, Any]] = {}
    for head in HEAD_ORDER:
        truths = by_head_truth[head]
        preds = by_head_pred[head]
        if not truths:
            continue
        accuracy = sum(1 for t, p in zip(truths, preds) if t == p) / len(truths)
        per_class = per_class_accuracy(truths, preds)
        f1s = {
            cls: precision_recall(
                true_positives=sum(1 for t, p in zip(truths, preds) if t == cls and p == cls),
                false_positives=sum(1 for t, p in zip(truths, preds) if t != cls and p == cls),
                false_negatives=sum(1 for t, p in zip(truths, preds) if t == cls and p != cls),
            )
            for cls in set(truths) | set(preds)
        }
        ece = expected_calibration_error(
            confidences=by_head_conf[head], correct=by_head_correct[head]
        )
        metrics[head] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1(f1s),
            "per_class_accuracy": per_class,
            "confusion": confusion_matrix(truths, preds),
            "expected_calibration_error": ece,
            "abstention": abstention_aware_correctness(
                truths=truths,
                predictions=by_head_pred_with_abstain[head],
            ),
            "support": len(truths),
        }
    return metrics


def evaluate_detector(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"support": 0}
    ious = [_coerce_float(r.get("detection_iou", 0.0)) for r in rows]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    iou_50 = sum(1 for i in ious if i >= 0.5) / len(ious)
    iou_75 = sum(1 for i in ious if i >= 0.75) / len(ious)
    return {
        "support": len(rows),
        "mean_iou": mean_iou,
        "recall@iou0.5": iou_50,
        "recall@iou0.75": iou_75,
    }


def aggregate_latency(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    keys = ("preprocess", "detector", "classifier", "total")
    aggregate: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        latency = row.get("latency_ms") or {}
        for key in keys:
            if key in latency:
                aggregate[key].append(_coerce_float(latency[key]))
    return {k: (sum(v) / len(v)) if v else 0.0 for k, v in aggregate.items()}


def build_eval_report(
    *,
    rows: Iterable[dict[str, Any]],
    dataset_lock_path: Path | None = None,
    model_record_paths: dict[str, Path] | None = None,
    notes: Iterable[str] = (),
) -> EvalReport:
    rows = list(rows)
    detector_metrics = evaluate_detector(rows)
    classifier_metrics = evaluate_classifier_heads(rows)
    abstention = {
        head: classifier_metrics[head]["abstention"]
        for head in classifier_metrics
        if "abstention" in classifier_metrics[head]
    }
    latency = aggregate_latency(rows)
    return EvalReport(
        generated_at=now_iso(),
        detector_metrics=detector_metrics,
        classifier_metrics=classifier_metrics,
        abstention_metrics=abstention,
        latency_ms=latency,
        dataset_lock_path=str(dataset_lock_path) if dataset_lock_path else None,
        model_record_paths={k: str(v) for k, v in (model_record_paths or {}).items()},
        notes=list(notes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connector evaluation harness")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions JSONL path")
    parser.add_argument("--out", type=Path, required=True, help="report output dir")
    parser.add_argument("--dataset-lock", type=Path, default=None)
    parser.add_argument(
        "--detector-record",
        type=Path,
        default=None,
        help="Path to detector model_record.json",
    )
    parser.add_argument(
        "--classifier-record",
        type=Path,
        default=None,
        help="Path to classifier model_record.json",
    )
    args = parser.parse_args(argv)

    rows = read_predictions(args.predictions)
    record_paths: dict[str, Path] = {}
    if args.detector_record:
        record_paths["detector"] = args.detector_record
    if args.classifier_record:
        record_paths["classifier"] = args.classifier_record

    report = build_eval_report(
        rows=rows,
        dataset_lock_path=args.dataset_lock,
        model_record_paths=record_paths,
    )
    paths = write_report(report, args.out)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
