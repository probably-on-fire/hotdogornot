from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfconnectorai.eval.evaluate_all import (
    aggregate_latency,
    build_eval_report,
    evaluate_classifier_heads,
    evaluate_detector,
    main,
    read_predictions,
)
from rfconnectorai.eval.reports import (
    abstention_aware_correctness,
    confusion_matrix,
    expected_calibration_error,
    iou,
    macro_f1,
    per_class_accuracy,
    precision_recall,
    render_model_card,
    write_report,
    EvalReport,
    now_iso,
)


def test_iou_disjoint_returns_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_identical_returns_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_partial_overlap():
    val = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert val == pytest.approx(25 / 175)


def test_precision_recall_zero_division_safe():
    p, r = precision_recall(true_positives=0, false_positives=0, false_negatives=0)
    assert p == 0.0 and r == 0.0


def test_macro_f1_empty():
    assert macro_f1({}) == 0.0


def test_per_class_accuracy_aligns():
    accuracy = per_class_accuracy(["A", "A", "B", "B"], ["A", "B", "B", "B"])
    assert accuracy == {"A": 0.5, "B": 1.0}


def test_confusion_matrix_counts():
    matrix = confusion_matrix(["A", "A", "B"], ["A", "B", "B"])
    assert matrix == {"A": {"A": 1, "B": 1}, "B": {"B": 1}}


def test_expected_calibration_error_perfect_calibration():
    confidences = [0.5, 0.5, 0.5, 0.5]
    correct = [1, 0, 1, 0]
    assert expected_calibration_error(confidences=confidences, correct=correct) == pytest.approx(0.0, abs=1e-6)


def test_abstention_aware_correctness_handles_none():
    metrics = abstention_aware_correctness(
        truths=["A", "A", "B", "B"],
        predictions=["A", None, "B", None],
    )
    assert metrics["coverage"] == 0.5
    assert metrics["correct_when_predicting"] == 1.0
    assert metrics["selective_score"] == 0.5


def _row(
    *,
    truth: dict, pred: dict, conf: dict, abstain: dict | None = None,
    iou_value: float = 0.9, latency: dict | None = None,
) -> dict:
    return {
        "ground_truth": truth,
        "prediction": pred,
        "confidences": conf,
        "abstain": abstain or {},
        "detection_iou": iou_value,
        "latency_ms": latency or {"preprocess": 1, "detector": 2, "classifier": 3, "total": 6},
    }


def test_evaluate_classifier_heads_with_full_match():
    rows = [
        _row(
            truth={"family": "SMA", "polarity": "standard"},
            pred={"family": "SMA", "polarity": "standard"},
            conf={"family": 0.9, "polarity": 0.85},
        ),
        _row(
            truth={"family": "BNC", "polarity": "not_applicable"},
            pred={"family": "SMA", "polarity": "not_applicable"},
            conf={"family": 0.6, "polarity": 0.7},
        ),
    ]
    metrics = evaluate_classifier_heads(rows)
    assert metrics["family"]["support"] == 2
    assert metrics["family"]["accuracy"] == 0.5
    assert metrics["polarity"]["accuracy"] == 1.0


def test_evaluate_classifier_heads_skips_low_iou():
    rows = [_row(
        truth={"family": "SMA"}, pred={"family": "SMA"},
        conf={"family": 0.9}, iou_value=0.1,
    )]
    metrics = evaluate_classifier_heads(rows, iou_threshold=0.5)
    assert "family" not in metrics


def test_evaluate_detector_recall_thresholds():
    rows = [
        _row(truth={}, pred={}, conf={}, iou_value=0.4),
        _row(truth={}, pred={}, conf={}, iou_value=0.6),
        _row(truth={}, pred={}, conf={}, iou_value=0.8),
    ]
    metrics = evaluate_detector(rows)
    assert metrics["support"] == 3
    assert metrics["recall@iou0.5"] == pytest.approx(2 / 3)
    assert metrics["recall@iou0.75"] == pytest.approx(1 / 3)


def test_aggregate_latency_means():
    rows = [
        _row(truth={}, pred={}, conf={}, latency={"total": 50, "detector": 30}),
        _row(truth={}, pred={}, conf={}, latency={"total": 100, "detector": 40}),
    ]
    latency = aggregate_latency(rows)
    assert latency["total"] == 75
    assert latency["detector"] == 35


def test_build_eval_report_packages_everything(tmp_path: Path):
    rows = [
        _row(
            truth={"family": "SMA", "polarity": "standard", "side_a_gender": "male_pin"},
            pred={"family": "SMA", "polarity": "standard", "side_a_gender": "male_pin"},
            conf={"family": 0.95, "polarity": 0.9, "side_a_gender": 0.92},
            abstain={"side_b_gender": True},
        ),
    ]
    report = build_eval_report(
        rows=rows,
        dataset_lock_path=tmp_path / "dataset.lock.json",
        model_record_paths={"detector": tmp_path / "det.json", "classifier": tmp_path / "cls.json"},
        notes=["smoke run"],
    )
    paths = write_report(report, tmp_path / "out")
    assert paths["metrics"].exists()
    assert paths["model_card"].exists()
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert payload["dataset_lock_path"].endswith("dataset.lock.json")
    assert payload["classifier"]["family"]["accuracy"] == 1.0


def test_main_cli_round_trip(tmp_path: Path):
    rows = [
        _row(
            truth={"family": "SMA"}, pred={"family": "SMA"},
            conf={"family": 0.9},
        ),
    ]
    pred_path = tmp_path / "preds.jsonl"
    pred_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rc = main([
        "--predictions", str(pred_path),
        "--out", str(tmp_path / "out"),
    ])
    assert rc == 0
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "model_card.md").exists()


def test_render_model_card_handles_empty_report():
    card = render_model_card(EvalReport(generated_at=now_iso()))
    assert "# Model Card" in card


def test_read_predictions_rejects_invalid_json(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_predictions(bad)
