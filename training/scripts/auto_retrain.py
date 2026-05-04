"""
Auto-retrain script — runs the classifier trainer when enough new approved
labeled images have accumulated since the last published model.

Designed to be invoked from cron / a scheduled task:

    # nightly at 02:00
    0 2 * * *   /path/to/.venv/bin/python -m scripts.auto_retrain ...

Logic:

  1. Count current images in `data/labeled/embedder/<CLASS>/` per class.
  2. Read `models/connector_classifier/version.json` — has the n_train_samples
     used at last training.
  3. If (current total - last total) >= --min-new-samples, retrain.
     Otherwise log "no retrain needed" and exit 0.
  4. After training, `train()` already calls `bump_version`, so the
     manifest auto-refreshes. Relay server's GET /model/version then
     advertises the new version on the next poll.

Exit codes:
  0  — retrained (or skipped because no new data yet)
  1  — training failed
  2  — invalid configuration (no labeled folders, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rfconnectorai.classifier.dataset import ConnectorFolderDataset
from rfconnectorai.classifier.train import TrainConfig, train


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

# A class needs at least this many training samples to be included in
# the model's output head. Below this we drop the class entirely rather
# than ship it as a softmax slot that can never be predicted correctly
# but still sucks gradient via label smoothing's uniform mass.
MIN_SAMPLES_PER_CLASS = 5


def _populated_classes(data_dir: Path, candidates: list[str],
                       min_samples: int) -> list[str]:
    """Filter `candidates` to classes whose folder under `data_dir` has at
    least `min_samples` images. Empty / undersized classes get dropped
    entirely from the trained head — their presence as zero-data softmax
    slots actively hurts inference (see classifier_journey.md)."""
    out = []
    skipped = []
    for c in candidates:
        d = data_dir / c
        n = sum(1 for p in d.glob("*.[jpJP]*[gG]")) if d.is_dir() else 0
        if n >= min_samples:
            out.append(c)
        else:
            skipped.append((c, n))
    if skipped:
        log.warning(
            "dropping %d undersized class(es) from training head: %s",
            len(skipped),
            ", ".join(f"{c}(n={n})" for c, n in skipped),
        )
    return out

log = logging.getLogger("auto_retrain")


def _last_training_size(model_dir: Path) -> int | None:
    """Read n_train_samples from the latest version.json. None if no prior train."""
    p = model_dir / "version.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text()).get("n_train_samples", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _current_dataset_size(data_dir: Path, class_names: list[str]) -> int:
    ds = ConnectorFolderDataset(data_dir, class_names=class_names)
    return len(ds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Labeled-data root, e.g. data/labeled/embedder/")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="Where the trained model lives, e.g. models/connector_classifier/")
    ap.add_argument("--min-new-samples", type=int, default=20,
                    help="Skip retrain unless at least this many new samples have been added.")
    ap.add_argument("--min-samples-per-class", type=int,
                    default=MIN_SAMPLES_PER_CLASS,
                    help="Drop classes with fewer than this many images "
                         "from the trained head (zero-data classes leak "
                         "label-smoothing gradient and produce confident "
                         "wrong predictions on background frames).")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--force", action="store_true",
                    help="Retrain regardless of new-sample count.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.data_dir.is_dir():
        log.error("data dir %s does not exist", args.data_dir)
        return 2

    classes = _populated_classes(args.data_dir, CANONICAL_CLASSES,
                                 args.min_samples_per_class)
    if len(classes) < 2:
        log.error("only %d class(es) have enough samples; need at least 2 "
                  "to train a classifier", len(classes))
        return 2

    current = _current_dataset_size(args.data_dir, classes)
    last = _last_training_size(args.model_dir) or 0
    delta = current - last

    log.info(
        "dataset size: current=%d (across %d classes), at_last_train=%d, "
        "delta=%d, threshold=%d",
        current, len(classes), last, delta, args.min_new_samples,
    )

    if not args.force and delta < args.min_new_samples:
        log.info("not enough new data to justify retrain — skipping")
        return 0

    if current < 16:
        log.error("only %d total labeled samples; need at least 16 to train", current)
        return 2

    config = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.model_dir,
        class_names=classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_fraction=args.val_fraction,
    )
    try:
        metrics = train(config)
    except Exception:
        log.exception("training failed")
        return 1

    last_epoch = metrics["history"][-1] if metrics.get("history") else {}
    log.info(
        "retrain complete. final val_acc=%.3f, n_train_samples=%d",
        last_epoch.get("val_acc", 0.0), current,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
