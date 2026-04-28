"""
Watch-and-ingest daemon.

Polls a directory tree for new uploads from the relay server. Each upload
is expected to be a subdirectory containing image frames + a manifest.json:

    incoming/
      <upload_id>/                ← created atomically by relay
        manifest.json             ← {"claimed_class": "2.4mm-M", "device": ..., ...}
        frame_001.jpg
        frame_002.jpg
        ...
        .ready                    ← sentinel — relay touches this LAST,
                                    after all frames + manifest are in place

The .ready sentinel is the daemon's "this upload is fully written" signal.
We only process directories that contain it; without it we assume the relay
is still writing and skip on this poll cycle.

After processing, the daemon writes a `_processed.json` sidecar alongside
the upload (so re-runs skip already-handled directories) and logs the
decision.

Usage:
    python -m scripts.ingestion_daemon \\
        --incoming-dir incoming \\
        --labeled-root data/labeled/embedder \\
        --quarantine-root data/quarantine \\
        --classifier-dir models/connector_classifier \\
        --interval 5

Hit Ctrl-C to stop. Designed to run forever as a long-lived process.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ingest.process_upload import IngestionConfig, process_upload


READY_SENTINEL = ".ready"
PROCESSED_SIDECAR = "_processed.json"
MANIFEST_FILENAME = "manifest.json"


log = logging.getLogger("ingestion_daemon")


def _load_predictor(classifier_dir: Path | None) -> EnsemblePredictor:
    """
    Load the ensemble predictor, gracefully falling back to measurement-only
    mode when no trained classifier exists yet (first-boot / pre-training state).
    """
    if classifier_dir is None:
        return EnsemblePredictor(classifier=None)
    try:
        return EnsemblePredictor.load(classifier_dir)
    except FileNotFoundError:
        log.info(
            "no classifier weights at %s yet — running measurement-only "
            "(restart the daemon after the first retrain to pick them up)",
            classifier_dir,
        )
        return EnsemblePredictor(classifier=None)
    except Exception as e:
        log.warning(
            "classifier load failed (%s); running measurement-only", e,
        )
        return EnsemblePredictor(classifier=None)


def _is_processed(upload_dir: Path) -> bool:
    return (upload_dir / PROCESSED_SIDECAR).exists()


def _is_ready(upload_dir: Path) -> bool:
    return (upload_dir / READY_SENTINEL).exists()


def _read_manifest(upload_dir: Path) -> dict:
    p = upload_dir / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"upload missing {MANIFEST_FILENAME}: {upload_dir}")
    return json.loads(p.read_text())


def _list_pending(incoming_dir: Path) -> list[Path]:
    """Return upload subdirectories that are ready and not yet processed."""
    if not incoming_dir.is_dir():
        return []
    pending = []
    for child in sorted(incoming_dir.iterdir()):
        if not child.is_dir():
            continue
        if not _is_ready(child):
            continue
        if _is_processed(child):
            continue
        pending.append(child)
    return pending


def _write_processed(upload_dir: Path, decision_blob: dict) -> None:
    (upload_dir / PROCESSED_SIDECAR).write_text(json.dumps(decision_blob, indent=2))


def process_one(
    upload_dir: Path,
    predictor: EnsemblePredictor,
    labeled_root: Path,
    quarantine_root: Path,
    config: IngestionConfig,
) -> dict:
    """Process a single ready upload directory. Returns a small dict for logging."""
    try:
        manifest = _read_manifest(upload_dir)
    except FileNotFoundError as e:
        log.warning("skipping upload (%s)", e)
        return {"upload_id": upload_dir.name, "decision": "error", "reason": str(e)}

    claimed_class = manifest.get("claimed_class")
    if not claimed_class:
        log.warning("upload %s missing claimed_class", upload_dir.name)
        return {"upload_id": upload_dir.name, "decision": "error",
                "reason": "manifest missing claimed_class"}

    decision = process_upload(
        upload_dir=upload_dir,
        claimed_class=claimed_class,
        predictor=predictor,
        labeled_root=labeled_root,
        quarantine_root=quarantine_root,
        upload_id=upload_dir.name,
        config=config,
    )
    blob = {
        "upload_id": decision.upload_id,
        "decision": decision.decision,
        "claimed_class": decision.claimed_class,
        "ensemble_class": decision.ensemble_class,
        "ensemble_confidence": decision.ensemble_confidence,
        "destination": str(decision.destination) if decision.destination else None,
        "reason": decision.reason,
        "n_frames_total": decision.n_frames_total,
        "n_frames_used": decision.n_frames_used,
    }
    _write_processed(upload_dir, blob)
    log.info(
        "processed %s: %s (claimed=%s, ensemble=%s conf=%.2f)",
        decision.upload_id, decision.decision, decision.claimed_class,
        decision.ensemble_class, decision.ensemble_confidence,
    )
    return blob


def run_loop(
    incoming_dir: Path,
    labeled_root: Path,
    quarantine_root: Path,
    classifier_dir: Path | None,
    interval_seconds: float,
    config: IngestionConfig,
    once: bool = False,
) -> None:
    """Poll incoming_dir on an interval, processing each ready+unprocessed upload."""
    predictor = _load_predictor(classifier_dir)
    log.info(
        "ingestion daemon up. watching %s, classifier=%s, interval=%.1fs",
        incoming_dir,
        "loaded" if predictor.classifier is not None else "measurement-only",
        interval_seconds,
    )
    while True:
        try:
            for upload_dir in _list_pending(incoming_dir):
                process_one(
                    upload_dir=upload_dir,
                    predictor=predictor,
                    labeled_root=labeled_root,
                    quarantine_root=quarantine_root,
                    config=config,
                )
        except Exception:
            log.exception("ingestion loop iteration failed; will retry")
        if once:
            break
        time.sleep(interval_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming-dir", type=Path, required=True)
    ap.add_argument("--labeled-root", type=Path, required=True)
    ap.add_argument("--quarantine-root", type=Path, required=True)
    ap.add_argument("--classifier-dir", type=Path, default=None,
                    help="Path to trained model dir; omit for measurement-only mode.")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="Polling interval in seconds.")
    ap.add_argument("--once", action="store_true",
                    help="Process current pending uploads, then exit (don't loop).")
    ap.add_argument("--approve-confidence", type=float, default=0.70)
    ap.add_argument("--approve-agree-fraction", type=float, default=0.50)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = IngestionConfig(
        approve_confidence=args.approve_confidence,
        approve_agree_fraction=args.approve_agree_fraction,
    )

    run_loop(
        incoming_dir=args.incoming_dir,
        labeled_root=args.labeled_root,
        quarantine_root=args.quarantine_root,
        classifier_dir=args.classifier_dir,
        interval_seconds=args.interval,
        config=config,
        once=args.once,
    )


if __name__ == "__main__":
    main()
