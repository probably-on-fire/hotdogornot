"""
Auto-trust ingestion for inline-correction uploads.

When the app captures an uncertain frame, the user manually labels it,
the bundle (frames + claimed_class + metadata) lands here. We route each
upload to one of three destinations:

  approve:    user's claim agrees with the ensemble's prediction (or the
              ensemble was empty and the user gave a label) → frames go
              straight to data/labeled/embedder/<CLASS>/
  quarantine: ensemble is confident but disagrees with the user → frames
              go to data/quarantine/<CLASS>/<upload_id>/ for engineer review
  drop:       no real signal anywhere — ensemble Unknown across all frames
              AND no usable images. Logged but not persisted.

Public surface:

    decision = process_upload(upload_dir, claimed_class, predictor,
                              labeled_root, quarantine_root)

Returns an `IngestionDecision` describing what happened and what (if any)
files were copied where. Idempotent — calling twice on the same upload_dir
re-runs the analysis but won't double-write (filenames are dedup'd by md5
the same way the DDG/Google fetchers already do).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.ensemble_averager import (
    AveragedEnsembleResult,
    average_ensemble,
)


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Auto-approve threshold tuning. Conservative on purpose — we'd rather
# quarantine a borderline case than poison the training set. Tighten/loosen
# in `IngestionConfig` per environment.
DEFAULT_APPROVE_CONFIDENCE = 0.70
DEFAULT_APPROVE_AGREE_FRACTION = 0.50   # at least half the frames in "agree"


@dataclass
class IngestionConfig:
    approve_confidence: float = DEFAULT_APPROVE_CONFIDENCE
    approve_agree_fraction: float = DEFAULT_APPROVE_AGREE_FRACTION


@dataclass
class IngestionDecision:
    upload_id: str
    decision: str                          # "approve" | "quarantine" | "drop"
    claimed_class: str
    ensemble_class: str | None
    ensemble_confidence: float
    n_frames_total: int
    n_frames_used: int
    destination: Path | None = None        # where frames landed (None if dropped)
    reason: str = ""
    metadata_written: Path | None = None   # path to a small json sidecar


def _existing_hashes(target_dir: Path) -> set[str]:
    seen: set[str] = set()
    if not target_dir.is_dir():
        return seen
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        try:
            seen.add(hashlib.md5(p.read_bytes()).hexdigest())
        except OSError:
            continue
    return seen


def _list_frames(upload_dir: Path) -> list[Path]:
    return sorted(
        p for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def _next_index(target_dir: Path, prefix: str) -> int:
    if not target_dir.is_dir():
        return 0
    max_idx = -1
    for p in target_dir.iterdir():
        stem = p.stem
        if stem.startswith(f"{prefix}_") and stem[len(prefix) + 1:].isdigit():
            max_idx = max(max_idx, int(stem[len(prefix) + 1:]))
    return max_idx + 1


def _copy_frames(
    frames: list[Path],
    target_dir: Path,
    prefix: str,
) -> list[Path]:
    """Copy frames into target_dir under <prefix>_NNNN.<ext>; dedup by md5."""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_hashes(target_dir)
    index = _next_index(target_dir, prefix)
    copied: list[Path] = []
    for src in frames:
        try:
            data = src.read_bytes()
        except OSError:
            continue
        md5 = hashlib.md5(data).hexdigest()
        if md5 in existing:
            continue
        dst = target_dir / f"{prefix}_{index:04d}{src.suffix.lower()}"
        try:
            shutil.copyfile(src, dst)
        except OSError:
            continue
        existing.add(md5)
        copied.append(dst)
        index += 1
    return copied


def _decide(
    avg: AveragedEnsembleResult,
    claimed_class: str,
    config: IngestionConfig,
) -> tuple[str, str]:
    """Return (decision, reason) for an upload's averaged ensemble result."""
    if avg.n_frames_used == 0:
        return ("drop", "no usable frames — ensemble produced Unknown across all frames")

    agree_count = avg.per_frame_agreement.get("agree", 0)
    agree_fraction = agree_count / avg.n_frames_used if avg.n_frames_used else 0.0

    # Auto-approve when the ensemble agrees with the user's claim AND the
    # ensemble itself is consistent + confident across the video.
    if (
        avg.class_name == claimed_class
        and avg.confidence >= config.approve_confidence
        and agree_fraction >= config.approve_agree_fraction
    ):
        return ("approve",
                f"ensemble agrees ({avg.class_name}, conf={avg.confidence:.2f}, "
                f"agree_fraction={agree_fraction:.2f})")

    # Ensemble disagrees with user's claim → quarantine, regardless of confidence.
    if avg.class_name != claimed_class:
        return ("quarantine",
                f"ensemble says {avg.class_name} (conf={avg.confidence:.2f}), "
                f"user said {claimed_class}")

    # Ensemble agrees on class but confidence too low → quarantine for review.
    return ("quarantine",
            f"ensemble agrees on {avg.class_name} but confidence too low "
            f"(conf={avg.confidence:.2f}, threshold={config.approve_confidence:.2f})")


def process_upload(
    upload_dir: Path,
    claimed_class: str,
    predictor: EnsemblePredictor,
    labeled_root: Path,
    quarantine_root: Path,
    upload_id: str | None = None,
    config: IngestionConfig | None = None,
    require_aruco: bool = False,
) -> IngestionDecision:
    """
    Process one upload directory containing extracted frames.

    `upload_dir` is expected to contain image files (frames extracted by the
    app or by `video_frames.extract_frames`). `claimed_class` is what the
    user said the connector is. `predictor` should already have the latest
    classifier loaded (or be measurement-only).
    """
    config = config or IngestionConfig()
    upload_dir = Path(upload_dir)
    if upload_id is None:
        upload_id = upload_dir.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    frame_paths = _list_frames(upload_dir)
    if not frame_paths:
        return IngestionDecision(
            upload_id=upload_id,
            decision="drop",
            claimed_class=claimed_class,
            ensemble_class=None,
            ensemble_confidence=0.0,
            n_frames_total=0,
            n_frames_used=0,
            reason="no image frames found in upload",
        )

    # Load each frame as RGB.
    frames = []
    for p in frame_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    avg = average_ensemble(frames, predictor, require_aruco=require_aruco)
    decision, reason = _decide(avg, claimed_class, config)

    # Route to the appropriate destination.
    destination: Path | None = None
    if decision == "approve":
        destination = labeled_root / claimed_class
        _copy_frames(frame_paths, destination, prefix="upload")
    elif decision == "quarantine":
        destination = quarantine_root / claimed_class / upload_id
        _copy_frames(frame_paths, destination, prefix="frame")

    metadata_written: Path | None = None
    if destination is not None:
        meta = {
            "upload_id": upload_id,
            "claimed_class": claimed_class,
            "ensemble_class": avg.class_name,
            "ensemble_confidence": avg.confidence,
            "ensemble_per_class_votes": avg.per_class_votes,
            "ensemble_agreement": avg.per_frame_agreement,
            "ensemble_softmax": avg.classifier_probabilities,
            "aperture_mm": avg.aperture_mm,
            "aperture_mm_stddev": avg.aperture_mm_stddev,
            "decision": decision,
            "reason": reason,
            "n_frames_total": avg.n_frames_total,
            "n_frames_used": avg.n_frames_used,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_written = destination / f"_ingest_{upload_id}.json"
        metadata_written.parent.mkdir(parents=True, exist_ok=True)
        metadata_written.write_text(json.dumps(meta, indent=2))

    return IngestionDecision(
        upload_id=upload_id,
        decision=decision,
        claimed_class=claimed_class,
        ensemble_class=avg.class_name,
        ensemble_confidence=avg.confidence,
        n_frames_total=avg.n_frames_total,
        n_frames_used=avg.n_frames_used,
        destination=destination,
        reason=reason,
        metadata_written=metadata_written,
    )
