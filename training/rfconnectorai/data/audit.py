"""Dataset audit utility.

Walks one or more dataset roots without modifying images and produces a
deterministic audit report covering counts, dimensions, duplicates,
synthetic-vs-real provenance heuristics, multi-connector heuristics, and
holdout/train leakage risk. Intended to run on the local PC for repo work
and in cloud for full-volume runs.

Outputs:

- Markdown summary at the user-supplied path.
- Companion JSON file with the same stem (`<path>.json`) for tooling.

Usage::

    python -m rfconnectorai.data.audit \\
        --data-dir training/data \\
        --out docs/DATASET_AUDIT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - Pillow ships in the training extras.
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

# Heuristics for "this path looks synthetic" vs "this path looks like a real
# phone/product photo". These are intentionally conservative; the audit is a
# signal, not ground truth.
SYNTHETIC_PATH_HINTS = (
    "synthetic",
    "synth",
    "render",
    "renders",
    "synthetic_faces",
    "synthetic_angled",
    "cad",
)
HOLDOUT_PATH_HINTS = ("test_holdout", "holdout")
REFERENCE_PATH_HINTS = ("reference",)

# Low-sample threshold for "this class likely needs more data" warning.
DEFAULT_MIN_PER_CLASS = 30


@dataclass
class FileSummary:
    path: str
    size_bytes: int
    extension: str
    class_label: str | None
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    is_unreadable: bool = False
    error: str | None = None
    is_synthetic_hint: bool = False
    is_holdout_hint: bool = False
    is_reference_hint: bool = False
    multi_connector_hint: bool = False


@dataclass
class RootSummary:
    root: str
    image_count: int = 0
    video_count: int = 0
    other_count: int = 0
    unreadable_count: int = 0
    synthetic_count: int = 0
    holdout_count: int = 0
    reference_count: int = 0
    multi_connector_hint_count: int = 0
    by_extension: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    files: list[FileSummary] = field(default_factory=list)


@dataclass
class AuditReport:
    generated_at: str
    data_dir: str
    roots: list[RootSummary]
    duplicate_groups: list[list[str]] = field(default_factory=list)
    leakage_groups: list[dict[str, Any]] = field(default_factory=list)
    missing_taxonomy_classes: list[str] = field(default_factory=list)
    low_sample_classes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Drop the per-file blobs from the JSON summary; they are only useful
        # for debugging and bloat the report otherwise.
        for root in payload["roots"]:
            root.pop("files", None)
        return payload


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through sha256."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def classify_path(rel_path: Path) -> dict[str, bool]:
    parts = {p.lower() for p in rel_path.parts}
    return {
        "synthetic": any(hint in parts for hint in SYNTHETIC_PATH_HINTS),
        "holdout": any(hint in parts for hint in HOLDOUT_PATH_HINTS),
        "reference": any(hint in parts for hint in REFERENCE_PATH_HINTS),
    }


def infer_class_label(rel_path: Path) -> str | None:
    """Best-effort class label from folder structure.

    Convention in this repo is `<root>/<...>/<class>/file.jpg`. We pick the
    immediate parent of the file as the class label. Audit consumers should
    treat this as a *folder-derived* label and not as a verified label.
    """
    if rel_path.parent == rel_path.parent.parent:
        return None
    parent = rel_path.parent.name
    if not parent or parent.lower() in {"images", "labeled", "embedder"}:
        return None
    return parent


def detect_multi_connector_hint(rel_path: Path) -> bool:
    name = rel_path.name.lower()
    hints = ("group", "set", "multi", "panel", "stack", "tray")
    return any(h in name for h in hints)


def read_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    if Image is None:
        return None, None, "Pillow not installed"
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], None
    except (UnidentifiedImageError, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def audit_root(root: Path, *, base_dir: Path, hash_files: bool = True) -> RootSummary:
    summary = RootSummary(root=str(root))
    extension_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()

    for path in iter_files(root):
        ext = path.suffix.lower()
        try:
            rel = path.relative_to(base_dir)
        except ValueError:
            rel = path

        is_image = ext in IMAGE_EXTS
        is_video = ext in VIDEO_EXTS

        hints = classify_path(rel)
        class_label = infer_class_label(rel) if is_image else None
        multi_hint = detect_multi_connector_hint(rel) if is_image else False

        size = path.stat().st_size
        width: int | None = None
        height: int | None = None
        sha: str | None = None
        unreadable = False
        error: str | None = None

        if is_image:
            width, height, error = read_dimensions(path)
            unreadable = error is not None
            if not unreadable and hash_files:
                try:
                    sha = hash_file(path)
                except OSError as exc:
                    unreadable = True
                    error = f"{type(exc).__name__}: {exc}"

        file_summary = FileSummary(
            path=str(rel),
            size_bytes=size,
            extension=ext,
            class_label=class_label,
            width=width,
            height=height,
            sha256=sha,
            is_unreadable=unreadable,
            error=error,
            is_synthetic_hint=hints["synthetic"],
            is_holdout_hint=hints["holdout"],
            is_reference_hint=hints["reference"],
            multi_connector_hint=multi_hint,
        )
        summary.files.append(file_summary)
        extension_counter[ext] += 1

        if is_image:
            summary.image_count += 1
            if class_label:
                class_counter[class_label] += 1
            if hints["synthetic"]:
                summary.synthetic_count += 1
            if hints["holdout"]:
                summary.holdout_count += 1
            if hints["reference"]:
                summary.reference_count += 1
            if multi_hint:
                summary.multi_connector_hint_count += 1
            if unreadable:
                summary.unreadable_count += 1
        elif is_video:
            summary.video_count += 1
        else:
            summary.other_count += 1

    summary.by_extension = dict(sorted(extension_counter.items()))
    summary.by_class = dict(sorted(class_counter.items()))
    return summary


def find_duplicate_groups(roots: list[RootSummary]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for root in roots:
        for file in root.files:
            if file.sha256:
                by_hash[file.sha256].append(file.path)
    return [sorted(paths) for paths in by_hash.values() if len(paths) > 1]


def find_leakage_groups(roots: list[RootSummary]) -> list[dict[str, Any]]:
    """A duplicate that crosses a holdout boundary is a hard leakage signal."""
    by_hash: dict[str, list[FileSummary]] = defaultdict(list)
    for root in roots:
        for file in root.files:
            if file.sha256:
                by_hash[file.sha256].append(file)

    leakage: list[dict[str, Any]] = []
    for sha, files in by_hash.items():
        if len(files) < 2:
            continue
        any_holdout = any(f.is_holdout_hint for f in files)
        any_train = any(not f.is_holdout_hint for f in files)
        if any_holdout and any_train:
            leakage.append(
                {
                    "sha256": sha,
                    "paths": sorted(f.path for f in files),
                    "kind": "holdout_train_overlap",
                }
            )
    return leakage


def find_missing_taxonomy_classes(
    roots: list[RootSummary], taxonomy_ids: Iterable[str]
) -> list[str]:
    seen = set()
    for root in roots:
        for class_label in root.by_class:
            seen.add(class_label.lower())
    expected = {tid.lower() for tid in taxonomy_ids}
    return sorted(expected - seen)


def find_low_sample_classes(
    roots: list[RootSummary], min_per_class: int
) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    for root in roots:
        totals.update(root.by_class)
    return [
        {"class": cls, "count": count, "min": min_per_class}
        for cls, count in sorted(totals.items())
        if count < min_per_class
    ]


def build_report(
    *,
    data_dir: Path,
    roots: list[Path],
    taxonomy_ids: Iterable[str] | None = None,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    hash_files: bool = True,
    now: datetime | None = None,
) -> AuditReport:
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    root_summaries: list[RootSummary] = [
        audit_root(root, base_dir=data_dir, hash_files=hash_files) for root in roots
    ]

    duplicates = find_duplicate_groups(root_summaries)
    leakage = find_leakage_groups(root_summaries)
    missing = (
        find_missing_taxonomy_classes(root_summaries, taxonomy_ids)
        if taxonomy_ids is not None
        else []
    )
    low = find_low_sample_classes(root_summaries, min_per_class)

    notes: list[str] = []
    if any(r.holdout_count == 0 for r in root_summaries):
        notes.append(
            "At least one audited root has no images flagged as holdout; "
            "verify the holdout root is included before training."
        )
    if not any(r.image_count for r in root_summaries):
        notes.append("No images were found in any audited root.")

    return AuditReport(
        generated_at=timestamp,
        data_dir=str(data_dir),
        roots=root_summaries,
        duplicate_groups=duplicates,
        leakage_groups=leakage,
        missing_taxonomy_classes=missing,
        low_sample_classes=low,
        notes=notes,
    )


def render_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("# Dataset Audit")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at}`")
    lines.append(f"- Data dir: `{report.data_dir}`")
    lines.append(f"- Roots audited: {len(report.roots)}")
    lines.append("")

    lines.append("## Summary By Root")
    lines.append("")
    lines.append("| Root | Images | Videos | Synthetic | Holdout | Reference | Multi-conn hint | Unreadable |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for root in report.roots:
        lines.append(
            f"| `{root.root}` | {root.image_count} | {root.video_count} | "
            f"{root.synthetic_count} | {root.holdout_count} | {root.reference_count} | "
            f"{root.multi_connector_hint_count} | {root.unreadable_count} |"
        )
    lines.append("")

    if report.duplicate_groups:
        lines.append("## Duplicate Groups (sha256)")
        lines.append("")
        for group in report.duplicate_groups[:50]:
            lines.append("- " + ", ".join(f"`{p}`" for p in group))
        if len(report.duplicate_groups) > 50:
            lines.append(f"- ... and {len(report.duplicate_groups) - 50} more groups")
        lines.append("")

    if report.leakage_groups:
        lines.append("## Train/Holdout Leakage")
        lines.append("")
        lines.append("Files with identical hashes appear in both training and holdout roots.")
        for group in report.leakage_groups[:50]:
            lines.append(f"- `{group['kind']}`: " + ", ".join(f"`{p}`" for p in group["paths"]))
        lines.append("")

    if report.missing_taxonomy_classes:
        lines.append("## Missing Taxonomy Classes")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in report.missing_taxonomy_classes))
        lines.append("")

    if report.low_sample_classes:
        lines.append("## Classes Below Minimum Sample Count")
        lines.append("")
        lines.append("| Class | Count | Min |")
        lines.append("|---|---:|---:|")
        for entry in report.low_sample_classes:
            lines.append(f"| `{entry['class']}` | {entry['count']} | {entry['min']} |")
        lines.append("")

    for root in report.roots:
        if not root.by_class:
            continue
        lines.append(f"## Class Distribution: `{root.root}`")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|---|---:|")
        for cls, count in root.by_class.items():
            lines.append(f"| `{cls}` | {count} |")
        lines.append("")

    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: AuditReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _default_roots(data_dir: Path) -> list[Path]:
    """Roots audited by default if --root is not given.

    Matches the layout described in `IMPLEMENTATION_PLAN.md` section 6.1.
    """
    candidates = [
        data_dir / "Images",
        data_dir / "data" / "labeled",
        data_dir / "data" / "test_holdout",
        data_dir / "data" / "reference",
        data_dir / "data" / "videos",
    ]
    return [c for c in candidates if c.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RF connector dataset audit")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Base directory used as the relative root for paths in the report.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Explicit root to audit. May be repeated. Defaults to the standard layout.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the markdown report. JSON written alongside.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=DEFAULT_MIN_PER_CLASS,
        help=f"Threshold for low-sample warnings (default {DEFAULT_MIN_PER_CLASS}).",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip sha256 hashing (faster on large datasets, disables duplicate/leakage detection).",
    )
    parser.add_argument(
        "--skip-taxonomy",
        action="store_true",
        help="Do not load the taxonomy for missing-class detection.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    roots = (
        [r.resolve() for r in args.root]
        if args.root is not None
        else _default_roots(data_dir)
    )

    taxonomy_ids: Iterable[str] | None = None
    if not args.skip_taxonomy:
        try:
            from rfconnectorai.schemas.taxonomy import REQUIRED_CONNECTOR_IDS
            taxonomy_ids = REQUIRED_CONNECTOR_IDS
        except Exception as exc:  # pragma: no cover - defensive only
            print(f"warning: taxonomy unavailable ({exc})", file=sys.stderr)
            taxonomy_ids = None

    report = build_report(
        data_dir=data_dir,
        roots=roots,
        taxonomy_ids=taxonomy_ids,
        min_per_class=args.min_per_class,
        hash_files=not args.no_hash,
    )
    write_report(report, args.out)
    print(f"audit report: {args.out}")
    print(f"audit json:   {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
