"""
Video → frames extractor for capture sessions.

When real video of connectors comes in, we want to drop a file like
`SMA-M.mp4` into the data manager and end up with N labeled frames in
`data/labeled/embedder/SMA-M/`. The frame averager and the existing
"Run pipeline" eval button then operate on those frames just like any
other source.

Public surface:

    extract_frames(video_path, out_dir, fps_target=2,
                   max_frames=None, prefix="video") -> list[Path]

Returns the list of saved frame paths. Filenames continue from the highest
existing `<prefix>_NNNN.jpg` index in `out_dir` so repeated runs accumulate.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def _next_index(out_dir: Path, prefix: str) -> int:
    if not out_dir.is_dir():
        return 0
    max_idx = -1
    for p in out_dir.iterdir():
        stem = p.stem
        if stem.startswith(f"{prefix}_") and stem[len(prefix) + 1:].isdigit():
            max_idx = max(max_idx, int(stem[len(prefix) + 1:]))
    return max_idx + 1


def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps_target: float = 2.0,
    max_frames: int | None = None,
    prefix: str = "video",
    jpeg_quality: int = 92,
) -> list[Path]:
    """
    Extract frames from `video_path` at approximately `fps_target` per second.

    `fps_target=2` means we keep ~2 frames per real second of video — for a
    30-second clip you get 60 frames, which is plenty for averaging without
    bloating the dataset with near-duplicates. Set `fps_target` higher for
    short capture sessions, lower for long videos.

    Frames are saved as `<prefix>_NNNN.jpg` continuing from the highest
    existing index in `out_dir`. Returns the list of written paths.

    Raises FileNotFoundError if the video can't be opened.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video (codec issue?): {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # How often (in source frames) do we keep one? Round to at least 1.
    stride = max(1, round(src_fps / fps_target))

    index = _next_index(out_dir, prefix)
    saved: list[Path] = []
    src_frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_frame_idx % stride == 0:
                out_path = out_dir / f"{prefix}_{index:04d}.jpg"
                # cv2.imwrite uses BGR, which is what cap.read() returns — no conversion.
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                saved.append(out_path)
                index += 1
                if max_frames is not None and len(saved) >= max_frames:
                    break
            src_frame_idx += 1
    finally:
        cap.release()

    return saved
