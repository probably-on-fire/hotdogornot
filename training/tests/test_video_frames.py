"""
Tests for the video → frames extractor. Builds a tiny synthetic video on
the fly with cv2.VideoWriter so we don't need test fixtures on disk.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from rfconnectorai.data_fetch.video_frames import extract_frames


def _make_test_video(path: Path, fps: int = 30, duration_sec: int = 2,
                     size: int = 64) -> None:
    """Write a short synthetic video with a colour-changing solid background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    assert writer.isOpened(), f"could not open writer for {path}"
    n_frames = fps * duration_sec
    for i in range(n_frames):
        # Fade red intensity over time so each frame is distinguishable.
        frame = np.full((size, size, 3), [0, 0, int(255 * i / n_frames)], dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extracts_expected_count_at_2fps(tmp_path):
    video = tmp_path / "clip.mp4"
    out_dir = tmp_path / "frames"
    _make_test_video(video, fps=30, duration_sec=2)  # 60 source frames

    saved = extract_frames(video, out_dir, fps_target=2.0)
    # 30fps source / 2fps target → stride 15 → 60 / 15 = 4 frames
    assert len(saved) == 4
    for p in saved:
        assert p.exists()
        assert p.suffix == ".jpg"


def test_max_frames_caps_output(tmp_path):
    video = tmp_path / "clip.mp4"
    out_dir = tmp_path / "frames"
    _make_test_video(video, fps=30, duration_sec=4)

    saved = extract_frames(video, out_dir, fps_target=10.0, max_frames=5)
    assert len(saved) == 5


def test_indexing_resumes_across_calls(tmp_path):
    video = tmp_path / "clip.mp4"
    out_dir = tmp_path / "frames"
    _make_test_video(video, fps=30, duration_sec=1)

    first = extract_frames(video, out_dir, fps_target=2.0)
    second = extract_frames(video, out_dir, fps_target=2.0)
    # Second call should not overwrite first call's filenames.
    first_indices = sorted(int(p.stem.split("_")[1]) for p in first)
    second_indices = sorted(int(p.stem.split("_")[1]) for p in second)
    assert max(first_indices) < min(second_indices)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "no_such_file.mp4", tmp_path, fps_target=2.0)


def test_custom_prefix(tmp_path):
    video = tmp_path / "clip.mp4"
    out_dir = tmp_path / "frames"
    _make_test_video(video, fps=30, duration_sec=1)
    saved = extract_frames(video, out_dir, fps_target=2.0, prefix="custom")
    assert all(p.stem.startswith("custom_") for p in saved)
