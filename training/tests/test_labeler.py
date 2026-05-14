"""
Tests for the labeler router — uses FastAPI TestClient against
create_router(). Each test gets a fresh tmp_path with the labeled-dir
env var pointed at it so we can build a fixture directory tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rfconnectorai.server import labeler


def _seed_class_dir(root: Path, cls: str, filenames: list[str]) -> None:
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (d / name).write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG bytes


@pytest.fixture
def labeler_dirs(tmp_path, monkeypatch):
    """Set up empty labeled + test-holdout directories and basic auth."""
    labeled = tmp_path / "labeled"
    holdout = tmp_path / "holdout"
    videos = tmp_path / "videos"
    labeled.mkdir()
    holdout.mkdir()
    videos.mkdir()
    monkeypatch.setenv("RFCAI_LABELED_DIR", str(labeled))
    monkeypatch.setenv("RFCAI_TEST_HOLDOUT_DIR", str(holdout))
    monkeypatch.setenv("RFCAI_VIDEOS_DIR", str(videos))
    monkeypatch.setenv("LABELER_USER", "u")
    monkeypatch.setenv("LABELER_PASS", "p")
    return labeled, holdout, videos


@pytest.fixture
def client(labeler_dirs):
    app = FastAPI()
    app.include_router(labeler.create_router())
    return TestClient(app)


def test_real_capture_counts_skips_synth_and_derived(labeler_dirs):
    labeled, _, _ = labeler_dirs
    _seed_class_dir(labeled, "2.4mm-M", [
        "photo_IMG_1.jpg",            # real
        "photo_IMG_2.jpg",            # real
        "video_0001.jpg",             # real (video extraction)
        "photo_IMG_1_clean.jpg",      # rembg-derived, skip
        "photo_IMG_1_mask.jpg",       # rembg mask, skip
        "photo_IMG_1_bg0.jpg",        # bg-randomized, skip
        "photo_IMG_1_z0.jpg",         # zoom-randomized, skip
        "photo_IMG_1_central.jpg",    # central crop, skip
        "photo_IMG_1_centralv2.jpg",  # central crop v2, skip
        "synth_000001.jpg",           # synth, skip
    ])
    counts = labeler._real_capture_counts(labeled)
    assert counts["2.4mm-M"] == 3
    assert counts["SMA-M"] == 0  # absent folders count as 0


def test_stats_requires_auth(client):
    r = client.get("/rfcai/labeler/stats")
    assert r.status_code == 401


def test_stats_returns_train_and_holdout_counts(client, labeler_dirs):
    labeled, holdout, _ = labeler_dirs
    _seed_class_dir(labeled, "2.4mm-M", ["photo_a.jpg", "photo_b.jpg"])
    _seed_class_dir(labeled, "2.4mm-M", ["photo_a_clean.jpg"])  # skip
    _seed_class_dir(holdout, "2.4mm-M", ["IMG_holdout.jpg"])

    r = client.get("/rfcai/labeler/stats", auth=("u", "p"))
    assert r.status_code == 200
    body = r.json()

    # All canonical classes present in both dicts, zero by default.
    for cls in labeler.CANONICAL_CLASSES:
        assert cls in body["train"]
        assert cls in body["holdout"]

    assert body["train"]["2.4mm-M"] == 2     # excludes _clean
    assert body["holdout"]["2.4mm-M"] == 1
    assert body["train"]["SMA-M"] == 0
    assert body["holdout"]["SMA-F"] == 0


def test_upload_video_requires_gender(client):
    # Without gender field → FastAPI 422 (missing required form field).
    r = client.post(
        "/rfcai/labeler/upload-video",
        auth=("u", "p"),
        data={"family": "2.4mm"},
        files={"file": ("clip.mp4", b"\x00\x00\x00\x00", "video/mp4")},
    )
    assert r.status_code == 422


def test_upload_video_rejects_invalid_class(client):
    r = client.post(
        "/rfcai/labeler/upload-video",
        auth=("u", "p"),
        data={"family": "2.4mm", "gender": "X"},
        files={"file": ("clip.mp4", b"\x00\x00\x00\x00", "video/mp4")},
    )
    assert r.status_code == 400
    assert "X" in r.text or "unknown" in r.text.lower()
