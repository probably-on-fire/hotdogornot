"""
Tests for the labeler router — uses FastAPI TestClient against
create_router(). Each test gets a fresh tmp_path with the labeled-dir
env var pointed at it so we can build a fixture directory tree.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


def test_upload_train_returns_json(client, labeler_dirs):
    labeled, _, _ = labeler_dirs
    r = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "2.4mm-M"},
        files=[
            ("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")),
            ("images", ("b.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["saved"]) == 2
    for entry in body["saved"]:
        assert entry["cls"] == "2.4mm-M"
        assert entry["path"].endswith(".jpg")
        # Paths are absolute or relative to labeled root — the client
        # only needs them to round-trip through /delete, so we check
        # the file actually exists on disk where path says.
        assert Path(entry["path"]).exists()


def test_upload_train_rejects_unknown_class(client):
    r = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "bogus"},
        files=[("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 400


def test_upload_test_returns_json(client, labeler_dirs):
    _, holdout, _ = labeler_dirs
    r = client.post(
        "/rfcai/labeler/upload-test",
        auth=("u", "p"),
        data={"cls": "SMA-F"},
        files=[("images", ("h.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["saved"]) == 1
    assert body["saved"][0]["cls"] == "SMA-F"
    assert Path(body["saved"][0]["path"]).exists()


def test_upload_then_delete_roundtrip(client, labeler_dirs):
    # The Undo flow on the client posts the path back to /delete.
    # Pin that the path the upload returns is a valid delete target.
    r1 = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "2.4mm-F"},
        files=[("images", ("c.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    path = r1.json()["saved"][0]["path"]
    assert Path(path).exists()

    r2 = client.post(
        "/rfcai/labeler/delete",
        auth=("u", "p"),
        data={"path": path},
    )
    assert r2.status_code == 200
    assert not Path(path).exists()


def test_upload_test_then_delete_roundtrip(client, labeler_dirs):
    # Mirror of test_upload_then_delete_roundtrip but for holdout uploads.
    # Pin that /delete accepts paths under _test_holdout_root() — the
    # Undo flow on the client posts the path back to /delete regardless
    # of whether the original upload was train or holdout.
    r1 = client.post(
        "/rfcai/labeler/upload-test",
        auth=("u", "p"),
        data={"cls": "2.4mm-F"},
        files=[("images", ("h.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    path = r1.json()["saved"][0]["path"]
    assert Path(path).exists()

    r2 = client.post(
        "/rfcai/labeler/delete",
        auth=("u", "p"),
        data={"path": path},
    )
    assert r2.status_code == 200
    assert not Path(path).exists()


def test_real_capture_counts_skips_non_image_sidecars(labeler_dirs):
    labeled, _, _ = labeler_dirs
    _seed_class_dir(labeled, "SMA-F", [
        "photo_real.jpg",       # real
        ".DS_Store",            # macOS sidecar, skip
        "Thumbs.db",            # Windows sidecar, skip
        "notes.txt",            # stray text, skip
        "photo_real.json",      # sidecar metadata, skip
    ])
    counts = labeler._real_capture_counts(labeled)
    assert counts["SMA-F"] == 1


def test_upload_train_default_session_is_today(client, labeler_dirs):
    labeled, _, _ = labeler_dirs
    r = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "2.4mm-M"},
        files=[("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 200
    path = r.json()["saved"][0]["path"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"photo_{today}_a.jpg" in path
    assert Path(path).exists()


def test_upload_train_honors_explicit_session(client, labeler_dirs):
    r = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "2.4mm-M", "session": "experiment_42"},
        files=[("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 200
    path = r.json()["saved"][0]["path"]
    assert "photo_experiment_42_a.jpg" in path


def test_upload_train_rejects_invalid_session(client):
    r = client.post(
        "/rfcai/labeler/upload-train",
        auth=("u", "p"),
        data={"cls": "2.4mm-M", "session": "../../etc"},
        files=[("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 400


def test_upload_test_default_session_is_today(client, labeler_dirs):
    _, holdout, _ = labeler_dirs
    r = client.post(
        "/rfcai/labeler/upload-test",
        auth=("u", "p"),
        data={"cls": "SMA-F"},
        files=[("images", ("h.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 200
    path = r.json()["saved"][0]["path"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"{today}_h.jpg" in path
    # Holdout filenames don't carry the "photo_" prefix.
    assert "photo_" not in Path(path).name
