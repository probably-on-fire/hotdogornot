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


def test_stats_is_public(client):
    """Read-only routes are public so the labeler grid can be shared."""
    r = client.get("/rfcai/labeler/stats")
    assert r.status_code == 200


def test_grid_is_public(client):
    # _list_records uses imagehash for duplicate-detection; skip if
    # not installed in this venv (it lives on the production box).
    pytest.importorskip("imagehash")
    r = client.get("/rfcai/labeler/grid")
    assert r.status_code == 200


def test_delete_still_requires_auth(client):
    r = client.post("/rfcai/labeler/delete", data={"path": "/dev/null"})
    assert r.status_code == 401


def test_upload_train_still_requires_auth(client):
    r = client.post(
        "/rfcai/labeler/upload-train",
        data={"cls": "2.4mm-M"},
        files=[("images", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))],
    )
    assert r.status_code == 401


def test_stats_returns_train_and_holdout_counts(client, labeler_dirs):
    labeled, holdout, _ = labeler_dirs
    _seed_class_dir(labeled, "2.4mm-M", ["photo_a.jpg", "photo_b.jpg"])
    _seed_class_dir(labeled, "2.4mm-M", ["photo_a_clean.jpg"])  # skip
    _seed_class_dir(holdout, "2.4mm-M", ["IMG_holdout.jpg"])

    # No auth — stats is public now.
    r = client.get("/rfcai/labeler/stats")
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


def test_snapshots_lists_tarballs(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "rfcai_session_2026-05-14.tar.gz").write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 8)
    (snap / "ignored_readme.txt").write_bytes(b"not a tarball")
    monkeypatch.setenv("RFCAI_SNAPSHOT_DIR", str(snap))
    monkeypatch.setenv("LABELER_USER", "u")
    monkeypatch.setenv("LABELER_PASS", "p")
    app = FastAPI()
    app.include_router(labeler.create_router())
    c = TestClient(app)
    r = c.get("/rfcai/labeler/snapshots")
    assert r.status_code == 200
    body = r.json()
    names = [s["name"] for s in body["snapshots"]]
    assert "rfcai_session_2026-05-14.tar.gz" in names
    assert "ignored_readme.txt" not in names  # only tar/gz/zip


def test_snapshot_download_and_path_traversal_rejected(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "ok.tar.gz").write_bytes(b"\x1f\x8b\x08\x00contents")
    monkeypatch.setenv("RFCAI_SNAPSHOT_DIR", str(snap))
    monkeypatch.setenv("LABELER_USER", "u")
    monkeypatch.setenv("LABELER_PASS", "p")
    app = FastAPI()
    app.include_router(labeler.create_router())
    c = TestClient(app)
    # Happy path: download the file.
    r = c.get("/rfcai/labeler/snapshots/ok.tar.gz")
    assert r.status_code == 200
    assert r.content.startswith(b"\x1f\x8b")
    # Traversal: dotdot must be rejected.
    r = c.get("/rfcai/labeler/snapshots/..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404)
    # Missing file → 404.
    r = c.get("/rfcai/labeler/snapshots/nope.tar.gz")
    assert r.status_code == 404


def test_login_get_returns_form(client):
    r = client.get("/rfcai/labeler/login")
    assert r.status_code == 200
    assert b"username" in r.content.lower()
    assert b"password" in r.content.lower()


def test_login_post_invalid_credentials_redirects_with_error(client, tmp_path, monkeypatch):
    db = tmp_path / "users.db"
    monkeypatch.setenv("RFCAI_USERS_DB", str(db))
    from rfconnectorai.server import auth as _auth
    _auth.init_db(db)
    _auth.create_user(db, "alice", "correctpw", role="admin")

    # We need a fresh client because the env var is read per call;
    # the fixture's client uses an older env. So rebuild here.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.add_middleware(
        __import__("starlette.middleware.sessions", fromlist=["SessionMiddleware"]).SessionMiddleware,
        secret_key="test-secret",
    )
    app.include_router(labeler.create_router())
    c = TestClient(app)

    r = c.post(
        "/rfcai/labeler/login",
        data={"username": "alice", "password": "wrong", "next": "/rfcai/labeler/"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "/rfcai/labeler/login" in r.headers["location"]
    assert "error=" in r.headers["location"]


def test_login_post_valid_credentials_sets_session(tmp_path, monkeypatch):
    db = tmp_path / "users.db"
    monkeypatch.setenv("RFCAI_USERS_DB", str(db))
    from rfconnectorai.server import auth as _auth
    _auth.init_db(db)
    _auth.create_user(db, "alice", "correctpw", role="admin")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.include_router(labeler.create_router())
    c = TestClient(app)

    r = c.post(
        "/rfcai/labeler/login",
        data={"username": "alice", "password": "correctpw", "next": "/rfcai/labeler/"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/rfcai/labeler/"
    # Session cookie was set.
    assert "session" in r.cookies


def test_logout_clears_session(tmp_path, monkeypatch):
    db = tmp_path / "users.db"
    monkeypatch.setenv("RFCAI_USERS_DB", str(db))
    from rfconnectorai.server import auth as _auth
    _auth.init_db(db)
    _auth.create_user(db, "alice", "pw", role="admin")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.include_router(labeler.create_router())
    c = TestClient(app)

    # Log in first.
    c.post("/rfcai/labeler/login",
           data={"username": "alice", "password": "pw"},
           follow_redirects=False)
    # Logout.
    r = c.post("/rfcai/labeler/logout", follow_redirects=False)
    assert r.status_code == 303


def test_require_admin_accepts_basic_auth_against_users_db(tmp_path, monkeypatch):
    """The Flutter app uses HTTP Basic; require_admin must validate
    against the users DB, not the old LABELER_USER/PASS env vars."""
    db = tmp_path / "users.db"
    monkeypatch.setenv("RFCAI_USERS_DB", str(db))
    from rfconnectorai.server import auth as _auth
    _auth.init_db(db)
    _auth.create_user(db, "alice", "pw", role="admin")

    # Build an isolated mini-app that has a route protected by require_admin.
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware
    from rfconnectorai.server.labeler import require_admin
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/protected")
    def protected(user=Depends(require_admin)):
        return {"username": user.username}

    c = TestClient(app)

    # No auth -> 401.
    assert c.get("/protected").status_code == 401

    # Correct basic auth -> 200.
    r = c.get("/protected", auth=("alice", "pw"))
    assert r.status_code == 200
    assert r.json()["username"] == "alice"

    # Wrong password -> 401.
    assert c.get("/protected", auth=("alice", "nope")).status_code == 401


def test_require_admin_accepts_session_cookie(tmp_path, monkeypatch):
    db = tmp_path / "users.db"
    monkeypatch.setenv("RFCAI_USERS_DB", str(db))
    from rfconnectorai.server import auth as _auth
    _auth.init_db(db)
    _auth.create_user(db, "alice", "pw", role="admin")

    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware
    from rfconnectorai.server.labeler import require_admin
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.include_router(labeler.create_router())

    @app.get("/protected")
    def protected(user=Depends(require_admin)):
        return {"username": user.username}

    c = TestClient(app)
    # Log in via the labeler /login endpoint; session cookie carries over.
    c.post("/rfcai/labeler/login",
           data={"username": "alice", "password": "pw"},
           follow_redirects=False)
    r = c.get("/protected")
    assert r.status_code == 200
    assert r.json()["username"] == "alice"
