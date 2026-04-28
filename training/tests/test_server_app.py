"""
Tests for the relay-server FastAPI app — uses TestClient (no live server).

Covers: auth gate, version endpoint, upload roundtrip → directory structure
on disk that the ingestion daemon will pick up.
"""
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rfconnectorai.server.app import create_app


DEVICE_TOKEN = "test-token-abc"


def _make_app(tmp_path: Path) -> TestClient:
    incoming = tmp_path / "incoming"
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "incoming_dir": incoming,
        "model_dir": model_dir,
        "device_token": DEVICE_TOKEN,
        "max_upload_bytes": 10 * 1024 * 1024,
    }
    return TestClient(create_app(config)), incoming, model_dir


def test_healthz_no_auth_required(tmp_path):
    client, _, _ = _make_app(tmp_path)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version_returns_zero_when_no_model(tmp_path):
    client, _, _ = _make_app(tmp_path)
    r = client.get("/model/version")
    assert r.status_code == 200
    assert r.json()["version"] == 0


def test_version_reads_manifest(tmp_path):
    client, _, model_dir = _make_app(tmp_path)
    (model_dir / "manifest.json").write_text(json.dumps({"version": 7}))
    r = client.get("/model/version")
    assert r.status_code == 200
    assert r.json()["version"] == 7


def test_latest_requires_auth(tmp_path):
    client, _, model_dir = _make_app(tmp_path)
    (model_dir / "manifest.json").write_text(json.dumps({
        "version": 1, "weights_filename": "weights.0001.pt"
    }))
    # No header
    r = client.get("/model/latest")
    assert r.status_code == 401
    # Bad header
    r = client.get("/model/latest", headers={"X-Device-Token": "wrong"})
    assert r.status_code == 401
    # Right header
    r = client.get("/model/latest", headers={"X-Device-Token": DEVICE_TOKEN})
    assert r.status_code == 200
    assert r.json()["version"] == 1


def test_weights_returns_file(tmp_path):
    client, _, model_dir = _make_app(tmp_path)
    (model_dir / "manifest.json").write_text(json.dumps({
        "version": 1, "weights_filename": "weights.0001.pt"
    }))
    weights_path = model_dir / "weights.0001.pt"
    weights_path.write_bytes(b"binary-weights-content" * 100)
    r = client.get("/model/weights", headers={"X-Device-Token": DEVICE_TOKEN})
    assert r.status_code == 200
    assert r.content == weights_path.read_bytes()


def test_labels_returns_file(tmp_path):
    client, _, model_dir = _make_app(tmp_path)
    labels_data = {"class_names": ["SMA-M", "SMA-F"]}
    (model_dir / "labels.json").write_text(json.dumps(labels_data))
    r = client.get("/model/labels", headers={"X-Device-Token": DEVICE_TOKEN})
    assert r.status_code == 200
    assert r.json() == labels_data


def test_upload_writes_atomic_directory(tmp_path):
    client, incoming, _ = _make_app(tmp_path)
    img1 = b"\xff\xd8\xff" + b"\x00" * 1000   # JPEG SOI prefix + filler
    img2 = b"\xff\xd8\xff" + b"\x00" * 1000
    files = [
        ("frames", ("f0.jpg", io.BytesIO(img1), "image/jpeg")),
        ("frames", ("f1.jpg", io.BytesIO(img2), "image/jpeg")),
    ]
    r = client.post(
        "/uploads",
        files=files,
        data={"claimed_class": "2.4mm-F", "device_id": "test-device", "capture_reason": "manual"},
        headers={"X-Device-Token": DEVICE_TOKEN},
    )
    assert r.status_code == 200, r.text
    upload_id = r.json()["upload_id"]
    upload_dir = incoming / upload_id
    assert upload_dir.exists()
    # Frames dropped, manifest written, ready sentinel touched.
    assert (upload_dir / "manifest.json").exists()
    assert (upload_dir / ".ready").exists()
    manifest = json.loads((upload_dir / "manifest.json").read_text())
    assert manifest["claimed_class"] == "2.4mm-F"
    assert manifest["device_id"] == "test-device"
    assert manifest["n_frames"] == 2


def test_upload_rejects_without_token(tmp_path):
    client, _, _ = _make_app(tmp_path)
    files = [("frames", ("f0.jpg", io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100), "image/jpeg"))]
    r = client.post(
        "/uploads",
        files=files,
        data={"claimed_class": "2.4mm-F", "device_id": "x"},
    )
    assert r.status_code == 401


def test_upload_rejects_too_many_frames(tmp_path):
    client, _, _ = _make_app(tmp_path)
    files = [
        ("frames", (f"f{i}.jpg", io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100), "image/jpeg"))
        for i in range(201)
    ]
    r = client.post(
        "/uploads",
        files=files,
        data={"claimed_class": "2.4mm-F", "device_id": "x"},
        headers={"X-Device-Token": DEVICE_TOKEN},
    )
    assert r.status_code == 400


def test_server_fails_closed_when_token_unconfigured(tmp_path):
    """If the operator forgets to set the token, auth-required endpoints
    must refuse rather than serve unauthenticated."""
    config = {
        "incoming_dir": tmp_path / "incoming",
        "model_dir": tmp_path / "model",
        "device_token": "",   # empty
        "max_upload_bytes": 1024,
    }
    client = TestClient(create_app(config))
    r = client.get("/model/latest", headers={"X-Device-Token": ""})
    assert r.status_code in (401, 503)
