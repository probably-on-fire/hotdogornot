# Contribute Screen Capture-Session Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make four small changes to the Flutter Contribute flow + labeler server so a bulk training-data capture session can run end-to-end on the phone (per-class stats, video gender, in-app Undo, on-device label-check).

**Architecture:** Server-first ordering — three small labeler-endpoint changes land first, then the Flutter `api.dart` client gets new methods, then the four UI changes go on top of `contribute_screen.dart`. Each task is independently testable so we can verify on the running predict-service box without waiting for the full feature.

**Tech Stack:** FastAPI (labeler router on the predict service), pytest + FastAPI `TestClient` for server tests, Flutter `camera` + `http` + `onnxruntime` for the client, `flutter_test` for what's testable without a real camera.

**Reference spec:** `docs/superpowers/specs/2026-05-14-contribute-capture-improvements-design.md`

---

## File map

**Modify (server):**
- `training/rfconnectorai/server/labeler.py` — add `/stats` endpoint, change `/upload-train` and `/upload-test` to JSON, add `gender` to `/upload-video`. New helper `_real_capture_counts()`.

**Create (server tests):**
- `training/tests/test_labeler.py` — new file. Uses `TestClient` against `create_router()` plus a temp `RFCAI_LABELED_DIR`.

**Modify (Flutter client):**
- `flutter/lib/src/api.dart` — change `uploadTrainingPhoto` / `uploadTestHoldoutPhoto` return type from `String` to a new `UploadResult` with parsed `saved` list; add `gender` param to `uploadTrainingVideo`; new methods `fetchLabelerStats()` and `deleteLabelerFile(path)`.
- `flutter/lib/src/settings.dart` — add a `labelerStatsUrl()` getter mirroring the existing `labelerUploadTrainUrl()` pattern (if not already present).
- `flutter/lib/src/screens/contribute_screen.dart` — per-class session counters, stats bottom-sheet, Undo stack + button, gender passthrough to video upload, on-device classifier integration.

**Modify (Flutter tests):**
- `flutter/test/api_test.dart` — extend with `UploadResult` JSON parsing tests.

---

## Task ordering rationale

1–4: server side (new stats, JSON responses on upload, gender on video). All tested with `pytest`. Deployable independently.
5: client API surface — extends `api.dart` to consume the new server shapes. Pure parser tests.
6: per-class session counters + stats bottom-sheet (§1).
7: video upload gender passthrough (§2).
8: Undo stack + button (§3).
9: on-device label-check toast (§4).
10: manual end-to-end smoke + spec sign-off.

---

## Task 1: Labeler test scaffold + `_real_capture_counts()` helper

**Files:**
- Create: `training/tests/test_labeler.py`
- Modify: `training/rfconnectorai/server/labeler.py` (add helper near line 232)

- [ ] **Step 1: Write the failing test for `_real_capture_counts`**

Create `training/tests/test_labeler.py`:

```python
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
        "photo_IMG_1.jpg",          # real
        "photo_IMG_2.jpg",          # real
        "video_0001.jpg",           # real (video extraction)
        "photo_IMG_1_clean.jpg",    # rembg-derived, skip
        "photo_IMG_1_bg0.jpg",      # bg-randomized, skip
        "photo_IMG_1_z0.jpg",       # zoom-randomized, skip
        "photo_IMG_1_central.jpg",  # central crop, skip
        "synth_000001.jpg",         # synth, skip
    ])
    counts = labeler._real_capture_counts(labeled)
    assert counts["2.4mm-M"] == 3
    assert counts["SMA-M"] == 0  # absent folders count as 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && python -m pytest tests/test_labeler.py::test_real_capture_counts_skips_synth_and_derived -v`
Expected: FAIL — `AttributeError: module 'rfconnectorai.server.labeler' has no attribute '_real_capture_counts'`

- [ ] **Step 3: Implement `_real_capture_counts`**

In `training/rfconnectorai/server/labeler.py`, immediately after the existing `_class_counts()` function (around line 238), add:

```python
def _real_capture_counts(root: Path) -> dict[str, int]:
    """
    Count real human captures per class — excludes rembg-derived
    variants (_clean / _bg* / _z* / _central) and pure synth files
    (synth_*). Used by the stats endpoint to show the user the number
    they're actually moving with each shutter tap, not the augmented
    training-set size.
    """
    out: dict[str, int] = {}
    for cls in CANONICAL_CLASSES:
        d = root / cls
        if not d.is_dir():
            out[cls] = 0
            continue
        n = 0
        for p in d.iterdir():
            if not p.is_file():
                continue
            stem = p.stem
            if stem.startswith("synth_"):
                continue
            if stem.endswith("_clean") or stem.endswith("_central"):
                continue
            # _bg<digits>, _z<digits> — derived augmentations.
            if "_bg" in stem and stem.rsplit("_bg", 1)[1].isdigit():
                continue
            if "_z" in stem and stem.rsplit("_z", 1)[1].isdigit():
                continue
            n += 1
        out[cls] = n
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd training && python -m pytest tests/test_labeler.py::test_real_capture_counts_skips_synth_and_derived -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/tests/test_labeler.py training/rfconnectorai/server/labeler.py
git commit -m "labeler: add _real_capture_counts helper (excludes synth/derived)"
```

---

## Task 2: `GET /labeler/stats` endpoint

**Files:**
- Modify: `training/rfconnectorai/server/labeler.py` (add route inside `create_router()`)
- Modify: `training/tests/test_labeler.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_labeler.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python -m pytest tests/test_labeler.py -v -k "stats"`
Expected: both FAIL — 404 (route does not exist).

- [ ] **Step 3: Implement the route**

In `training/rfconnectorai/server/labeler.py`, inside `create_router()`, just after the `r = APIRouter(...)` line (around line 276), add (before the existing `@r.get("/")`):

```python
    @r.get("/stats")
    def stats(_: str = Depends(_require_basic_auth)):
        """
        Per-class real-capture counts for the training set and the
        held-out test set. Used by the Flutter Contribute screen to
        show the user which classes are starved before they capture.
        """
        return {
            "train": _real_capture_counts(_data_root()),
            "holdout": _real_capture_counts(_test_holdout_root()),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python -m pytest tests/test_labeler.py -v -k "stats"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/server/labeler.py training/tests/test_labeler.py
git commit -m "labeler: add GET /stats returning per-class real-capture counts"
```

---

## Task 3: `POST /upload-video` accepts `gender`

**Files:**
- Modify: `training/rfconnectorai/server/labeler.py:511-547`
- Modify: `training/tests/test_labeler.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_labeler.py`:

```python
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
```

(We don't test the full happy-path extraction here — ffmpeg + Hough are
heavy and out-of-scope for this change. The validation tests pin the
new behavior; the legacy extraction path is unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python -m pytest tests/test_labeler.py -v -k "upload_video"`
Expected: both FAIL — first because `gender` is silently dropped today (returns 200 or different error), second because no class-validation step exists yet.

- [ ] **Step 3: Implement gender field**

In `training/rfconnectorai/server/labeler.py`, change the `upload_video` route signature and the `target_cls` assignment.

Replace lines 511–520:

```python
    @r.post("/upload-video", response_class=HTMLResponse)
    async def upload_video(
        family: str = Form(...),
        fps: float = Form(5.0),
        sensitivity: float = Form(2.0),
        max_crops: int = Form(5),
        file: UploadFile = File(...),
        _: str = Depends(_require_basic_auth),
    ):
        if family not in CANONICAL_FAMILIES:
            raise HTTPException(400, f"unknown family {family!r}")
```

with:

```python
    @r.post("/upload-video", response_class=HTMLResponse)
    async def upload_video(
        family: str = Form(...),
        gender: str = Form(...),
        fps: float = Form(5.0),
        sensitivity: float = Form(2.0),
        max_crops: int = Form(5),
        file: UploadFile = File(...),
        _: str = Depends(_require_basic_auth),
    ):
        if family not in CANONICAL_FAMILIES:
            raise HTTPException(400, f"unknown family {family!r}")
        target_cls = f"{family}-{gender}"
        if target_cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {target_cls!r}")
```

Then further down in the same function (around line 547), replace:

```python
        target_cls = f"{family}-M"
```

with:

```python
        # target_cls already validated above
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python -m pytest tests/test_labeler.py -v -k "upload_video"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/server/labeler.py training/tests/test_labeler.py
git commit -m "labeler: require gender on /upload-video, validate class"
```

---

## Task 4: `/upload-train` and `/upload-test` return JSON

**Files:**
- Modify: `training/rfconnectorai/server/labeler.py:428-509`
- Modify: `training/tests/test_labeler.py`

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_labeler.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd training && python -m pytest tests/test_labeler.py -v -k "upload_t or roundtrip"`
Expected: FAIL — endpoints currently return HTML, `r.json()` raises.

- [ ] **Step 3: Convert responses to JSON**

In `training/rfconnectorai/server/labeler.py`, modify the `upload_train` and `upload_test` routes.

Replace the `@r.post("/upload-train", response_class=HTMLResponse)` decorator and the `return HTMLResponse(...)` lines so the route looks like:

```python
    @r.post("/upload-train")
    async def upload_train(
        cls: str = Form(...),
        images: list[UploadFile] = File(...),
        _: str = Depends(_require_basic_auth),
    ):
        """Drop phone photos directly into the training set for a class.

        Saved to data/labeled/embedder/<class>/photo_<stem><ext>.
        Returns JSON {saved: [{cls, path}], errors: [str]} so the Flutter
        Undo flow can stash the server-authoritative path."""
        if cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {cls!r}")
        out_dir = _data_root() / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        errors: list[str] = []
        for image in images:
            if not image.filename:
                errors.append("empty filename")
                continue
            ext = Path(image.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                errors.append(f"bad ext {image.filename!r}")
                continue
            stem = Path(image.filename).stem or f"upload_{int(time.time())}"
            stem = Path(stem).name
            dst = out_dir / f"photo_{stem}{ext}"
            n = 1
            while dst.exists():
                dst = out_dir / f"photo_{stem}_dup{n}{ext}"
                n += 1
            data = await image.read()
            dst.write_bytes(data)
            saved.append({"cls": cls, "path": str(dst)})
        global _signals_cache
        _signals_cache = {}
        return {"saved": saved, "errors": errors}
```

And do the same for `upload_test`:

```python
    @r.post("/upload-test")
    async def upload_test(
        cls: str = Form(...),
        images: list[UploadFile] = File(...),
        _: str = Depends(_require_basic_auth),
    ):
        if cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {cls!r}")
        out_dir = _test_holdout_root() / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        errors: list[str] = []
        for image in images:
            if not image.filename:
                errors.append("empty filename")
                continue
            ext = Path(image.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                errors.append(f"bad ext {image.filename!r}")
                continue
            stem = Path(image.filename).stem or f"upload_{int(time.time())}"
            stem = Path(stem).name
            dst = out_dir / f"{stem}{ext}"
            n = 1
            while dst.exists():
                dst = out_dir / f"{stem}_dup{n}{ext}"
                n += 1
            data = await image.read()
            dst.write_bytes(data)
            saved.append({"cls": cls, "path": str(dst)})
        return {"saved": saved, "errors": errors}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd training && python -m pytest tests/test_labeler.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/server/labeler.py training/tests/test_labeler.py
git commit -m "labeler: /upload-train and /upload-test return JSON with saved paths"
```

---

## Task 5: Flutter API client — UploadResult, stats, delete, video gender

**Files:**
- Modify: `flutter/lib/src/api.dart`
- Modify: `flutter/lib/src/settings.dart` (only if `labelerStatsUrl()` is missing — read first)
- Modify: `flutter/test/api_test.dart`

- [ ] **Step 1: Add stats + delete URLs to settings**

In `flutter/lib/src/settings.dart`, after the existing `labelerUploadVideoUrl()` (line 73), add:

```dart
  String labelerStatsUrl() => '$relayBaseUrl/labeler/stats';

  String labelerDeleteUrl() => '$relayBaseUrl/labeler/delete';
```

(All existing methods use the `$relayBaseUrl/labeler/...` interpolation
pattern; match it.)

- [ ] **Step 2: Write the failing tests**

Append to `flutter/test/api_test.dart`:

```dart
import 'package:connector_id/src/api.dart' show UploadResult, UploadRecord;
// (If the import line at the top already imports `api.dart`, just add
// UploadResult and UploadRecord to the existing show clause.)

void main_uploadResult() {  // illustrative, fold into existing main()
  group('UploadResult.fromJson', () {
    test('parses saved + errors', () {
      final r = UploadResult.fromJson({
        'saved': [
          {'cls': '2.4mm-M', 'path': '/data/labeled/embedder/2.4mm-M/photo_a.jpg'},
          {'cls': '2.4mm-M', 'path': '/data/labeled/embedder/2.4mm-M/photo_b.jpg'},
        ],
        'errors': <String>[],
      });
      expect(r.saved, hasLength(2));
      expect(r.saved.first.cls, '2.4mm-M');
      expect(r.saved.first.path,
          '/data/labeled/embedder/2.4mm-M/photo_a.jpg');
      expect(r.errors, isEmpty);
    });

    test('parses errors-only response', () {
      final r = UploadResult.fromJson({
        'saved': const [],
        'errors': const ['bad ext "x.bmp"'],
      });
      expect(r.saved, isEmpty);
      expect(r.errors, hasLength(1));
    });

    test('missing errors field defaults to empty', () {
      final r = UploadResult.fromJson({
        'saved': [
          {'cls': 'SMA-F', 'path': '/p.jpg'},
        ],
      });
      expect(r.errors, isEmpty);
    });
  });
}
```

(Merge the `group` into the existing `main()` instead of creating
`main_uploadResult`; the function shown above is just for clarity in
the plan. Imports go at the top of the file.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd flutter && flutter test test/api_test.dart`
Expected: FAIL — `UploadResult` undefined.

- [ ] **Step 4: Implement `UploadResult` + new client methods**

In `flutter/lib/src/api.dart`, at the top of the file (above `class Prediction`), add:

```dart
/// One file the server saved. The `path` round-trips through
/// `/labeler/delete` for the Undo flow.
class UploadRecord {
  UploadRecord({required this.cls, required this.path});
  final String cls;
  final String path;

  factory UploadRecord.fromJson(Map<String, dynamic> j) =>
      UploadRecord(cls: j['cls'] as String, path: j['path'] as String);
}

/// Parsed JSON from `/upload-train` and `/upload-test`.
class UploadResult {
  UploadResult({required this.saved, required this.errors});
  final List<UploadRecord> saved;
  final List<String> errors;

  factory UploadResult.fromJson(Map<String, dynamic> j) {
    final saved = (j['saved'] as List? ?? [])
        .map((e) => UploadRecord.fromJson(e as Map<String, dynamic>))
        .toList();
    final errors = (j['errors'] as List? ?? [])
        .map((e) => e.toString())
        .toList();
    return UploadResult(saved: saved, errors: errors);
  }
}

/// Per-class capture counts returned by `/labeler/stats`.
class LabelerStats {
  LabelerStats({required this.train, required this.holdout});
  final Map<String, int> train;
  final Map<String, int> holdout;

  factory LabelerStats.fromJson(Map<String, dynamic> j) {
    Map<String, int> _ints(Map<String, dynamic> m) =>
        m.map((k, v) => MapEntry(k, (v as num).toInt()));
    return LabelerStats(
      train: _ints(j['train'] as Map<String, dynamic>),
      holdout: _ints(j['holdout'] as Map<String, dynamic>),
    );
  }
}
```

Change the existing methods:

```dart
  Future<UploadResult> uploadTrainingPhoto(File imageFile, String cls) async {
    final body = await _uploadMultipart(
      url: settings.labelerUploadTrainUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      file: imageFile,
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  Future<UploadResult> uploadTrainingPhotoBytes(
    Uint8List bytes, String cls, {String filename = 'photo.jpg'},
  ) async {
    final body = await _uploadMultipartBytes(
      url: settings.labelerUploadTrainUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      bytes: bytes,
      filename: filename,
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  Future<UploadResult> uploadTestHoldoutPhoto(File imageFile, String cls) async {
    final body = await _uploadMultipart(
      url: settings.labelerUploadTestUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      file: imageFile,
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  Future<UploadResult> uploadTestHoldoutPhotoBytes(
    Uint8List bytes, String cls, {String filename = 'photo.jpg'},
  ) async {
    final body = await _uploadMultipartBytes(
      url: settings.labelerUploadTestUrl(),
      fields: {'cls': cls},
      fileField: 'images',
      bytes: bytes,
      filename: filename,
    );
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }
```

Change `uploadTrainingVideo` signature:

```dart
  Future<String> uploadTrainingVideo(
      File videoFile, String family, String gender) async {
    return _uploadMultipart(
      url: settings.labelerUploadVideoUrl(),
      fields: {
        'family': family,
        'gender': gender,
        'fps': '5',
        'sensitivity': '2.0',
        'max_crops': '5',
      },
      fileField: 'file',
      file: videoFile,
    );
  }
```

Add two new methods to `ApiClient`:

```dart
  /// GET /labeler/stats — per-class real-capture counts.
  Future<LabelerStats> fetchLabelerStats() async {
    final req = http.Request('GET', Uri.parse(settings.labelerStatsUrl()));
    final basic = base64Encode(utf8.encode(
      '${settings.labelerUser}:${settings.labelerPass}',
    ));
    req.headers['Authorization'] = 'Basic $basic';
    final streamed = await req.send().timeout(const Duration(seconds: 15));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
    return LabelerStats.fromJson(jsonDecode(resp.body));
  }

  /// POST /labeler/delete — unlink one server-side file by path.
  /// Used by the in-app Undo flow.
  Future<void> deleteLabelerFile(String path) async {
    final req = http.MultipartRequest(
      'POST', Uri.parse(settings.labelerDeleteUrl()),
    );
    final basic = base64Encode(utf8.encode(
      '${settings.labelerUser}:${settings.labelerPass}',
    ));
    req.headers['Authorization'] = 'Basic $basic';
    req.fields['path'] = path;
    final streamed = await req.send().timeout(const Duration(seconds: 15));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw _ApiError(resp.statusCode, resp.body);
    }
  }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd flutter && flutter test test/api_test.dart`
Expected: PASS (including new `UploadResult.fromJson` group).

- [ ] **Step 6: Commit**

```bash
git add flutter/lib/src/api.dart flutter/lib/src/settings.dart flutter/test/api_test.dart
git commit -m "flutter/api: UploadResult, LabelerStats, deleteLabelerFile, video gender"
```

---

## Task 6: Contribute screen — per-class session counters + stats bottom-sheet (§1)

**Files:**
- Modify: `flutter/lib/src/screens/contribute_screen.dart`

- [ ] **Step 1: Add per-class session counters and uploaded-class tracking**

In `_ContributeScreenState`, near the existing counter fields:

```dart
  final Map<String, int> _sessionCounts = {};       // training uploads per class
  final Map<String, int> _sessionHoldout = {};      // holdout uploads per class
```

In both `_uploadFile` and `_uploadBytes`, on a successful upload, replace:

```dart
      if (!mounted) return;
      setState(() => _uploadedCount++);
```

with:

```dart
      if (!mounted) return;
      setState(() {
        _uploadedCount++;
        final m = isHoldout ? _sessionHoldout : _sessionCounts;
        m[cls] = (m[cls] ?? 0) + 1;
      });
```

(The new code is the same in both methods because `cls` and `isHoldout`
are already captured at the top of each.)

- [ ] **Step 2: Make the counter pill tappable, open the bottom-sheet**

In `_buildTopBar`, wrap `_CounterPill` in a `GestureDetector` that opens the sheet:

```dart
        GestureDetector(
          onTap: _showStatsSheet,
          child: _CounterPill(
            count: _uploadedCount,
            inFlight: _uploadInFlight,
          ),
        ),
```

Add the method on `_ContributeScreenState`:

```dart
  Future<void> _showStatsSheet() async {
    HapticFeedback.selectionClick();
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF1B1B1B),
      isScrollControlled: true,
      builder: (ctx) => _StatsSheet(
        settings: widget.settings,
        sessionTrain: Map.unmodifiable(_sessionCounts),
        sessionHoldout: Map.unmodifiable(_sessionHoldout),
      ),
    );
  }
```

- [ ] **Step 3: Add the `_StatsSheet` widget**

At the bottom of `contribute_screen.dart`, after the other private widgets:

```dart
class _StatsSheet extends StatefulWidget {
  const _StatsSheet({
    required this.settings,
    required this.sessionTrain,
    required this.sessionHoldout,
  });
  final Settings settings;
  final Map<String, int> sessionTrain;
  final Map<String, int> sessionHoldout;

  @override
  State<_StatsSheet> createState() => _StatsSheetState();
}

class _StatsSheetState extends State<_StatsSheet> {
  LabelerStats? _stats;
  String? _err;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final s = await ApiClient(widget.settings).fetchLabelerStats();
      if (!mounted) return;
      setState(() {
        _stats = s;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  static const _kClasses = [
    'SMA-M', 'SMA-F',
    '1.85mm-M', '1.85mm-F',
    '2.4mm-M', '2.4mm-F',
    '2.92mm-M', '2.92mm-F',
    '3.5mm-M', '3.5mm-F',
  ];

  @override
  Widget build(BuildContext context) {
    final stats = _stats;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'Per-class progress',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.refresh, color: Colors.white70),
                  onPressed: _loading ? null : _refresh,
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (_loading)
              const Padding(
                padding: EdgeInsets.all(24),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_err != null)
              Text(
                'Failed to load stats: $_err',
                style: const TextStyle(color: Colors.redAccent),
              )
            else
              ..._kClasses.map((cls) {
                final sessTrain = widget.sessionTrain[cls] ?? 0;
                final sessHoldout = widget.sessionHoldout[cls] ?? 0;
                final serverTrain = stats?.train[cls] ?? 0;
                final serverHoldout = stats?.holdout[cls] ?? 0;
                final starved = serverTrain < 5;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      if (starved)
                        const Padding(
                          padding: EdgeInsets.only(right: 6),
                          child: Icon(Icons.circle,
                              color: Colors.redAccent, size: 8),
                        )
                      else
                        const SizedBox(width: 14),
                      Expanded(
                        child: Text(
                          cls,
                          style: TextStyle(
                            color: starved
                                ? Colors.redAccent
                                : Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      _statCell(
                          label: 'session', train: sessTrain, holdout: sessHoldout),
                      const SizedBox(width: 16),
                      _statCell(
                          label: 'server', train: serverTrain, holdout: serverHoldout),
                    ],
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }

  Widget _statCell({
    required String label,
    required int train,
    required int holdout,
  }) {
    return SizedBox(
      width: 90,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            label,
            style: const TextStyle(color: Colors.white38, fontSize: 10),
          ),
          Text(
            holdout > 0 ? '$train  +$holdout' : '$train',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 4: Build the app, verify in the simulator**

Run: `cd flutter && flutter analyze && flutter run -d <device>` (or use whichever device target is normal).

Manually verify:
- Open Contribute, capture two photos of different chips.
- Tap the counter pill — sheet opens, session column shows 1 + 1.
- Refresh — server column updates.
- Close, capture again — sheet reflects new totals.

(No widget test for this — `CameraController` is hard to fake. Manual check is the verification step.)

- [ ] **Step 5: Commit**

```bash
git add flutter/lib/src/screens/contribute_screen.dart
git commit -m "flutter/contribute: per-class session counts + stats bottom-sheet"
```

---

## Task 7: Contribute screen — thread gender through video upload (§2)

**Files:**
- Modify: `flutter/lib/src/screens/contribute_screen.dart`

- [ ] **Step 1: Pass gender to the API client**

Find `_pickAndUploadVideo` (around line 206). Replace:

```dart
      await api.uploadTrainingVideo(File(picked.path!), _family);
```

with:

```dart
      await api.uploadTrainingVideo(File(picked.path!), _family, _gender);
```

(The toast and other lines stay; only the call site changes.)

- [ ] **Step 2: Verify it compiles**

Run: `cd flutter && flutter analyze`
Expected: no errors. Compile-time signature check is sufficient — the API client has the new param required, so a missing argument is a compile error.

- [ ] **Step 3: Commit**

```bash
git add flutter/lib/src/screens/contribute_screen.dart
git commit -m "flutter/contribute: pass gender through to video upload"
```

---

## Task 8: Contribute screen — Undo stack + button (§3)

**Files:**
- Modify: `flutter/lib/src/screens/contribute_screen.dart`

- [ ] **Step 1: Add the Undo stack to state**

In `_ContributeScreenState`:

```dart
  // Tail-ordered stack of server-acked uploads in this session.
  // Tapping Undo pops + DELETEs the tail. Capped at 50 to bound memory.
  final List<UploadRecord> _undoStack = [];
```

- [ ] **Step 2: Populate the stack from upload responses**

In `_uploadFile` and `_uploadBytes`, replace the success branches that currently only increment counters. The new shape for each method's `try` block:

```dart
      final api = ApiClient(widget.settings);
      final UploadResult result = isHoldout
          ? await api.uploadTestHoldoutPhoto(f, cls)
          : await api.uploadTrainingPhoto(f, cls);
      if (!mounted) return;
      setState(() {
        for (final rec in result.saved) {
          _undoStack.add(rec);
          if (_undoStack.length > 50) _undoStack.removeAt(0);
          _uploadedCount++;
          final m = isHoldout ? _sessionHoldout : _sessionCounts;
          m[rec.cls] = (m[rec.cls] ?? 0) + 1;
        }
      });
      _showToast('✓ #$_uploadedCount $cls${isHoldout ? " · holdout" : ""}');
      HapticFeedback.selectionClick();
```

(The same shape applies to `_uploadBytes` — only `api.upload...Bytes(...)` vs `api.upload...(...)` differs.)

- [ ] **Step 3: Add `_undoLast()`**

```dart
  Future<void> _undoLast() async {
    if (_undoStack.isEmpty) return;
    final last = _undoStack.removeLast();
    if (mounted) setState(() {});
    try {
      await ApiClient(widget.settings).deleteLabelerFile(last.path);
      if (!mounted) return;
      setState(() {
        _uploadedCount = (_uploadedCount - 1).clamp(0, 1 << 30);
        final m = _sessionHoldout.containsKey(last.cls)
                  && (_sessionHoldout[last.cls] ?? 0) > 0
            ? _sessionHoldout
            : _sessionCounts;
        // Decrement whichever map the upload landed in — we can't tell
        // from the record alone, so prefer holdout if it has a positive
        // count for this class (matches the most-recent-shot
        // assumption). If neither matches, fall back to _sessionCounts.
        if ((m[last.cls] ?? 0) > 0) {
          m[last.cls] = m[last.cls]! - 1;
        }
      });
      _showToast('↩ Undone ${last.cls}');
    } catch (e) {
      // Restore the stack entry so the user can retry.
      if (mounted) {
        setState(() => _undoStack.add(last));
      }
      _showToast('Undo failed: ${_friendlyError(e)}', error: true);
    }
  }
```

(Heuristic decrement is a known approximation — for a perfectly
accurate decrement we'd need `UploadRecord` to remember `isHoldout`.
See Step 4 for the better fix.)

- [ ] **Step 4: Make Undo holdout-accurate**

Replace the `UploadRecord` usage in this screen with a local wrapper
that remembers holdout. At the top of `contribute_screen.dart`:

```dart
class _SessionRecord {
  _SessionRecord({required this.record, required this.holdout});
  final UploadRecord record;
  final bool holdout;
}
```

Change the stack type to `List<_SessionRecord>`:

```dart
  final List<_SessionRecord> _undoStack = [];
```

And in the upload success branch from Step 2:

```dart
        for (final rec in result.saved) {
          _undoStack.add(_SessionRecord(record: rec, holdout: isHoldout));
          if (_undoStack.length > 50) _undoStack.removeAt(0);
          ...
        }
```

And `_undoLast()` becomes:

```dart
  Future<void> _undoLast() async {
    if (_undoStack.isEmpty) return;
    final last = _undoStack.removeLast();
    if (mounted) setState(() {});
    try {
      await ApiClient(widget.settings).deleteLabelerFile(last.record.path);
      if (!mounted) return;
      setState(() {
        _uploadedCount = (_uploadedCount - 1).clamp(0, 1 << 30);
        final m = last.holdout ? _sessionHoldout : _sessionCounts;
        if ((m[last.record.cls] ?? 0) > 0) {
          m[last.record.cls] = m[last.record.cls]! - 1;
        }
      });
      _showToast('↩ Undone ${last.record.cls}');
    } catch (e) {
      if (mounted) setState(() => _undoStack.add(last));
      _showToast('Undo failed: ${_friendlyError(e)}', error: true);
    }
  }
```

- [ ] **Step 5: Add the Undo small-action button**

In `_buildControls`, modify the bottom `Row` that hosts Gallery + Video:

```dart
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _SmallAction(
              icon: Icons.photo_library,
              label: 'Gallery',
              onTap: _pickFromGallery,
            ),
            const SizedBox(width: 24),
            _SmallAction(
              icon: Icons.videocam,
              label: 'Video',
              onTap: _pickAndUploadVideo,
            ),
            const SizedBox(width: 24),
            _SmallAction(
              icon: Icons.undo,
              label: _undoStack.isEmpty ? 'Undo' : 'Undo (${_undoStack.length})',
              onTap: _undoStack.isEmpty ? null : _undoLast,
            ),
          ],
        ),
```

Also modify `_SmallAction` to disable when `onTap` is null:

```dart
class _SmallAction extends StatelessWidget {
  const _SmallAction({
    required this.icon,
    required this.label,
    required this.onTap,
  });
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(disabled ? 0.3 : 0.55),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color: disabled ? Colors.white12 : Colors.white24),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14,
                color: disabled ? Colors.white24 : Colors.white70),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: disabled ? Colors.white38 : Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 6: Build and manually verify**

Run: `cd flutter && flutter analyze && flutter run -d <device>`

Verify:
- Capture three photos in a row, then tap Undo three times — counter drops to 0, server-side files are gone (check labeler grid UI on aired.com).
- Tap Undo with empty stack — disabled, no action.
- Toggle Holdout, capture, Undo — holdout file is deleted (verify via Streamlit Manage Data).

- [ ] **Step 7: Commit**

```bash
git add flutter/lib/src/screens/contribute_screen.dart
git commit -m "flutter/contribute: Undo last upload (single-tap, stack of 50)"
```

---

## Task 9: Contribute screen — on-device label-check toast (§4)

**Files:**
- Modify: `flutter/lib/src/screens/contribute_screen.dart`

- [ ] **Step 1: Preload the on-device classifier in `_initCamera`**

Import at the top of `contribute_screen.dart`:

```dart
import '../ondevice/classifier.dart';
```

Inside `_initCamera()`, near the start (before the camera setup) add:

```dart
    // Fire-and-forget warm-up — singleton means Identify benefits too.
    unawaited(OnDeviceClassifier.instance.init().catchError((_) {}));
```

(`OnDeviceClassifier.instance.init()` is the actual public method name —
verified in `flutter/lib/src/ondevice/classifier.dart:65`. The `predict`
method's signature is `Future<OnDevicePrediction> predict(Uint8List
imageBytes)` — note it takes bytes, not a File. Step 2 reflects this.)

- [ ] **Step 2: Add `_runOnDeviceCheck` and wire it after shutter**

Add a method to `_ContributeScreenState`:

```dart
  Future<void> _runOnDeviceCheck(File f, String cls) async {
    try {
      final bytes = await f.readAsBytes();
      final pred = await OnDeviceClassifier.instance.predict(bytes);
      if (!mounted) return;
      final agree = pred.className == cls;
      if (agree) {
        _showToast(
          '✓ #$_uploadedCount $cls (agree ${pred.confidence.toStringAsFixed(2)})',
        );
        return;
      }
      final selFamily = cls.contains('-')
          ? cls.substring(0, cls.lastIndexOf('-'))
          : cls;
      final familyMatch = pred.family == selFamily;
      final lowConf = pred.confidence < 0.4;
      if (familyMatch && !lowConf) {
        _showToast(
          '⚠ picked ${cls.substring(cls.lastIndexOf("-") + 1)}, '
          'model says ${pred.gender} (${pred.confidence.toStringAsFixed(2)})',
          error: true,
        );
      } else {
        _showToast(
          '⚠ model says ${pred.className} '
          '(${pred.confidence.toStringAsFixed(2)})',
          error: true,
        );
      }
    } catch (_) {
      // Model not loaded / inference failed — silently fall back.
    }
  }
```

- [ ] **Step 3: Fire the check in parallel with upload**

In `_onShutter`, immediately after the existing `cam.takePicture()` /
fire-and-forget upload, add:

```dart
        unawaited(_runOnDeviceCheck(File(shot.path), _classLabel));
```

So the relevant block becomes:

```dart
      if (cam != null && cam.value.isInitialized) {
        final shot = await cam.takePicture();
        unawaited(_uploadFile(File(shot.path)));
        unawaited(_runOnDeviceCheck(File(shot.path), _classLabel));
      } else {
        // ... gallery fallback unchanged
      }
```

The fork is intentional: upload and prediction run independently. The
late-arriving prediction overwrites the success toast — this matches
the spec's "toast morphs" behavior. The user can ignore a false-flag
warning or tap Undo.

- [ ] **Step 4: Build and verify**

Run: `cd flutter && flutter analyze && flutter run -d <device>`

Verify (use connectors with already-trained classes, e.g., 2.4mm-M):
- Capture a 2.4mm-M frame with the chip set to 2.4mm-M — toast shows green agreement.
- Set chip to 2.4mm-F and shoot the same male connector — toast shows yellow gender-mismatch warning.
- Set chip to SMA-M and shoot the same 2.4mm-M connector — toast shows red full-class warning.
- Tap Undo while the warning is up — file is deleted server-side.

- [ ] **Step 5: Commit**

```bash
git add flutter/lib/src/screens/contribute_screen.dart
git commit -m "flutter/contribute: on-device label-check toast (non-blocking)"
```

---

## Task 10: End-to-end smoke + deploy

- [ ] **Step 1: Deploy server changes to the box**

```bash
git push
echo '<sudo-pwd>' | ssh chris@192.168.20.235 \
  'sudo -S sh -c "sudo -u rfcai git -C /opt/rfcai/training pull --ff-only \
                  && systemctl restart rfcai-predict"'
```

- [ ] **Step 2: Verify the deployed labeler**

```bash
curl -s -u $LABELER_USER:$LABELER_PASS https://aired.com/rfcai/labeler/stats | jq .
```

Expected: `{"train": {...}, "holdout": {...}}` with the 10 canonical
classes. The four currently-empty classes (SMA-M/F, 1.85mm-M/F) read
0/0.

- [ ] **Step 3: Sideload the Flutter build**

```bash
cd flutter
flutter build apk --release       # Android
# or
flutter build ipa --release       # iOS
```

Install on the device (Android: `adb install`; iOS: TestFlight / Xcode).

- [ ] **Step 4: Run the field smoke checklist**

For each capture path:
- Single still capture → server total increments in the stats sheet.
- Video upload of a female connector → file lands in
  `data/labeled/embedder/<family>-F/`, not `-M/`. Verify by
  refreshing the stats sheet.
- Mis-tap → Undo → counter decrements, file gone server-side.
- On-device check fires within ~1s of shutter; agree path is green,
  disagree path is red/yellow with Undo callout still tappable.

- [ ] **Step 5: Commit any field-discovered fixes and tag**

```bash
git tag -a contribute-capture-improvements -m "Field-tested on Android + iPhone"
git push --tags
```

---

## Self-review against spec

| Spec section | Plan tasks |
|---|---|
| §1 Per-class progress visibility | Tasks 1, 2, 5, 6 |
| §2 Video upload with gender | Tasks 3, 5, 7 |
| §3 Undo last upload | Tasks 4, 5, 8 |
| §4 On-device label-check | Task 9 |
| Error handling table | Captured per-task (auth tests, JSON validation, undo restore) |
| Testing — server | Tasks 1–4 each include pytest cases |
| Testing — client | Task 5 extends `api_test.dart` for JSON parsers; Tasks 6/8/9 use manual verification (camera widget tests skipped per Flutter test-cost trade-off) |
| Out-of-scope items | None added — plan stays inside the spec's scope |

No placeholders. No "TBD" anywhere. All type names referenced in later
tasks (`UploadResult`, `UploadRecord`, `LabelerStats`, `_SessionRecord`)
are defined in earlier tasks.
