# Contribute Screen — Capture-Session Improvements

**Date:** 2026-05-14
**Goal:** Reduce friction in the Flutter Contribute flow before a bulk
training-data capture session, so the user can shoot dozens to hundreds of
labeled images per phone session without leaving the camera screen to
manage class balance, fix mislabels, or recover from fat-finger taps.

## Motivation

A shipment of physical connectors just arrived and the user is about to
generate the first large batch of phone-captured training data. The
current Contribute screen is already camera-first with background
uploads, sticky chips, holdout toggle, video upload, and a session
counter — but the workflow has four specific friction points that bite
hardest when capture volume is high:

1. The phone gives no per-class progress signal, so the user has to
   guess which classes are starved (per the runbook, SMA-M/F and
   1.85mm-M/F currently have **zero training samples** — a balance
   problem that's invisible from the phone).
2. The video-upload endpoint hardcodes `<family>-M`, so any video of a
   female connector silently lands in the wrong class folder and has to
   be flipped manually in the labeler UI.
3. There is no in-app way to undo a fat-finger gender/family mistake —
   recovery requires switching to Streamlit on a laptop.
4. There is no real-time mislabel detection at capture time, even
   though a Tier-1 ONNX classifier is already loaded on the phone for
   the Identify screen.

## Scope

In scope:
- One new labeler-server endpoint for stats.
- One labeler-server endpoint signature change (`upload-video` gains
  `gender`).
- Response-format change on two existing endpoints (`upload-train`,
  `upload-test`) from HTML to JSON so the client can stash the saved
  server-side path.
- Four Flutter UI changes on `contribute_screen.dart` (per-class stats
  bottom-sheet, gender threaded through video upload, Undo small-action,
  on-device label-check toast).
- No changes to the predict service, ingestion daemon, or training
  pipeline.

Out of scope (intentionally deferred):
- ArUco / capture-quality hints in the app
- In-app blur/exposure detection
- Burst-mode capture
- Auto-class detection from the marker

## Architecture overview

```
                 +---------------------------+
                 |  contribute_screen.dart   |
                 |---------------------------|
                 |  + per-class stats sheet  |
                 |  + Undo stack             |
                 |  + on-device sanity check |
                 +-----+-----------+---------+
                       |           |
              (1) tap pill         |
                       v           |
              GET /labeler/stats   |  (every shot)
                       |           |
                       |  +--------v---------+
                       |  | OnDeviceClassifier|
                       |  |  (already exists,|
                       |  |   singleton)     |
                       |  +------------------+
                       |
                       |  (2) shutter
                       |  POST /labeler/upload-train  (JSON resp now)
                       |  POST /labeler/upload-test   (JSON resp now)
                       |  POST /labeler/upload-video  (+gender now)
                       |
                       |  (3) Undo
                       |  POST /labeler/delete  (existing)
                       v
              +-----------------+
              |  labeler.py     |
              |  (FastAPI router)|
              +-----------------+
                       |
                       v
              data/labeled/embedder/<cls>/photo_*.jpg
              data/test_holdout/<cls>/*.jpg
```

## Section 1 — Per-class progress visibility

### Server

New endpoint: `GET /rfcai/labeler/stats` (basic auth).

Returns:

```json
{
  "train": {"2.4mm-M": 47, "2.4mm-F": 31, "SMA-M": 0, "SMA-F": 0,
            "1.85mm-M": 0, "1.85mm-F": 0, "2.92mm-M": 22, "2.92mm-F": 18,
            "3.5mm-M": 26, "3.5mm-F": 25},
  "holdout": {"2.4mm-M": 1, "2.4mm-F": 1, "...": 0}
}
```

Counting rule: for each canonical class, count files in
`data/labeled/embedder/<cls>/` and `data/test_holdout/<cls>/` whose
basename does **not** match any of the synth/derivative patterns
(`*_clean.*`, `*_bg*.*`, `*_z*.*`, `*_central.*`, `synth_*.*`). Files in
`_quarantine_*` folders are not counted (they live alongside, not
under, the class folder).

The intent is to surface "real human captures" — the number the user is
actually moving with each shutter tap — not the augmented training-set
size.

Cost: ~10 × `Path.iterdir()` calls; no caching needed. Endpoint is
read-only and auth-gated like the rest of the labeler.

### Client

The counter pill in the top-left of the Contribute screen becomes
tappable. Tap opens a Material `showModalBottomSheet` with a table:

```
Class       | Session | Server
2.4mm-M     | 5       | 52
2.4mm-F     | 0       | 31
SMA-M       | 0       |  0 🔴
SMA-F       | 0       |  0 🔴
1.85mm-M    | 0       |  0 🔴
1.85mm-F    | 0       |  0 🔴
...
```

- Red dot on any class with server total <5 (matches the
  `MIN_SAMPLES_PER_CLASS` threshold the auto-retrain uses to keep
  classes out of the trained head).
- "Pull to refresh" re-fetches `/stats`.
- Camera stays live underneath. Closing the sheet is a swipe-down.

Session counts live in `Map<String, int> _sessionCountsByClass`,
incremented on each successful upload. Holdout uploads tally separately
in a parallel map.

## Section 2 — Video upload with gender

### Server

`/upload-video` gains required form field `gender: str` and validates
`f"{family}-{gender}"` against `CANONICAL_CLASSES`. Replaces:

```python
target_cls = f"{family}-M"
```

with:

```python
target_cls = f"{family}-{gender}"
if target_cls not in CANONICAL_CLASSES:
    raise HTTPException(400, f"unknown class {target_cls!r}")
```

### Client

`ApiClient.uploadTrainingVideo` signature becomes
`(File video, String family, String gender)`. The Contribute screen
passes the currently-selected chips for both. No UI affordance changes —
the chips are already shown.

## Section 3 — Undo last upload

### Server

Two response-format changes (HTML → JSON):

**`/upload-train`**

Before: HTML success/error message.

After:

```json
{
  "saved": [
    {"cls": "2.4mm-M",
     "path": "data/labeled/embedder/2.4mm-M/photo_IMG_1234.jpg"}
  ],
  "errors": []
}
```

**`/upload-test`** — identical shape, paths under `data/test_holdout/<cls>/`.

The HTML responses on these endpoints are not consumed by any current
caller (the labeler grid UI uses its own endpoints, not these), so the
change is safe. Verify by `git grep` before shipping.

The existing `POST /labeler/delete` endpoint is reused unchanged — it
takes a `path` form field and unlinks via `_safe_path`.

### Client

State changes on `_ContributeScreenState`:

```dart
final List<_UploadRecord> _undoStack = [];

class _UploadRecord {
  final String path;       // server-side path, authoritative
  final String cls;
  final bool holdout;
}
```

On a successful `_uploadFile` or `_uploadBytes`, parse the JSON, append
each saved record. Cap stack at 50 entries (FIFO eviction).

New small-action `Undo` between Gallery and Video, disabled when stack
is empty, label shows the count: `Undo (3)`.

Tap behavior — single tap, no confirmation prompt:

1. Pop tail of stack: `last = _undoStack.removeLast()`.
2. POST `/labeler/delete` with `path: last.path`.
3. On success: decrement `_uploadedCount`, decrement the per-class
   session count, show toast: `↩ Undone 2.4mm-M`.
4. On failure: re-append `last` to stack, show error toast.

Rationale: confirmation prompts add friction the user has explicitly
asked us to minimize. The blast radius of an accidental Undo is one
re-shot photo, which is cheap during a bulk capture session. The
button label shows the stack size (`Undo (3)`) so the user has a
visible count before tapping.

Undo is class-agnostic — server returns the path; client just sends it
back. The stack survives chip changes; the user can shoot 2.4mm-M,
flip the chip to 2.92mm-F, shoot, then Undo and the 2.92mm-F shot
goes — not the 2.4mm-M shot. Tail order, not class-scoped.

## Section 4 — On-device label-check

### Flow

When a photo is captured:

1. Upload fires (unchanged, background).
2. **In parallel:** `OnDeviceClassifier.instance.predict(file)` returns
   `OnDevicePrediction` in ~200–500ms on a modern phone.
3. When prediction returns and the upload has acked:
   - `predClass == selectedClass` →
     toast: `✓ #N 2.4mm-M (agree 0.92)` (green, 2s)
   - `predFamily == selectedFamily` but gender differs →
     toast: `⚠ picked M, model says F (0.71)` (yellow, 4s) +
     inline `Undo` button
   - `predFamily != selectedFamily` or top-1 conf <0.4 →
     toast: `⚠ model says SMA-F (0.55)` (red, 4s) +
     inline `Undo` button

The check is **post-hoc and non-blocking** — the upload always
completes. The user decides whether to Undo. False positives from the
on-device model (which has a known accuracy gap vs the server ensemble)
are tolerated; the user can ignore the warning. Confidence threshold
0.4 is conservative so we don't yellow-flag every shot.

### Initialization

`OnDeviceClassifier.instance` is currently lazy-loaded by
`identify_screen.dart`. Preload it inside `_initCamera()` on Contribute
so the ~1s ONNX session init overlaps with camera startup. The singleton
ensures we don't double-load when both screens have been visited.

If the model file is missing or loading fails, the on-device check is
silently disabled — the toast falls back to the current
`✓ #N 2.4mm-M` text with no agree/disagree decoration. Upload behavior
is unaffected.

## Data flow per shutter tap

```
User taps shutter
  ├── camera.takePicture() → File f
  ├── (fork)
  │   └── api.uploadTrainingPhoto(f, cls) (async)
  │       ├── on 200: parse JSON, push _UploadRecord(path, cls, false)
  │       │            increment session counters
  │       │            emit toast (then merged with on-device result)
  │       └── on err: error toast
  └── (fork)
      └── OnDeviceClassifier.predict(f) (async)
          └── on result: emit/merge agree-or-disagree toast
```

Both forks finish independently. The toast is updated when the upload
response arrives, then updated again when the on-device prediction
arrives. A small ID ties them together so a late-arriving prediction
doesn't decorate the wrong shot's toast.

## Error handling

| Failure | Behavior |
|---|---|
| Stats endpoint 4xx/5xx | Bottom-sheet shows last-known counts + error pill; pull-to-refresh retries |
| Upload 4xx/5xx | Existing toast; record not pushed to Undo stack |
| Undo `/delete` 4xx/5xx | Record re-appended to stack, error toast |
| Video upload missing gender (400 from new validation) | Client validation catches before send; if server returns 400, current friendly-error toast applies |
| On-device classifier missing/fails | Silent fallback to current toast |
| Network offline | All endpoints fail with friendly-error toast; user can keep shooting (records lost — same as today) |

## Testing

**Server:**
- Add unit/integration tests for `/labeler/stats` (counts match a
  hand-built fixture directory; synth files excluded).
- Add test for `/upload-video` 400 on missing or invalid `gender`.
- Add test for `/upload-train` and `/upload-test` JSON response shape.
- Regression check: `_safe_path` still rejects path-traversal for delete.

**Client:**
- Widget test: tapping counter pill opens bottom-sheet; rows render.
- Widget test: Undo stack populates on successful upload, pops on Undo
  tap, re-pushes on delete failure.
- Widget test: on-device disagree path shows yellow/red toast.

**Manual:**
- Capture 10 photos across 3 classes on a real device, verify session
  counts in bottom-sheet.
- Capture, then Undo immediately, verify file gone from server (check
  via labeler grid).
- Capture a video of a 2.4mm-F connector, verify it lands in `2.4mm-F/`
  not `2.4mm-M/`.

## Non-goals confirmed out-of-scope

- ArUco-marker presence detection in-app
- Blur/exposure detection
- Burst-mode capture
- Auto-class detection from the marker
- Sync across Android + iPhone — each phone is a self-contained session
- Multi-tier offline queueing — current "fire-and-forget, lose on
  network failure" behavior is unchanged

## Decisions to confirm in review

- Stats count rule: real human captures only (exclude synth/derived
  variants, exclude quarantine). Adopted per recommendation.
- Undo works for both training and holdout uploads, tail-ordered across
  classes. Adopted per recommendation.
- Undo is single-tap, no confirmation prompt. See rationale in Section 3.
- On-device label-check confidence threshold = 0.4 (below this, flag as
  disagree even if family matches). Tunable; defaulted conservatively.
