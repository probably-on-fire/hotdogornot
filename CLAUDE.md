# RF Connector AI — Claude session context

You are picking up an active project. Read this first. Most of this is current as of **2026-05-25**; the **2-way ensemble investigation** below was added 2026-05-26.

## Elevator pitch

Phone app that identifies RF coaxial connectors (SMA, 1.85mm, 2.4mm, 2.92mm, 3.5mm in male/female). Companion web app at **aired.com** for managing training data and granting collaborator logins. Closed-signup multi-user auth.

The Flutter app (`flutter/`) and the Python service (`training/rfconnectorai/server/`) talk to each other through aired.com. The actual ML runs on a LAN box that reverse-SSH-tunnels into aired.com's public-facing nginx.

## Current classifier state (2026-05-25, post-deploy)

Production now serves **`combined_v2` hybrid** (Jerry YOLO + our EffNetV2-S trained on Jerry's 638 photos + ~2,913 quality-filtered video keepers). Deployed 2026-05-25.

- **Production: combined_v2 hybrid** — `RFCAI_USE_JERRY_PIPELINE=1` + `RFCAI_JERRY_MODEL_DIR=/home/rfcai/training/models/jerry_v19_hybrid_combined_v2`. **81.4% Full / 86.0% Family / 90.7% Gender** on the cleaned 43-image holdout. ~155ms/image on CPU. Pre-deploy env backup at `/etc/default/rfcai-predict.bak.2026-05-25` on the box.
- **Legacy (rollback target):** Hough/edge-density + ResNet-18. 39.5% Full / 74.4% Gender on the cleaned holdout, ~5-7s/image. Rollback: `sudo cp /etc/default/rfcai-predict.bak.2026-05-25 /etc/default/rfcai-predict && sudo systemctl restart rfcai-predict`.
- **Jerry's pretrained (prior benchmark):** 81.4% Full / 88.4% Gender on the cleaned holdout. Combined_v2 ties on Full but beats him on Family/Gender. The 97.1% in earlier notes was on a stale, smaller 35-image holdout — NOT the current set.

**Known gaps:**
- 8 of 43 holdout misses; 7 cluster in the 2.4/2.92/3.5 fine-pitch family-confusion zone.
- 2.92mm-M class regressed 6/7→4/7 in combined_v2 vs jerry_full_nobal — likely SMA-F dominating the unbalanced training (925 SMA-F samples vs 151 2.92mm-F). A balanced retry is the obvious next experiment.
- The IMG_02XX OOD camera batch (~6 samples) defeats both Jerry's pretrained AND combined_v2 — different camera/lighting than the bulk training data.

See [[combined-v2-deploy]] memory + `docs/session_jerry_replication_2026-05-25.md` for the full investigation.

## Overnight 2-way ensemble investigation (2026-05-26, **NOT YET DEPLOYED**)

Softmax-averaging `combined_v2` + `jerry_full_nobal` (both already on the box, both ~80% single-model) lifts the cleaned 43-img holdout to **88.4% Full / 90.7% Family / 95.3% Gender** — +7.0pt Full / +4.6pt Gender over current prod. Crucially, **at the app's 0.65 confidence threshold the ensemble is 100% accurate on 31/43 confident predictions** — every one of the 5 misses falls below threshold (0.308, 0.388, 0.495, 0.495, 0.585) and abstains. Zero confidently-wrong outputs. See [[ensemble-confidence-calibrated]].

- **Code change is local + uncommitted.** `training/rfconnectorai/pipeline/jerry_pipeline.py` (new `extra_model_dirs` kwarg) + `training/rfconnectorai/server/predict_service.py` (new `RFCAI_JERRY_EXTRA_MODEL_DIRS` env var, comma-separated bundle dirs). Backward-compatible: empty env → single-model 81.4% reproduces exactly.
- **Deploy runbook:** `docs/deploy_2way_ensemble_runbook.md`. Tested end-to-end against the live box.
- **Things that did NOT help (all tried 2026-05-26):**
  - TTA (hflip + vflip + rot±5 + rot±10 in any combo) — +5pt on single model, 0pt on top of the ensemble. See [[tta-redundant-atop-ensemble]].
  - Balanced retrain of combined_v2 — `--balance-to-smallest` capped at 1,289 samples vs 3,555 → 74.4% Full alone, and **regressed** the 2-way to 86.0% when added as a 3rd. The 6.1× class ratio is *below* the threshold where balancing helps. See [[two-way-ensemble-winner]].
  - 3-way ensembles with any of `jerry_full`, `jerry_v3`, `jerry_replica`, `photos_only` as a 3rd member — tied or regressed.
- **Memories added:** [[two-way-ensemble-winner]], [[ensemble-support-jerry-pipeline]], [[tta-redundant-atop-ensemble]], [[ensemble-confidence-calibrated]], [[miss-diagnosis-2way]], [[combined-photos-dilutes-ensemble]], [[tight-crop-no-lift]], [[two-2way-ensembles-share-misses]], [[holdout-has-mislabeled-duplicate]].
- **Holdout has 1 mislabeled duplicate** (discovered 2026-05-26 04:00am-ish): `2.4mm-M/2_4mm-m.jpeg` and `2.92mm-M/IMG_0274.jpeg` are byte-identical — same MD5, same file size. The "43-img holdout" is really 42 unique images. Removing the wrong copy → **90.5% Full** without any retraining. See [[holdout-has-mislabeled-duplicate]]. *Cheapest accuracy bump available.*
- **Best-CLS-conf box selection** (discovered 2026-05-26 ~04:30am): the YOLO detector picks bad crops on some OOD images (right-edge strips, etc.). Classifying ALL detected boxes (box_min=0.05, top 8) and picking the one with highest classifier confidence lifts the 2-way ensemble from **88.4 → 90.7% Full** and **95.3 → 100% Gender**, with zero regressions on currently-correct images. Combined with the dupe fix → **92.9% Full / 100% Gender**. Latency cost: ~400-500ms/image (vs current ~310ms). See [[best-cls-conf-box-selection]].


The Flutter app (commit `c034312`) also added a **reticle-crop UX**: user fits the connector inside a centered circle, app crops to a 60% centered square before upload. Train and inference now share scale. Confidence threshold tightened 0.40 → 0.65. See [the memory](MEMORY.md) for the rationale.

**Holdout was cleaned 2026-05-25:** 5 mislabeled samples (IMG_02xx batch) were moved to their true classes via Sonnet vision audit. Backup at `/tmp/holdout_pre_relabel_2026-05-25/` on the box. See [[holdout-quality-issues]] memory.

**Datasets admin UI deployed 2026-05-25:** `https://aired.com/rfcai/labeler/admin/datasets` for browsing + bulk-editing every dataset on disk. Live but uncommitted locally — see [[datasets-admin-ui]] memory. SFTP'd patches to: `labeler.py`, `templates/labeler/_base.html`, `admin_datasets.html` (new), `admin_dataset_grid.html` (new).

## Investigation session 2026-05-25: "why is Jerry's fork so much better than ours"

Spent a session systematically replicating Jerry's pipeline + isolating the gap. Findings:

- **Pipeline (architecture, preprocessing, training code) is identical** between his fork and ours. Confirmed by diff of his `train.py` vs ours.
- **Training data is the differentiator**, not the recipe. Our 23K-sample training set was 95% redundant video-frame augmentations of ~600 unique scenes; Jerry's 638 hand-curated photos = 638 unique scenes.
- **Critical recipe gotchas**: (1) Jerry does NOT use `--balance-to-smallest` on his already-balanced data — using it cost us 12pts. (2) `RandomResizedCrop(scale=(0.55, 1.0))` floor is too high for tight Hough crops — photo-trained models score 16% standalone but 79% in YOLO hybrid. See [[crop-scale-mismatch]] memory.
- **The 97.1% baseline was stale** — it was on the older 35-img holdout. Both Jerry's pretrained and our replica score ~79-81% on the current 43-img holdout.
- **6 of remaining ~9 holdout misses are an OOD `IMG_02XX` batch** that neither training set saw. Adding samples like those to training would close most of the remaining gap.

Full recipe + numbers: see [[jerry-replication-recipe]] memory.

## Immediate goal of this session

**Build and install the iOS app on the user's iPhone, smoke-test the flow, then commit any iOS-specific fixes.**

See `flutter/ios/README.md` for the canonical iOS build steps. Quick version:

```bash
git pull
cd flutter
flutter pub get
cd ios && pod install && cd ..
open ios/Runner.xcworkspace
# Xcode → Signing & Capabilities → set Team to your Apple ID.
# Plug iPhone, pick it as destination, hit ▶.
# On iPhone: Settings → General → VPN & Device Management → trust dev cert.
```

iOS deployment target is **13.0** (required by `flutter_secure_storage 9.x` + `onnxruntime 1.x`). `Info.plist` already has camera/photo-library/microphone usage strings.

## Credentials (you need these to do almost anything)

### SSH access

| Host | User | Password | Notes |
|---|---|---|---|
| `aired.com` | `chris` | `Elad9651!` | Public-facing nginx + reverse-tunnel landing. Static files at `/var/www/aired/html/`. |
| `192.168.20.235` (the box, LAN only) | `chris` | `Elad9651!` | Runs predict service, labeler, training. Repo at `/opt/rfcai/training`. |

Sudo password is the same as login on both. SSH from a no-TTY tool: `echo 'Elad9651!' \| ssh ... 'sudo -S ...'` — use single quotes around the password so the `!` doesn't trigger bash history expansion.

### Labeler web users (closed signup, all admin role)

- `chris` / `Elad9651!`
- `jdcrunchman` / `4AA_8ZOJ1T6jybQDK1OVua9Z`
- `zapperman` / `HAHGObgH7xjlvIyyA6gD7Vjo`

Stored hashed (scrypt) in `/opt/rfcai/repo/training/data/labeler_users.db`. Rotate via the admin UI or `scripts/seed_labeler_users.py --force`.

### Predict service device token (legacy, anonymous Identify-tab uploads)

`66c72f6b1495e406d8b69f8a569c2d57d67614cdc63235f8c7f4c072f4fea4e1`

### Session secret on the box

In `/etc/default/rfcai-predict`: `RFCAI_SESSION_SECRET=...` (generated 2026-05-15). Don't change without invalidating all sessions.

## Public URLs

| URL | What |
|---|---|
| `https://aired.com/` | Landing page (hero + live stats from /labeler/stats + CTAs) |
| `https://aired.com/rfcai/labeler/` | Public training-data grid view |
| `https://aired.com/rfcai/labeler/login` | Web sign-in |
| `https://aired.com/rfcai/labeler/admin/users` | Admin user management (signed in only) |
| `https://aired.com/rfcai/labeler/stats` | Per-class counts JSON (public) |
| `https://aired.com/rfcai/labeler/snapshots` | List of dataset tarball downloads |
| `https://aired.com/rfcai/labeler/snapshots/rfcai_session_2026-05-14.tar.gz` | First session snapshot (249 MB) |
| `https://aired.com/rfcai/predict` | Predict API (X-Device-Token auth) |
| `https://aired.com/rfcai/labeler/api-tokens/exchange` | POST username+password → Bearer token (for Flutter app) |
| `https://aired.com/demo/` | Streamlit demo |

## Architecture in one minute

Two physical servers:

- **aired.com** — public-facing. Nginx serves `/var/www/aired/html/` and proxies `/rfcai/*` + `/demo/` to the box via reverse SSH tunnels.
- **lrn-ai-serv1** (`192.168.20.235`, LAN-only) — runs the FastAPI predict + labeler service, Streamlit demo, ingestion daemon, nightly retrain.

Box services (systemd, owned by user `rfcai`):

- `rfcai-predict.service` — FastAPI app on :8503. Single process serves /predict + /labeler/*.
- `rfcai-predict-tunnel.service` — reverse SSH to aired.com:8504.
- `rfcai-streamlit.service` — :8501 → tunnel → aired.com:8502.
- `rfcai-ingestion-daemon.service` — pulls phone uploads from the relay.
- `rfcai-auto-retrain.service` (.timer-driven) — nightly retrain.

Repo on the box: `/opt/rfcai/training` (symlinked to `/opt/rfcai/repo/training/`). Pull with `sudo -u rfcai git -C /opt/rfcai/training pull --ff-only`.

## Standard deploy commands

**Server (Python/template changes):**

```bash
echo 'Elad9651!' | ssh chris@192.168.20.235 \
  'sudo -S sh -c "sudo -u rfcai git -C /opt/rfcai/training pull --ff-only && systemctl restart rfcai-predict"'
```

Service takes 30-60s to warm up after restart (loads classifier + rembg + YOLO models). Poll `curl -sf https://aired.com/rfcai/labeler/stats` until it returns 200.

**aired.com static files (`aired_site/index.html`):**

```python
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("aired.com", username="chris", password="Elad9651!",
          allow_agent=False, look_for_keys=False)
c.open_sftp().put(r"E:\anduril\aired_site\index.html",
                  "/var/www/aired/html/index.html")
```

(On the Mac, use a posix path instead of the Windows raw string.)

**Android APK:**

```bash
cd flutter
flutter build apk --release
# Output: flutter/build/app/outputs/flutter-apk/app-release.apk
```

To share with a phone over local Wi-Fi: `python -m http.server 8000` from the APK directory; phone visits `http://<PC-IP>:8000/app-release.apk`.

## Repo layout

```
flutter/                    Flutter app (Android + iOS)
  lib/src/
    screens/
      identify_screen.dart  Camera live preview + /predict API. Pinch zoom,
                            reticle overlay, post-capture bbox painter.
      contribute_screen.dart Capture flow (auth-gated). Same reticle + zoom.
                            Camera uploads are now reticle-cropped JPEGs
                            (filename: reticle_crop.jpg).
      about_screen.dart     App info + on-device toggle
      main_shell.dart       Bottom-nav with three tabs
      login_screen.dart     (none — login is a card inside contribute_screen)
    widgets/
      reticle.dart          Shared centered target circle (28% min-dim
                            radius). Used by Identify + Contribute.
    auth.dart               AuthService (ChangeNotifier, Bearer tokens)
    api.dart                ApiClient — predict + labeler endpoints
    settings.dart           Relay URL + on-device toggle (NO creds anymore)
    ondevice/classifier.dart Bundled ONNX ResNet-18 inference
  assets/
    icon/icon.png           AI Red brain logo (used on About + launcher)
    models/                 connector_classifier.onnx + labels.json
  ios/
    README.md               iOS build guide (start here on Mac)
    Runner/Info.plist       Camera/photo permissions wired

training/
  rfconnectorai/server/
    labeler.py              All labeler routes (~40 endpoints)
    predict_service.py      FastAPI app + SessionMiddleware + plugin glue.
                            Reads RFCAI_USE_JERRY_PIPELINE to route through
                            jerry_pipeline.py instead of Hough+ResNet18.
    auth.py                 Users + API tokens + scrypt (stdlib only)
    templates/labeler/
      _base.html            Shared chrome (CSS variables, header, nav)
      index.html            Grid view
      login.html            Sign-in card
      admin_users.html      User management UI
  rfconnectorai/pipeline/
    jerry_pipeline.py       YOLO11n + EfficientNetV2-S ONNX inference.
                            Ported from trextrader/hotdogornot. Uses
                            PIL.BILINEAR (NOT cv2.INTER_LINEAR — see the
                            preprocessing note below).
    detect_classify.py      Legacy multi-head training-time pipeline.
    predict_cli.py          Batch CLI for the multi-head model.
  scripts/
    seed_labeler_users.py   Seeds 3 admin users with random passwords
    auto_retrain.py         Training loop (--data-dir --model-dir --epochs)
    eval_holdout.py         Hits /predict against a directory; markdown report
    _kick_retrain.sh        One-line retrain helper used on the box
  tests/                    pytest suite — 58 tests pass, 1 skipped (imagehash)
  docs/
    runbook.md              Operational reference
    capture_protocol.md     How to shoot training data well
    classifier_journey.md   Why the pipeline is the way it is

aired_site/                 Static HTML for aired.com root landing page
  index.html                Hero + live stats + about + footer
  README.md                 Deploy command (SFTP via paramiko)

docs/
  superpowers/
    specs/                  Design docs from completed work
    plans/                  Implementation plans
  runbook.md, etc.          Various operational + architectural notes
```

## Recent work (2026-05-23 → 2026-05-25)

1. **Jerry replication study** — systematically isolated why Jerry's fork outperformed ours. Trained 7+ model variants. Ended at parity (~79% Full / 95% Gender) with Jerry's pretrained when using his exact recipe + his exact data. Full saga: `docs/session_jerry_replication_2026-05-25.md` + [[jerry-replication-recipe]] memory.
2. **`train_v19_effnet.py` patched** — added Jerry's exact recipe defaults (epochs=50, batch=16, lr=1e-4, save-best-val-acc checkpoint, crash-safe weights_last.pt per epoch). Standalone trainer at `training/scripts/train_v19_effnet.py`.
3. **Holdout cleaned** — 5 mislabeled samples (`2_4mm-m.jpeg`, `IMG_0271`, `IMG_0272`, `IMG_0274`, `IMG_0276`) moved to true classes via Sonnet vision audit. The 97.1% Jerry baseline in earlier notes was on a smaller, cleaner 35-img holdout. Current cleaned 43-img holdout has Jerry at 81.4% Full, ours at 79.1% Full.
4. **Datasets admin UI** — `/labeler/admin/datasets` lists every dataset under `data/`. Per-dataset grid view with multi-select bulk-delete + bulk-move-to-class. Test holdout flagged read-only. Bulk-delete hardlinks to `data/_deleted/` for recovery. SFTP-deployed to box; uncommitted locally. See [[datasets-admin-ui]] memory.
5. **Sonnet angle-filter pass** — 40 of 92 batches (1,600 of 3,649 crops) classified as side-profile vs head-on. Most video crops are head-on (wrong distribution for app where users photograph side-profile). Halted at 40 batches once jerry_full_nobal proved data composition mattered more than angle-filtering existing videos.
6. **Models on box** under `/home/rfcai/training/models/`: `classifier_v19_effnet_jerry_full_nobal/` (best replica), `classifier_v19_effnet_photos_only/`, `classifier_v19_effnet_jerry_replica/`, `classifier_v19_effnet_jerry_v3/`, hybrid dirs at `jerry_v19_hybrid_*/`. Combined-dataset training (511 + 638 dedupe) running as of writing.

## Recent work (2026-05-17 → 2026-05-18)

1. **35-image carved holdout** — moved 5 photo_* files per class from train → holdout for 7 classes via the labeler API (see `tmp_carve.py` + `tmp_carve_execute.py`). Holdout grew 8 → 43. Gaps: 3.5mm-M and 2.4mm-F have no photo_* in train (only synthetic), 2.4mm-M labeler /grid hung on enumeration. Carve scripts kept the `source_backup/` hardlinks so nothing is actually lost.
2. **Baseline v18 eval (clean)** — 68.6% Full / 91.4% Gender on the 35 new images (`tmp_baseline_eval.md`). v18 was trained 2026-05-05; carved images are dated 2026-05-14+, so this is honest out-of-training-set performance, not inflated.
3. **Partner pipeline benchmark** — `trextrader/hotdogornot`'s YOLO11n + EfficientNetV2-S scored **94.3%** on the same 35 images (`tmp_partner_eval.md`), 40× faster latency.
4. **Pipeline ported into our repo** — `training/rfconnectorai/pipeline/jerry_pipeline.py` is a Python implementation of Jerry's `exports/web/app.js`. Routes through `predict_service.py` when `RFCAI_USE_JERRY_PIPELINE=1` is set in `/etc/default/rfcai-predict`. Local sanity test (`tmp_jerry_pipeline_eval.py`): **97.1%** on our 35-image holdout. Critical preprocessing gotcha: use PIL.BILINEAR, not cv2.INTER_LINEAR — costs 14pts otherwise.
5. **Flutter reticle-crop UX** — pinch-to-zoom + shared `widgets/reticle.dart` overlay on both Identify and Contribute. Captured photos are cropped to a centered 60%-of-min-dim square ON-DEVICE before upload, so train and inference share scale. Filename for camera uploads: `reticle_crop.jpg`. Tightened `_kMinAcceptedConfidence` 0.40 → 0.65 in identify_screen.dart. Post-capture: detector bbox painted on the frozen frame (green = accepted, amber = below threshold).
6. **Runbooks** — `docs/backend_swap_jerry_pipeline_runbook.md` (deploy Jerry's pipeline on the box, ~10 min on LAN) and `docs/v20_tighter_crop_runbook.md` (alternative tighter-crop experiment if you ever want to claw back accuracy on the legacy ResNet-18 path).

## Recent work (last session, 2026-05-14 → 2026-05-15)

1. **Contribute screen rebuild** — per-class session counters with bottom-sheet, Undo stack (single-tap, 500-entry cap), on-device classifier check toast (serialized to prevent OOM under burst), session-prefixed filenames (`photo_YYYY-MM-DD_*`), video uploads now require gender.
2. **Server-side hardlink backup** — every `/upload-train` and `/upload-test` also hardlinks to `data/source_backup/<cls>/`. Deletion from the working dir leaves the backup intact (different inode-shared name).
3. **Multi-user auth** — replaced env-var Basic auth with a real users table (SQLite + scrypt). Three roles, but currently all seeded as admin. Cookie-based web login + Bearer tokens for API/Flutter. HTTP Basic still works as a transitional fallback (Task 9 will remove).
4. **Labeler grid public reads** — anonymous visitors can browse + view + download snapshots. Writes (upload, delete, flip) are admin-only.
5. **Snapshot endpoint** — `/labeler/snapshots/*` serves pre-built dataset tarballs for forks (Jerry has the first one).
6. **aired.com landing page** — replaced minimalist single-link homepage with a real landing (hero + live stats + CTA to labeler + footer). Real AI Red brain logo image; previously a blue RF-mark SVG stand-in.
7. **About screen polish** — uses the actual app icon (red brain), dropped the Advanced (relay/token) section, on-device toggle now always visible above the footer.
8. **Login screen polish** — branded card, slow pulse on the brand mark, closed-group help text, "← Back to aired.com" link.
9. **iOS deployment target** bumped to 13.0 (`flutter_secure_storage` requirement).

## Pending work

- **Deploy the 2-way ensemble** (highest leverage) — see `docs/deploy_2way_ensemble_runbook.md`. Commit the local `jerry_pipeline.py` + `predict_service.py` changes, push, pull on the box, append the env var, restart service. +7pt Full / +4.6pt Gender, 5 misses left (3 below abstain), no new training required. Rollback is removing one env var line.
- **Combined-dataset training result** (in flight) — our 511 photos + Jerry's 638 deduped, training with Jerry's recipe. If it beats 81.4% Full on the cleaned holdout, that's the new production candidate.
- **Commit the datasets UI changes** — SFTP'd to box but uncommitted. Files: `training/rfconnectorai/server/labeler.py`, `templates/labeler/_base.html`, `templates/labeler/admin_datasets.html` (new), `templates/labeler/admin_dataset_grid.html` (new).
- **Commit the train_v19_effnet.py patches** — Jerry-recipe defaults + save-best logic. File: `training/scripts/train_v19_effnet.py`.
- **Deploy Jerry's pipeline to the box** — runbook at `docs/backend_swap_jerry_pipeline_runbook.md`. Needs LAN access to scp ONNX files + set env vars in `/etc/default/rfcai-predict`. Existing `/predict` API contract unchanged — Flutter app gets the win automatically.
- **Build + test new Flutter APK** — commit `c034312` adds reticle crop, pinch zoom, tighter abstention, bbox overlay. Pull on Mac, `flutter pub get`, `flutter analyze`, `flutter build apk --release` (Android) or `pod install` + Xcode (iOS).
- **Capture more reticle-cropped training data** — once new APK is on phones, every Contribute upload is a `reticle_crop.jpg`. Prioritize 2.4mm-F (now 0 holdout samples after cleanup), 3.5mm-M (only 2 holdout samples), and the IMG_02XX-style camera distribution that both models can't recognize.
- **Task 9 cleanup pass** (still queued) — remove the HTTP Basic auth fallback from `require_admin`, delete `_require_basic_auth`, remove `LABELER_USER`/`LABELER_PASS` env vars from `/etc/default/rfcai-predict`, add `itsdangerous` to `requirements.txt`/`pyproject.toml`. Do this AFTER iOS smoke-tests Bearer tokens.
- **iOS smoke test** — actually build and install on iPhone.

## Common gotchas

- **PIL.BILINEAR ≠ cv2.INTER_LINEAR for the YOLO+EffNet pipeline** — both are "bilinear" but different sampling-center conventions produce different pixel values. Costs ~14pts of accuracy on fine-pitch female connectors (2.92mm-F / 3.5mm-F / 1.85mm-F confusions). `jerry_pipeline.py` uses PIL throughout to match the partner's torchvision training stack. If you ever see <90% on the 35-image holdout after the swap, suspect resampling first.
- **Labeler /grid hangs on cold load for most classes** — see [memory](C:\Users\chris\.claude\projects\E--anduril\memory\labeler_grid_hang.md). Restart `rfcai-predict` on the box before any /grid enumeration.
- **Box `nvidia-smi` shows driver mismatch** — cosmetic. CUDA runtime works. Don't reboot — production processes have open GPU handles.
- **`ProtectSystem=strict` on rfcai-predict.service** — only `/home/rfcai`, `/opt/rfcai/repo/training/data`, `/opt/rfcai/repo/training/models` are writeable. Adding code that writes elsewhere requires editing `ReadWritePaths` in the unit file + `systemctl daemon-reload`.
- **`itsdangerous` is needed** for FastAPI's `SessionMiddleware`. Installed manually in the rfcai venv (`pip install itsdangerous`); should be added to a requirements file eventually.
- **`_safe_path` must accept both data roots** — training AND holdout. If Bearer-auth Undo of a holdout upload returns 400, this is why.
- **SQLite `PRAGMA foreign_keys = ON`** — `auth.py`'s `_connect()` helper sets this on every connection. The api_tokens CASCADE delete relies on it.
- **Camera + auth handoff bug fixed in `ab30c9d`** — `_onAuthChanged` listener replaces the broken `didUpdateWidget` auth-flip detection. If the camera doesn't init after sign-in, that listener isn't firing.
- **APK browser cache** — Chrome on Android aggressively caches APK downloads. Use a `?v=anything` query string on the URL to force a fresh download.
- **Windows line endings** — git warns about LF→CRLF on commits from the Win machine. Cosmetic, doesn't affect runtime.

## What to test on iPhone (this session's smoke checklist)

1. App installs without signing errors (set your Team in Xcode if needed).
2. First launch: Identify tab shows live camera preview.
3. About tab: red AI Red brain logo at top, on-device toggle visible, "Powered by aired.com" footer.
4. Contribute tab: Sign In card appears (no hardcoded creds).
5. Sign in with `chris / Elad9651!` — camera should open immediately (no tab-dance needed; `ab30c9d` fixed this).
6. Capture a shot. Toast shows `✓ #1 2.4mm-M` (or whichever chip is selected).
7. Tap counter pill → stats bottom-sheet fetches `/labeler/stats` and renders 10 classes.
8. Tap "Sign out" from the avatar pill → Sign In card returns. Token cleared from Keychain.
9. Sign back in. Token persisted across — confirm immediate camera (no re-prompt).

If any step fails, capture the exact error or screenshot, then:
- Single-shot crashes → unlikely; we tested Android end-to-end. iOS-specific causes are usually missing Info.plist keys (already present) or signing.
- Token-related failures → check `https://aired.com/rfcai/labeler/api-tokens/exchange` from `curl` first.
- Camera doesn't show after sign-in → verify the listener fix from commit `ab30c9d` is on the branch you built (`git log --oneline | head -10`).

## Useful one-liners for debugging on the Mac

Hit the labeler stats:
```bash
curl -s https://aired.com/rfcai/labeler/stats | python -m json.tool
```

Exchange creds for a token (test the auth flow without the app):
```bash
curl -s -X POST -F username=chris -F password=Elad9651! \
  -F name=mac-debug \
  https://aired.com/rfcai/labeler/api-tokens/exchange | python -m json.tool
```

Use that token on a write route:
```bash
TOKEN='<paste-from-above>'
curl -s -H "Authorization: Bearer $TOKEN" \
  https://aired.com/rfcai/labeler/admin/users -o /tmp/admin.html
```

Tail the predict service logs on the box:
```bash
ssh chris@aired.com
# (or chris@192.168.20.235 if you're on LAN)
sudo journalctl -u rfcai-predict.service -f
```

## Skills posture for this codebase

The user prefers **velocity + verification over ceremony**. They use the `superpowers` skill set (brainstorming → writing-plans → subagent-driven-development → verification-before-completion) but only when scope genuinely warrants. For small tweaks, dive in. For features that touch multiple files or affect production, write a tight spec + plan first.

The user is decisive. They give one-line direction; ask one clarifying question if scope is genuinely ambiguous, otherwise propose and execute.

When committing on the user's behalf without explicit ask: bias toward NOT committing — show the change, let the user say "commit and push" or amend. They've corrected mid-session commits before.

When something on the server breaks: pull `journalctl -u rfcai-predict.service` first. The most common breaks are import errors after deploy (missing pip dep) or template-render errors after a CSS pass.
