# Connector Capture Protocol

Practical guide for capturing the video and still images that drive the
measurement pipeline + classifier training.

The protocol exists because **what you capture directly determines the
ceiling on accuracy.** A clean perpendicular shot with the marker in frame
gives sub-mm aperture measurements; a 3/4-view shot can only ever be
identified by the classifier (no measurement). Both are useful — they go to
different folders and feed different models.

---

## Equipment

- Phone with rear camera (any modern smartphone — iPhone Pro adds LiDAR but
  it's not required)
- **Printed 25mm ArUco marker** — `docs/printables/aruco_marker_25mm.png`,
  print at 100% scale, place on a flat surface next to the connector
- Plain background (white printer paper works) under the connector
- Even lighting — a desk lamp from above + ambient room light is fine, no
  hard shadows
- A fixture or putty to hold the connector vertical so you can rotate the
  camera around it without it moving

---

## What to capture per connector class

For each of the 8 classes (SMA-M, SMA-F, 3.5mm-M, 3.5mm-F, 2.92mm-M,
2.92mm-F, 2.4mm-M, 2.4mm-F), capture:

### 1. Mating-face video (~30 seconds, primary)

- Phone perpendicular to the connector's mating face, ArUco marker visible
  in the same frame
- Distance: connector face fills ~30-50% of the frame
- Slowly rotate the phone roughly ±15° around the connector axis to give
  the averager varied frames
- Vary distance slightly during capture (closer/farther) so we get a range
  of pixel scales
- Filename convention: `<CLASS>_face.mp4` (e.g. `SMA-M_face.mp4`)

This drives the **measurement pipeline** + **multi-frame averager**.
Goal: 60+ frames extracted at 2 fps, sub-mm aperture accuracy.

### 2. Multi-angle stills or video (~15 seconds, secondary)

- Walk the camera around the connector at ~30-45° elevation, capturing
  side and 3/4 views
- Marker doesn't need to be in frame — these go to the classifier, not
  the measurement pipeline
- Filename convention: `<CLASS>_angles.mp4` or 10-15 `<CLASS>_<n>.jpg` stills

This drives the **ResNet-18 classifier**. Goal: variety in viewing angle so
the classifier learns the connector's appearance from multiple perspectives.

---

## Filename + folder layout

After capture, drop videos into a single staging folder and run the
**Process Video** Streamlit page. For each video:

1. Pick the connector class
2. Pick frames-per-second (2 fps for the face video → ~60 frames; 1-2 fps
   for the angles video → ~15-30 frames)
3. Hit Extract — frames land in `data/labeled/embedder/<CLASS>/video_NNNN.jpg`

Repeated extracts accumulate (numbering continues from the highest existing
index), so capturing additional angles later just adds to the dataset.

---

## Targets

Per class, end-state for a good dataset:

| Source                 | Frames per class | Goes to                           |
|------------------------|------------------|-----------------------------------|
| Mating-face video      | 60+              | Measurement pipeline + averager   |
| Multi-angle video/stills | 15-30          | Classifier training               |
| Internet (DDG/Google)  | 30+              | Classifier training (already have ~30/class) |

Total target per class: ~100-120 images, mix of perpendicular + angled.

---

## Sanity checks before training

After extracting frames, in the **Manage Data** page for each class:

- **"Open folder in Explorer"** → eyeball the thumbnails, delete obvious
  garbage (motion blur, occluded shots)
- **"Run pipeline"** on a class with mating-face frames → should see >80%
  predicted correctly with the marker in frame; a long list of "Unknown"
  predictions means the framing was off-axis (those frames go to the
  classifier instead — the measurement pipeline rejecting them is correct
  behavior, not a bug)

After training the classifier (Train Classifier page):

- val_acc should be >90% with a balanced dataset
- Sample-image prediction at the bottom of the page should agree with the
  Manage Data measurement pipeline on perpendicular shots

---

## Troubleshooting

**Measurement pipeline returns Unknown on most frames:**
- ArUco marker not in frame, or partially occluded
- Camera not perpendicular enough — hex shape distorts past detection threshold
- Connector too small in the frame (hex contour rejected for being too small)

**Classifier val_acc stuck below 80%:**
- Class imbalance — check per-class counts in the sidebar
- Need more variety per class — augment via more capture angles
- Visually-similar classes (2.92 vs 2.4) confused — bake the marker measurement
  into the training data as a per-image annotation (deferred — current ML stack
  is image-only)

**ArUco marker not detected:**
- Print at 100% scale (verify with a ruler — should be exactly 25mm × 25mm)
- Avoid glare from overhead lighting — matte paper helps
- The marker's quiet zone (white border) is required — don't crop it
