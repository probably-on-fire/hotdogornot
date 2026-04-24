# Unity Editor Walkthrough — Scanner + Enroll + Curate

**Date:** 2026-04-24
**For:** The one sitting where you finish all the Unity-Editor work in order. Covers the pending editor tasks from Plans 2 and 2b combined.

All the C# code is on master and compiles. This doc covers only the GUI clicks needed to make the three scenes runnable.

---

## Before you start

Open Unity Hub → Open → `E:\anduril\unity\RFConnectorAR`. Let Unity finish indexing packages. **Window → Console** should be clean (no compile errors).

Run the EditMode tests once to confirm the starting state: **Window → General → Test Runner → EditMode tab → Run All**. Expect 35 passing.

If anything is red in the Console before you begin, stop and ping me.

---

## Part A — Scanner scene

This is the core identification scene. Do this first; Enroll and Curate reuse its pattern.

### A1 — Create the scene

**File → New Scene → Basic (Built-in)**. Save as `Assets/Scenes/Scanner.unity`.

Delete the default `Main Camera` and `Directional Light` from the hierarchy — AR Foundation provides its own camera.

### A2 — AR session + XR Origin

Right-click in the Hierarchy:

- **XR → AR Session** — creates a `AR Session` GameObject.
- **XR → XR Origin (AR)** — creates `XR Origin` with a child `Camera Offset → Main Camera` (already has `AR Camera Manager`, `AR Camera Background`, Tracked Pose Driver).

Select `XR Origin` → **Add Component → AR Anchor Manager**.

### A3 — Attach perception scripts

Select the Main Camera (`XR Origin → Camera Offset → Main Camera`):
- **Add Component → Camera Frame Source**

Select `XR Origin`:
- **Add Component → Anchor Manager** (our wrapper, not AR Anchor Manager — both must be present)

### A4 — Overlay canvas

Right-click Hierarchy → **UI → Canvas** named `UI`.
- Canvas component: Render Mode = Screen Space - Overlay.
- Canvas Scaler: UI Scale Mode = Scale With Screen Size, reference resolution 1080×2400, match 0.5.

### A5 — HUD text elements

Inside `UI`:
- Right-click → **UI → Legacy → Text** named `HintText`. Anchor bottom-center. Font size ~36. Color white.
- Right-click → **UI → Legacy → Text** named `StatusText`. Anchor top-left. Font size ~20.

### A6 — Ring prefab

In Hierarchy: right-click → **3D Object → Torus** named `ConnectorRing`.
- Transform Scale: **(0.06, 0.005, 0.06)**.
- **Create → Material** named `RingGreen`. Albedo = bright green. Assign to the torus's Mesh Renderer.

Drag `ConnectorRing` from Hierarchy into `Assets/Prefabs/` (create the Prefabs folder if it doesn't exist). Then delete the instance from the scene.

### A7 — Label prefab

In Hierarchy: right-click → **3D Object → 3D Text** named `ConnectorLabel`.
- Character Size: 0.05. Anchor Middle Center. Alignment Center. Color white.

Drag `ConnectorLabel` into `Assets/Prefabs/`. Delete the scene instance.

### A8 — App GameObject + wiring

In Hierarchy: right-click → **Create Empty** named `App`.

Add these components to `App`:
- **Overlay Controller**
- **App Bootstrap**
- **Scanner HUD** (wait — this goes on the `UI` Canvas; see below)

Select the `UI` Canvas:
- **Add Component → Scanner HUD**
  - `Hint` field: drag `HintText` from the Hierarchy
  - `Status` field: drag `StatusText`

Back on `App`, wire the components:
- **Overlay Controller**:
  - `Ring Prefab`: drag `Assets/Prefabs/ConnectorRing.prefab`
  - `Label Prefab`: drag `Assets/Prefabs/ConnectorLabel.prefab`
  - `Anchor Manager`: drag the `XR Origin` GameObject (which has the `AnchorManager` component)
- **App Bootstrap**:
  - `Camera Frame Source`: drag the Main Camera
  - `Overlay`: drag `App` itself (which has `OverlayController`)
  - `Hud`: drag the `UI` Canvas

### A9 — Add scene to Build Settings

**File → Build Settings**. Click **Add Open Scenes**. Make sure `Assets/Scenes/Scanner.unity` is at index 0.

### A10 — Test

Press **Play**. AR simulation mode activates (you'll see a stylized checker environment). Within a second or two you should see:
- A green ring floating ~30 cm in front of the simulated camera
- Label text `SMA-F` (stub perception always reports this)
- Status text at top-left: `Model stub-v0 • 0 confirmations`
- Hint text at bottom: `SMA-F [High]` or similar

Stop Play. If any of this is missing, check the Console for errors and re-verify the inspector drag-wiring.

---

## Part B — Enroll scene

### B1 — Create scene

**File → New Scene → Basic**. Save as `Assets/Scenes/Enroll.unity`.

Delete default camera and light. Repeat the AR setup from A2: add **XR → AR Session** and **XR → XR Origin (AR)**. Attach **Camera Frame Source** to the Main Camera.

(You don't need AR Anchor Manager or our `AnchorManager` component in this scene — enroll doesn't place world-anchored overlays.)

### B2 — Canvas + UI elements

Right-click Hierarchy → **UI → Canvas** named `UI`. Same scaler settings as Scanner.

Inside `UI`, create:
- **UI → Legacy → Input Field** named `ClassNameInput`. Anchor top. Placeholder text: "Class name (e.g. SMA-M)".
- **UI → Legacy → Button** named `StartButton`. Anchor top-right. Text: "Start".
- **UI → Legacy → Text** named `ProgressText`. Anchor middle. Font size 24.
- **UI → Legacy → Slider** named `ProgressBar`. Anchor just below `ProgressText`. Width ~60% of screen.

### B3 — App + script wiring

Create empty `App` GameObject.

On the `UI` Canvas: **Add Component → Enroll HUD**
- `Class Name Input`: drag `ClassNameInput`
- `Start Button`: drag `StartButton`
- `Progress Text`: drag `ProgressText`
- `Progress Bar`: drag `ProgressBar`
- `Hint`: (leave empty or drag `ProgressText` again — the script uses this optionally)

Wait — `EnrollHUD.cs` doesn't exist yet in the code tree (it's listed in Plan 2b Task 7 which we haven't dispatched). **Stop here.** Ping me and I'll dispatch a subagent to write `EnrollHUD.cs` + `EnrollController.cs`, then come back to this step.

### B4 — Add scene to Build Settings

When B3 is unblocked: **File → Build Settings → Add Open Scenes**. Index 1 (Scanner stays 0).

### B5 — Test

Press Play. Type "TestConnector" in the input field. Click Start. The progress bar should fill over ~5 seconds and show "Done" messaging when complete. File `Application.persistentDataPath/references.bin` now exists (check via Unity's `Application.persistentDataPath` log or the OS file browser).

---

## Part C — Curate scene

### C1 — Create scene

**File → New Scene → Basic**. Save as `Assets/Scenes/Curate.unity`.

Delete default camera and light. No AR needed — this is pure UI.

### C2 — Row prefab

In a temporary scene or directly in Curate: right-click Hierarchy → **UI → Panel** named `RowTemplate`.
- Inside: **UI → Legacy → Text** named `Label`
- Inside: **UI → Legacy → Button** named `DeleteButton`, text "Delete"

Drag `RowTemplate` into `Assets/Prefabs/EnrolledClassRow.prefab`. Delete the scene instance.

### C3 — Canvas + scroll view

Right-click → **UI → Canvas** named `UI`.

Inside `UI`:
- Right-click → **UI → Legacy → Scroll View** named `ListScroll`. Anchor to fill most of the screen.
- Inside `ListScroll → Viewport → Content`: **Add Component → Vertical Layout Group**.
- **UI → Legacy → Text** named `EmptyMessage`. Anchor center. Text: "No enrolled classes yet — go to Enroll to teach the app."

### C4 — App + script wiring

Wait — `CurateHUD.cs` and `CurateController.cs` don't exist yet (Plan 2b Task 8). Same pause as B3. Ping me for the subagent dispatch.

### C5 — Test

After the dispatch and wiring: press Play. If you enrolled something in Part B, it should appear in the list with a Delete button. Click Delete; it disappears.

### C6 — Add to Build Settings

Index 2.

---

## Part D — Mode bar (Scanner ↔ Enroll ↔ Curate)

In each of the three scenes, add a bottom navigation bar.

### D1 — For each scene

Inside the Canvas:
- Right-click → **UI → Panel** named `ModeBar`. Anchor bottom-stretch. Height 60. Background semi-transparent black.
- On the panel: **Add Component → Horizontal Layout Group**. Child Force Expand: Width = true, Height = true.
- Add three child buttons: `ScannerBtn`, `EnrollBtn`, `CurateBtn`. Text: "Scan", "Enroll", "Curate".

On the Canvas GameObject (or a new empty `App` GameObject if none exists in Curate):
- **Add Component → Mode Router**

For each button's **OnClick** event (in the Inspector):
- **ScannerBtn OnClick**: `ModeRouter.GoToScanner()`
- **EnrollBtn OnClick**: `ModeRouter.GoToEnroll()`
- **CurateBtn OnClick**: `ModeRouter.GoToCurate()`

Drag `ModeRouter`'s containing GameObject into each button's object slot when wiring OnClick.

### D2 — Test

Press Play in Scanner. Click the Enroll button in the bottom bar. Enroll scene loads. Click Curate. Curate scene loads. Click Scan. Back to Scanner. No errors in Console.

---

## Part E — Device build smoke test

### E1 — Android

**File → Build Settings → Platform: Android → Switch Platform** (wait for reimport).

Click **Build**. Save the APK to `unity/RFConnectorAR/Builds/Android/` (Builds is gitignored).

Verify: the APK builds without errors. If you have an ARCore-capable device connected via `adb`, `adb install -r Builds/Android/RFConnectorAR.apk` should deploy it.

### E2 — iOS

**File → Build Settings → Platform: iOS → Switch Platform**. **Build**. Output an Xcode project to `unity/RFConnectorAR/Builds/iOS/`.

If you have a Mac: open `Unity-iPhone.xcodeproj` there, select your signing team and a real ARKit-capable device, **Run**.

If you're Windows-only: just verify the Xcode project was produced. On-device verification happens later.

---

## Final checkpoint

Back in EditMode in Unity:

1. **Window → General → Test Runner → EditMode → Run All** should still be 35/35 green.
2. Scanner Play: green ring + SMA-F label appear.
3. Enroll Play: type a name, hit Start, progress bar fills, "Done" message.
4. Curate Play: enrolled names appear with delete buttons; deletion works.
5. Mode bar buttons navigate between all three scenes cleanly.

When all five are ✅, Plan 2 + Plan 2b are fully demoable with stub perception. Plan 3 is the next piece — swapping the stubs for real Sentis-backed ONNX inference.

---

## Known pause points

- **B3** needs a subagent dispatch for `EnrollHUD.cs` + `EnrollController.cs` (Plan 2b Task 7).
- **C4** needs a subagent dispatch for `CurateHUD.cs` + `CurateController.cs` (Plan 2b Task 8).

Ping me when you hit those and I'll kick them off. Everything else you can do straight through.
