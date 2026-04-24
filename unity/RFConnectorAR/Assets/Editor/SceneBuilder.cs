#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RFConnectorAR.App;
using RFConnectorAR.AR;
using RFConnectorAR.Curate;
using RFConnectorAR.Enroll;
using RFConnectorAR.UI;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;

namespace RFConnectorAR.EditorTools
{
    /// <summary>
    /// Programmatically builds the Scanner, Enroll, and Curate scenes so the
    /// human doesn't have to click through a 50-step editor walkthrough. Run from
    /// the menu item or headlessly via:
    ///
    ///     Unity.exe -batchmode -projectPath &lt;path&gt; \
    ///         -executeMethod RFConnectorAR.EditorTools.SceneBuilder.BuildAll -quit
    ///
    /// After the build, scenes exist, GameObjects are placed, components are
    /// attached, [SerializeField] references are wired, and prefabs are created
    /// and saved under Assets/Prefabs/. The three scenes are registered in
    /// Build Settings in the expected order (Scanner = 0, Enroll = 1, Curate = 2).
    ///
    /// Known limitations (manual tweaks the human does after):
    /// - UI anchoring / sizing is functional but not pixel-perfect. Tune in Play.
    /// - Mode-bar button OnClick events are wired via UnityEventTools.AddPersistentListener,
    ///   which is supported but less common — confirm they work before doing heavy
    ///   customization.
    /// </summary>
    public static class SceneBuilder
    {
        private const string ScenesDir = "Assets/Scenes";
        private const string PrefabsDir = "Assets/Prefabs";

        // -----------------------------------------------------------------
        // Top-level entry points
        // -----------------------------------------------------------------

        [MenuItem("RFConnectorAR/Build All Scenes")]
        public static void BuildAll()
        {
            EnsureDirectory(ScenesDir);
            EnsureDirectory(PrefabsDir);

            BuildConnectorRingPrefab();
            BuildConnectorLabelPrefab();
            BuildEnrolledClassRowPrefab();

            BuildScannerScene();
            BuildEnrollScene();
            BuildCurateScene();

            UpdateBuildSettings();

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("SceneBuilder: BuildAll complete.");
        }

        // -----------------------------------------------------------------
        // Prefabs
        // -----------------------------------------------------------------

        private static void BuildConnectorRingPrefab()
        {
            string path = $"{PrefabsDir}/ConnectorRing.prefab";
            string meshPath = $"{PrefabsDir}/ConnectorRingMesh.asset";

            // Generate a procedural torus mesh (Unity's built-in primitives don't
            // include Torus). Radii chosen so the ring visually wraps the detected
            // connector at ~6 cm outer diameter, ~5 mm tube thickness.
            Mesh torus = GenerateTorusMesh(
                majorRadius: 0.030f,   // 6 cm outer diameter
                minorRadius: 0.003f,   // 3 mm tube radius
                majorSegments: 48,
                minorSegments: 12);
            AssetDatabase.CreateAsset(torus, meshPath);

            var ring = new GameObject("ConnectorRing", typeof(MeshFilter), typeof(MeshRenderer));
            ring.GetComponent<MeshFilter>().sharedMesh =
                AssetDatabase.LoadAssetAtPath<Mesh>(meshPath);

            var mat = new Material(Shader.Find("Standard"));
            mat.color = new Color(0.2f, 0.9f, 0.4f);
            string matPath = $"{PrefabsDir}/RingGreen.mat";
            AssetDatabase.CreateAsset(mat, matPath);
            ring.GetComponent<MeshRenderer>().sharedMaterial =
                AssetDatabase.LoadAssetAtPath<Material>(matPath);

            PrefabUtility.SaveAsPrefabAsset(ring, path);
            Object.DestroyImmediate(ring);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        /// <summary>
        /// Parametric torus mesh generator. Produces a closed torus centred on the
        /// origin, lying flat on the XZ plane (like a ring sitting on a table).
        /// </summary>
        private static Mesh GenerateTorusMesh(
            float majorRadius, float minorRadius, int majorSegments, int minorSegments)
        {
            var mesh = new Mesh { name = "ConnectorRingTorus" };
            int vertCount = majorSegments * minorSegments;
            var verts = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            var tris = new int[majorSegments * minorSegments * 6];

            for (int i = 0; i < majorSegments; i++)
            {
                float theta = (i / (float)majorSegments) * Mathf.PI * 2f;
                float cosT = Mathf.Cos(theta);
                float sinT = Mathf.Sin(theta);

                for (int j = 0; j < minorSegments; j++)
                {
                    float phi = (j / (float)minorSegments) * Mathf.PI * 2f;
                    float cosP = Mathf.Cos(phi);
                    float sinP = Mathf.Sin(phi);

                    int idx = i * minorSegments + j;
                    verts[idx] = new Vector3(
                        (majorRadius + minorRadius * cosP) * cosT,
                        minorRadius * sinP,
                        (majorRadius + minorRadius * cosP) * sinT);
                    normals[idx] = new Vector3(cosP * cosT, sinP, cosP * sinT);
                    uvs[idx] = new Vector2(i / (float)majorSegments, j / (float)minorSegments);
                }
            }

            int triIdx = 0;
            for (int i = 0; i < majorSegments; i++)
            {
                int iNext = (i + 1) % majorSegments;
                for (int j = 0; j < minorSegments; j++)
                {
                    int jNext = (j + 1) % minorSegments;
                    int a = i * minorSegments + j;
                    int b = iNext * minorSegments + j;
                    int c = iNext * minorSegments + jNext;
                    int d = i * minorSegments + jNext;
                    tris[triIdx++] = a; tris[triIdx++] = b; tris[triIdx++] = c;
                    tris[triIdx++] = a; tris[triIdx++] = c; tris[triIdx++] = d;
                }
            }

            mesh.vertices = verts;
            mesh.normals = normals;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void BuildConnectorLabelPrefab()
        {
            string path = $"{PrefabsDir}/ConnectorLabel.prefab";

            var label = new GameObject("ConnectorLabel");
            var mesh = label.AddComponent<TextMesh>();
            mesh.text = "SMA-F";
            mesh.characterSize = 0.05f;
            mesh.anchor = TextAnchor.MiddleCenter;
            mesh.alignment = TextAlignment.Center;
            mesh.color = Color.white;
            label.AddComponent<MeshRenderer>();

            PrefabUtility.SaveAsPrefabAsset(label, path);
            Object.DestroyImmediate(label);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        private static void BuildEnrolledClassRowPrefab()
        {
            string path = $"{PrefabsDir}/EnrolledClassRow.prefab";

            var row = new GameObject("EnrolledClassRow", typeof(RectTransform), typeof(Image));
            var rowRect = row.GetComponent<RectTransform>();
            rowRect.sizeDelta = new Vector2(400, 50);

            var rowImg = row.GetComponent<Image>();
            rowImg.color = new Color(0.15f, 0.15f, 0.15f);

            var labelGo = new GameObject("Label", typeof(RectTransform), typeof(Text));
            labelGo.transform.SetParent(row.transform, worldPositionStays: false);
            var labelRect = labelGo.GetComponent<RectTransform>();
            labelRect.anchorMin = new Vector2(0, 0);
            labelRect.anchorMax = new Vector2(0.7f, 1);
            labelRect.offsetMin = new Vector2(12, 4);
            labelRect.offsetMax = new Vector2(-4, -4);
            var labelText = labelGo.GetComponent<Text>();
            labelText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            labelText.alignment = TextAnchor.MiddleLeft;
            labelText.color = Color.white;
            labelText.fontSize = 20;
            labelText.text = "ClassName";

            var btnGo = new GameObject("DeleteButton", typeof(RectTransform), typeof(Image), typeof(Button));
            btnGo.transform.SetParent(row.transform, worldPositionStays: false);
            var btnRect = btnGo.GetComponent<RectTransform>();
            btnRect.anchorMin = new Vector2(0.7f, 0.1f);
            btnRect.anchorMax = new Vector2(1f, 0.9f);
            btnRect.offsetMin = new Vector2(4, 4);
            btnRect.offsetMax = new Vector2(-8, -4);
            var btnImg = btnGo.GetComponent<Image>();
            btnImg.color = new Color(0.7f, 0.2f, 0.2f);

            var btnTextGo = new GameObject("Text", typeof(RectTransform), typeof(Text));
            btnTextGo.transform.SetParent(btnGo.transform, worldPositionStays: false);
            var btnTextRect = btnTextGo.GetComponent<RectTransform>();
            btnTextRect.anchorMin = Vector2.zero;
            btnTextRect.anchorMax = Vector2.one;
            btnTextRect.offsetMin = Vector2.zero;
            btnTextRect.offsetMax = Vector2.zero;
            var btnText = btnTextGo.GetComponent<Text>();
            btnText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            btnText.alignment = TextAnchor.MiddleCenter;
            btnText.color = Color.white;
            btnText.fontSize = 16;
            btnText.text = "Delete";

            PrefabUtility.SaveAsPrefabAsset(row, path);
            Object.DestroyImmediate(row);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        // -----------------------------------------------------------------
        // Scanner scene
        // -----------------------------------------------------------------

        private static void BuildScannerScene()
        {
            string path = $"{ScenesDir}/Scanner.unity";
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var arSession = BuildARSession();
            var xrOrigin = BuildXROrigin(withAnchorManager: true);

            // Find the AR Camera under XR Origin and attach CameraFrameSource.
            var arCamera = xrOrigin.GetComponentInChildren<ARCameraManager>();
            if (arCamera != null && arCamera.GetComponent<CameraFrameSource>() == null)
            {
                arCamera.gameObject.AddComponent<CameraFrameSource>();
            }

            // Attach our AnchorManager wrapper to XR Origin.
            AnchorManager anchorMgr;
            if (!xrOrigin.TryGetComponent(out anchorMgr))
            {
                anchorMgr = xrOrigin.AddComponent<AnchorManager>();
            }

            // UI canvas.
            var canvas = BuildUICanvas();
            var hintText = BuildHudText(canvas, "HintText", TextAnchor.LowerCenter, fontSize: 36,
                anchor: new Vector4(0.1f, 0f, 0.9f, 0.15f));
            var statusText = BuildHudText(canvas, "StatusText", TextAnchor.UpperLeft, fontSize: 20,
                anchor: new Vector4(0.02f, 0.92f, 0.98f, 0.98f));

            // ScannerHUD on the canvas.
            var scannerHud = canvas.GetComponent<ScannerHUD>() ?? canvas.AddComponent<ScannerHUD>();
            SetPrivateField(scannerHud, "_hint", hintText);
            SetPrivateField(scannerHud, "_status", statusText);

            // App GameObject with OverlayController + AppBootstrap.
            var app = new GameObject("App");

            var overlay = app.AddComponent<OverlayController>();
            var ringPrefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{PrefabsDir}/ConnectorRing.prefab");
            var labelPrefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{PrefabsDir}/ConnectorLabel.prefab");
            SetPrivateField(overlay, "_ringPrefab", ringPrefab);
            SetPrivateField(overlay, "_labelPrefab", labelPrefab);
            SetPrivateField(overlay, "_anchorManager", anchorMgr);

            var bootstrap = app.AddComponent<AppBootstrap>();
            var cameraFrameSource = arCamera != null ? arCamera.GetComponent<CameraFrameSource>() : null;
            SetPrivateField(bootstrap, "_cameraFrameSource", cameraFrameSource);
            SetPrivateField(bootstrap, "_overlay", overlay);
            SetPrivateField(bootstrap, "_hud", scannerHud);

            // ModeRouter + mode bar.
            var router = app.AddComponent<ModeRouter>();
            AddModeBar(canvas, router);

            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        // -----------------------------------------------------------------
        // Enroll scene
        // -----------------------------------------------------------------

        private static void BuildEnrollScene()
        {
            string path = $"{ScenesDir}/Enroll.unity";
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var arSession = BuildARSession();
            var xrOrigin = BuildXROrigin(withAnchorManager: false);

            var arCamera = xrOrigin.GetComponentInChildren<ARCameraManager>();
            if (arCamera != null && arCamera.GetComponent<CameraFrameSource>() == null)
            {
                arCamera.gameObject.AddComponent<CameraFrameSource>();
            }

            var canvas = BuildUICanvas();

            // Input field (class name).
            var input = BuildInputField(canvas, "ClassNameInput", "Class name (e.g. SMA-M)",
                anchor: new Vector4(0.1f, 0.80f, 0.7f, 0.88f));

            // Start button.
            var startBtn = BuildButton(canvas, "StartButton", "Start",
                anchor: new Vector4(0.72f, 0.80f, 0.9f, 0.88f));

            // Progress text.
            var progressText = BuildHudText(canvas, "ProgressText", TextAnchor.MiddleCenter, fontSize: 24,
                anchor: new Vector4(0.1f, 0.45f, 0.9f, 0.6f));
            progressText.text = "Type a class name and press Start.";

            // Progress slider.
            var progressBar = BuildSlider(canvas, "ProgressBar",
                anchor: new Vector4(0.2f, 0.35f, 0.8f, 0.4f));

            // EnrollHUD on canvas.
            var hud = canvas.AddComponent<EnrollHUD>();
            SetPrivateField(hud, "_classNameInput", input);
            SetPrivateField(hud, "_startButton", startBtn);
            SetPrivateField(hud, "_progressText", progressText);
            SetPrivateField(hud, "_progressBar", progressBar);

            // App with EnrollController + ModeRouter.
            var app = new GameObject("App");
            var controller = app.AddComponent<EnrollController>();
            var cameraFrameSource = arCamera != null ? arCamera.GetComponent<CameraFrameSource>() : null;
            SetPrivateField(controller, "_cameraFrameSource", cameraFrameSource);
            SetPrivateField(controller, "_hud", hud);

            var router = app.AddComponent<ModeRouter>();
            AddModeBar(canvas, router);

            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        // -----------------------------------------------------------------
        // Curate scene
        // -----------------------------------------------------------------

        private static void BuildCurateScene()
        {
            string path = $"{ScenesDir}/Curate.unity";
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // No AR Session needed for Curate — but we do need a Camera so the scene
            // renders. Add a basic UI-only camera.
            var cam = new GameObject("Main Camera", typeof(Camera));
            cam.tag = "MainCamera";
            cam.GetComponent<Camera>().clearFlags = CameraClearFlags.SolidColor;
            cam.GetComponent<Camera>().backgroundColor = new Color(0.08f, 0.08f, 0.08f);

            var canvas = BuildUICanvas();

            // Scroll view for enrolled classes.
            var scrollView = BuildScrollView(canvas, "EnrolledList",
                anchor: new Vector4(0.05f, 0.2f, 0.95f, 0.9f));
            var content = scrollView.transform.Find("Viewport/Content").GetComponent<RectTransform>();
            var vlg = content.gameObject.AddComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset(8, 8, 8, 8);
            vlg.spacing = 4;
            vlg.childControlWidth = true;
            vlg.childControlHeight = false;
            vlg.childForceExpandWidth = true;
            vlg.childForceExpandHeight = false;

            // Empty-state message.
            var emptyMessage = BuildHudText(canvas, "EmptyMessage", TextAnchor.MiddleCenter, fontSize: 18,
                anchor: new Vector4(0.2f, 0.5f, 0.8f, 0.6f));
            emptyMessage.text = "No enrolled classes yet — go to Enroll to teach the app.";

            // CurateHUD on canvas.
            var hud = canvas.AddComponent<CurateHUD>();
            SetPrivateField(hud, "_listRoot", content);
            var rowPrefab = AssetDatabase.LoadAssetAtPath<GameObject>($"{PrefabsDir}/EnrolledClassRow.prefab");
            SetPrivateField(hud, "_rowPrefab", rowPrefab);
            SetPrivateField(hud, "_emptyMessage", emptyMessage);

            // App with CurateController + ModeRouter.
            var app = new GameObject("App");
            var controller = app.AddComponent<CurateController>();
            SetPrivateField(controller, "_hud", hud);

            var router = app.AddComponent<ModeRouter>();
            AddModeBar(canvas, router);

            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"SceneBuilder: wrote {path}");
        }

        // -----------------------------------------------------------------
        // Shared helpers
        // -----------------------------------------------------------------

        private static GameObject BuildARSession()
        {
            var go = new GameObject("AR Session");
            go.AddComponent<ARSession>();
            return go;
        }

        private static GameObject BuildXROrigin(bool withAnchorManager)
        {
            // AR Foundation 6 provides XROrigin in the Unity.XR.CoreUtils namespace.
            // Create manually to stay off the AR Foundation internal prefabs, which
            // vary between versions.
            var root = new GameObject("XR Origin");
            var xrOrigin = root.AddComponent<Unity.XR.CoreUtils.XROrigin>();

            var cameraOffset = new GameObject("Camera Offset");
            cameraOffset.transform.SetParent(root.transform, worldPositionStays: false);

            var mainCam = new GameObject("Main Camera");
            mainCam.transform.SetParent(cameraOffset.transform, worldPositionStays: false);
            mainCam.tag = "MainCamera";
            var cam = mainCam.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
            cam.nearClipPlane = 0.05f;
            mainCam.AddComponent<ARCameraManager>();
            mainCam.AddComponent<ARCameraBackground>();

            xrOrigin.Camera = cam;
            xrOrigin.CameraFloorOffsetObject = cameraOffset;

            if (withAnchorManager)
            {
                root.AddComponent<ARAnchorManager>();
            }

            return root;
        }

        private static GameObject BuildUICanvas()
        {
            var canvas = new GameObject("UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var c = canvas.GetComponent<Canvas>();
            c.renderMode = RenderMode.ScreenSpaceOverlay;

            var scaler = canvas.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1080, 2400);
            scaler.matchWidthOrHeight = 0.5f;

            // EventSystem (required for UI input).
            if (Object.FindFirstObjectByType<UnityEngine.EventSystems.EventSystem>() == null)
            {
                var es = new GameObject("EventSystem",
                    typeof(UnityEngine.EventSystems.EventSystem),
                    typeof(UnityEngine.EventSystems.StandaloneInputModule));
            }

            return canvas;
        }

        private static Text BuildHudText(
            GameObject canvas, string name, TextAnchor alignment, int fontSize, Vector4 anchor)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Text));
            go.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(anchor.x, anchor.y);
            rect.anchorMax = new Vector2(anchor.z, anchor.w);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            var text = go.GetComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.alignment = alignment;
            text.color = Color.white;
            text.fontSize = fontSize;
            text.text = name;
            return text;
        }

        private static InputField BuildInputField(
            GameObject canvas, string name, string placeholder, Vector4 anchor)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(InputField));
            go.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(anchor.x, anchor.y);
            rect.anchorMax = new Vector2(anchor.z, anchor.w);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            var img = go.GetComponent<Image>();
            img.color = new Color(0.1f, 0.1f, 0.1f);

            // Placeholder text child.
            var phGo = new GameObject("Placeholder", typeof(RectTransform), typeof(Text));
            phGo.transform.SetParent(go.transform, worldPositionStays: false);
            var phRect = phGo.GetComponent<RectTransform>();
            phRect.anchorMin = Vector2.zero;
            phRect.anchorMax = Vector2.one;
            phRect.offsetMin = new Vector2(8, 4);
            phRect.offsetMax = new Vector2(-8, -4);
            var phText = phGo.GetComponent<Text>();
            phText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            phText.alignment = TextAnchor.MiddleLeft;
            phText.color = new Color(1, 1, 1, 0.5f);
            phText.fontSize = 20;
            phText.text = placeholder;

            // Actual text child.
            var txtGo = new GameObject("Text", typeof(RectTransform), typeof(Text));
            txtGo.transform.SetParent(go.transform, worldPositionStays: false);
            var txtRect = txtGo.GetComponent<RectTransform>();
            txtRect.anchorMin = Vector2.zero;
            txtRect.anchorMax = Vector2.one;
            txtRect.offsetMin = new Vector2(8, 4);
            txtRect.offsetMax = new Vector2(-8, -4);
            var txt = txtGo.GetComponent<Text>();
            txt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            txt.alignment = TextAnchor.MiddleLeft;
            txt.color = Color.white;
            txt.fontSize = 20;
            txt.supportRichText = false;

            var input = go.GetComponent<InputField>();
            input.placeholder = phText;
            input.textComponent = txt;
            input.targetGraphic = img;
            return input;
        }

        private static Button BuildButton(
            GameObject canvas, string name, string label, Vector4 anchor)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(anchor.x, anchor.y);
            rect.anchorMax = new Vector2(anchor.z, anchor.w);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            var img = go.GetComponent<Image>();
            img.color = new Color(0.12f, 0.3f, 0.12f);

            var textGo = new GameObject("Text", typeof(RectTransform), typeof(Text));
            textGo.transform.SetParent(go.transform, worldPositionStays: false);
            var textRect = textGo.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = Vector2.zero;
            textRect.offsetMax = Vector2.zero;
            var text = textGo.GetComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.alignment = TextAnchor.MiddleCenter;
            text.color = Color.white;
            text.fontSize = 20;
            text.text = label;

            var btn = go.GetComponent<Button>();
            btn.targetGraphic = img;
            return btn;
        }

        private static Slider BuildSlider(GameObject canvas, string name, Vector4 anchor)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(anchor.x, anchor.y);
            rect.anchorMax = new Vector2(anchor.z, anchor.w);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            // Background.
            var bg = new GameObject("Background", typeof(RectTransform), typeof(Image));
            bg.transform.SetParent(go.transform, worldPositionStays: false);
            var bgRect = bg.GetComponent<RectTransform>();
            bgRect.anchorMin = Vector2.zero;
            bgRect.anchorMax = Vector2.one;
            bgRect.offsetMin = Vector2.zero;
            bgRect.offsetMax = Vector2.zero;
            bg.GetComponent<Image>().color = new Color(0.15f, 0.15f, 0.15f);

            // Fill Area.
            var fillArea = new GameObject("Fill Area", typeof(RectTransform));
            fillArea.transform.SetParent(go.transform, worldPositionStays: false);
            var faRect = fillArea.GetComponent<RectTransform>();
            faRect.anchorMin = Vector2.zero;
            faRect.anchorMax = Vector2.one;
            faRect.offsetMin = new Vector2(4, 4);
            faRect.offsetMax = new Vector2(-4, -4);

            var fill = new GameObject("Fill", typeof(RectTransform), typeof(Image));
            fill.transform.SetParent(fillArea.transform, worldPositionStays: false);
            var fillRect = fill.GetComponent<RectTransform>();
            fillRect.anchorMin = Vector2.zero;
            fillRect.anchorMax = Vector2.one;
            fillRect.offsetMin = Vector2.zero;
            fillRect.offsetMax = Vector2.zero;
            fill.GetComponent<Image>().color = new Color(0.2f, 0.8f, 0.3f);

            var slider = go.AddComponent<Slider>();
            slider.fillRect = fillRect;
            slider.targetGraphic = bg.GetComponent<Image>();
            slider.direction = Slider.Direction.LeftToRight;
            slider.minValue = 0f;
            slider.maxValue = 1f;
            slider.value = 0f;
            return slider;
        }

        private static GameObject BuildScrollView(GameObject canvas, string name, Vector4 anchor)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(ScrollRect), typeof(Image));
            go.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(anchor.x, anchor.y);
            rect.anchorMax = new Vector2(anchor.z, anchor.w);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            go.GetComponent<Image>().color = new Color(0.05f, 0.05f, 0.05f);

            var viewport = new GameObject("Viewport", typeof(RectTransform), typeof(Image), typeof(Mask));
            viewport.transform.SetParent(go.transform, worldPositionStays: false);
            var vpRect = viewport.GetComponent<RectTransform>();
            vpRect.anchorMin = Vector2.zero;
            vpRect.anchorMax = Vector2.one;
            vpRect.offsetMin = Vector2.zero;
            vpRect.offsetMax = Vector2.zero;
            viewport.GetComponent<Image>().color = new Color(1, 1, 1, 0.01f);
            viewport.GetComponent<Mask>().showMaskGraphic = false;

            var content = new GameObject("Content", typeof(RectTransform));
            content.transform.SetParent(viewport.transform, worldPositionStays: false);
            var contentRect = content.GetComponent<RectTransform>();
            contentRect.anchorMin = new Vector2(0, 1);
            contentRect.anchorMax = new Vector2(1, 1);
            contentRect.pivot = new Vector2(0.5f, 1);
            contentRect.offsetMin = Vector2.zero;
            contentRect.offsetMax = Vector2.zero;
            contentRect.sizeDelta = new Vector2(0, 400);

            var sr = go.GetComponent<ScrollRect>();
            sr.viewport = vpRect;
            sr.content = contentRect;
            sr.horizontal = false;
            sr.vertical = true;

            return go;
        }

        private static void AddModeBar(GameObject canvas, ModeRouter router)
        {
            var bar = new GameObject("ModeBar", typeof(RectTransform), typeof(Image), typeof(HorizontalLayoutGroup));
            bar.transform.SetParent(canvas.transform, worldPositionStays: false);
            var barRect = bar.GetComponent<RectTransform>();
            barRect.anchorMin = new Vector2(0, 0);
            barRect.anchorMax = new Vector2(1, 0.08f);
            barRect.offsetMin = Vector2.zero;
            barRect.offsetMax = Vector2.zero;
            bar.GetComponent<Image>().color = new Color(0, 0, 0, 0.5f);

            var hlg = bar.GetComponent<HorizontalLayoutGroup>();
            hlg.childControlWidth = true;
            hlg.childControlHeight = true;
            hlg.childForceExpandWidth = true;
            hlg.childForceExpandHeight = true;
            hlg.padding = new RectOffset(8, 8, 8, 8);
            hlg.spacing = 8;

            // Three buttons wired to ModeRouter via UnityEventTools.
            var scanBtn = BuildModeBarButton(bar, "ScannerBtn", "Scan");
            UnityEventTools.AddPersistentListener(scanBtn.onClick, router.GoToScanner);

            var enrollBtn = BuildModeBarButton(bar, "EnrollBtn", "Enroll");
            UnityEventTools.AddPersistentListener(enrollBtn.onClick, router.GoToEnroll);

            var curateBtn = BuildModeBarButton(bar, "CurateBtn", "Curate");
            UnityEventTools.AddPersistentListener(curateBtn.onClick, router.GoToCurate);
        }

        private static Button BuildModeBarButton(GameObject parent, string name, string label)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(parent.transform, worldPositionStays: false);
            var img = go.GetComponent<Image>();
            img.color = new Color(0.15f, 0.15f, 0.15f);

            var textGo = new GameObject("Text", typeof(RectTransform), typeof(Text));
            textGo.transform.SetParent(go.transform, worldPositionStays: false);
            var textRect = textGo.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = Vector2.zero;
            textRect.offsetMax = Vector2.zero;
            var text = textGo.GetComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.alignment = TextAnchor.MiddleCenter;
            text.color = Color.white;
            text.fontSize = 20;
            text.text = label;

            var btn = go.GetComponent<Button>();
            btn.targetGraphic = img;
            return btn;
        }

        // -----------------------------------------------------------------
        // Build Settings
        // -----------------------------------------------------------------

        private static void UpdateBuildSettings()
        {
            var scenes = new List<EditorBuildSettingsScene>
            {
                new EditorBuildSettingsScene($"{ScenesDir}/Scanner.unity", true),
                new EditorBuildSettingsScene($"{ScenesDir}/Enroll.unity",  true),
                new EditorBuildSettingsScene($"{ScenesDir}/Curate.unity",  true),
            };
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log("SceneBuilder: build settings updated (Scanner=0, Enroll=1, Curate=2).");
        }

        // -----------------------------------------------------------------
        // Low-level utilities
        // -----------------------------------------------------------------

        private static void EnsureDirectory(string assetPath)
        {
            string fs = assetPath.Replace('/', Path.DirectorySeparatorChar);
            if (!Directory.Exists(fs))
            {
                Directory.CreateDirectory(fs);
            }
        }

        /// <summary>
        /// Set a private [SerializeField] field on a UnityEngine.Object via
        /// SerializedObject. This is the canonical editor-scripting way to
        /// wire inspector references without making fields public.
        /// </summary>
        private static void SetPrivateField(Object target, string fieldName, Object value)
        {
            if (target == null) return;
            var so = new SerializedObject(target);
            var sp = so.FindProperty(fieldName);
            if (sp == null)
            {
                Debug.LogWarning(
                    $"SceneBuilder: field {fieldName} not found on {target.GetType().Name}");
                return;
            }
            sp.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }
    }
}
#endif
