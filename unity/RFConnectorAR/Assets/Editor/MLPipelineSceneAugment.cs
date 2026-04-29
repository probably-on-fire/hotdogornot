#if UNITY_EDITOR
using RFConnectorAR.AR;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;
using RFConnectorAR.UI;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;

namespace RFConnectorAR.EditorTools
{
    /// <summary>
    /// One-shot Editor utility that augments the Scanner scene with the new
    /// two-tab UI:
    ///
    ///   [Scan tab]  — live camera + classifier prediction overlay
    ///   [Train tab] — class picker + record + upload to /rfcai/uploads
    ///
    /// Idempotent — re-running rebuilds the panels cleanly. Run via:
    ///   menu: "RFConnectorAR/Augment Scanner with ML Pipeline"
    ///   batch:
    ///     Unity.exe -batchmode -projectPath &lt;path&gt; \
    ///         -executeMethod RFConnectorAR.EditorTools.MLPipelineSceneAugment.AugmentScanner -quit
    ///
    /// After running, set the device token on the ModelUpdater + TrainPanel
    /// (Inspector) before building the APK.
    /// </summary>
    public static class MLPipelineSceneAugment
    {
        private const string ScannerScenePath = "Assets/Scenes/Scanner.unity";

        // Layout constants — keeps the magic numbers in one place
        private const float TabBarHeightFrac = 0.10f;

        [MenuItem("RFConnectorAR/Augment Scanner with ML Pipeline")]
        public static void AugmentScanner()
        {
            var scene = EditorSceneManager.OpenScene(ScannerScenePath, OpenSceneMode.Single);

            var cameraSource = Object.FindFirstObjectByType<CameraFrameSource>();
            if (cameraSource == null)
            {
                Debug.LogError("[MLPipelineSceneAugment] no CameraFrameSource — run SceneBuilder first.");
                return;
            }

            var canvas = Object.FindFirstObjectByType<Canvas>();
            if (canvas == null)
            {
                Debug.LogError("[MLPipelineSceneAugment] no Canvas in Scanner scene.");
                return;
            }

            // Remove any pre-existing ML pipeline UI so re-running is clean.
            var oldPanel = canvas.transform.Find("InlineCorrectionPanel");
            if (oldPanel != null) Object.DestroyImmediate(oldPanel.gameObject);
            var oldRoot = canvas.transform.Find("MLPipelineUI");
            if (oldRoot != null) Object.DestroyImmediate(oldRoot.gameObject);
            var oldBoot = canvas.transform.Find("ARBootstrapOverlay");
            if (oldBoot != null) Object.DestroyImmediate(oldBoot.gameObject);

            // ---- MLPipeline GameObject (services) -----------------------
            var pipelineGo = GameObject.Find("MLPipeline");
            if (pipelineGo == null)
            {
                pipelineGo = new GameObject("MLPipeline");
                EditorSceneManager.MoveGameObjectToScene(pipelineGo, scene);
            }
            var modelUpdater = pipelineGo.GetComponent<ModelUpdater>() ?? pipelineGo.AddComponent<ModelUpdater>();
            var loop = pipelineGo.GetComponent<ClassifierLoop>() ?? pipelineGo.AddComponent<ClassifierLoop>();
            var videoCapture = pipelineGo.GetComponent<VideoCapture>() ?? pipelineGo.AddComponent<VideoCapture>();
            SerializeFieldRef(videoCapture, "cameraSource", cameraSource);

            // ---- ARBootstrapper — handles ARCore install + perms -------
            var arSession = Object.FindFirstObjectByType<ARSession>();
            if (arSession == null)
                Debug.LogWarning("[MLPipelineSceneAugment] no ARSession in scene — ARBootstrapper will be inert");
            var bootstrapper = pipelineGo.GetComponent<ARBootstrapper>() ?? pipelineGo.AddComponent<ARBootstrapper>();
            if (arSession != null) SerializeFieldRef(bootstrapper, "session", arSession);

            // ---- Build the tabbed UI hierarchy --------------------------
            var rootGo = new GameObject("MLPipelineUI", typeof(RectTransform));
            rootGo.transform.SetParent(canvas.transform, worldPositionStays: false);
            var rootRt = rootGo.GetComponent<RectTransform>();
            rootRt.anchorMin = Vector2.zero;
            rootRt.anchorMax = Vector2.one;
            rootRt.offsetMin = Vector2.zero;
            rootRt.offsetMax = Vector2.zero;

            // Two panels (sit above the tab bar)
            var scanPanel = NewUiNode("ScanPanel", rootGo.transform,
                new Vector2(0, TabBarHeightFrac), Vector2.one);
            var trainPanel = NewUiNode("TrainPanel", rootGo.transform,
                new Vector2(0, TabBarHeightFrac), Vector2.one);

            BuildScanPanel(scanPanel.transform);
            BuildTrainPanel(trainPanel.transform, cameraSource, videoCapture);

            // Tab bar at the bottom
            var tabBar = NewUiNode("TabBar", rootGo.transform,
                Vector2.zero, new Vector2(1, TabBarHeightFrac));
            tabBar.AddComponent<Image>().color = new Color(0, 0, 0, 0.85f);
            var hlg = tabBar.AddComponent<HorizontalLayoutGroup>();
            hlg.spacing = 4;
            hlg.padding = new RectOffset(8, 8, 8, 8);
            hlg.childForceExpandWidth = true;
            hlg.childForceExpandHeight = true;
            hlg.childControlWidth = true;
            hlg.childControlHeight = true;

            var scanBtn = MakeTabButton(tabBar.transform, "Scan");
            var trainBtn = MakeTabButton(tabBar.transform, "Train");

            // Wire the TabRouter
            var router = rootGo.AddComponent<TabRouter>();
            var routerSo = new SerializedObject(router);
            var tabsList = routerSo.FindProperty("tabs");
            tabsList.arraySize = 2;
            FillTab(tabsList.GetArrayElementAtIndex(0), "Scan", scanBtn, scanPanel);
            FillTab(tabsList.GetArrayElementAtIndex(1), "Train", trainBtn, trainPanel);
            routerSo.ApplyModifiedProperties();

            // ---- Wire ClassifierLoop -----------------------------------
            var scanComp = scanPanel.GetComponent<ScanPanel>();
            SerializeFieldRef(loop, "cameraSource", cameraSource);
            SerializeFieldRef(loop, "modelUpdater", modelUpdater);
            SerializeFieldRef(loop, "scanPanel", scanComp);

            // ---- Bootstrap overlay (full-screen, dismissed when AR ready)
            var bootOverlay = NewUiNode("ARBootstrapOverlay", canvas.transform,
                Vector2.zero, Vector2.one);
            bootOverlay.AddComponent<Image>().color = new Color(0, 0, 0, 0.9f);
            var bootStatus = AddText(bootOverlay.transform, "Initializing AR...", 24);
            bootStatus.color = Color.white;
            bootStatus.alignment = TextAnchor.MiddleCenter;
            var bootRt = bootStatus.GetComponent<RectTransform>();
            bootRt.anchorMin = new Vector2(0.05f, 0.45f);
            bootRt.anchorMax = new Vector2(0.95f, 0.55f);
            bootRt.offsetMin = bootRt.offsetMax = Vector2.zero;
            // Ensure overlay sits on TOP of the tabbed UI
            bootOverlay.transform.SetAsLastSibling();
            SerializeFieldRef(bootstrapper, "overlayRoot", bootOverlay);
            SerializeFieldRef(bootstrapper, "statusText", bootStatus);

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);

            Debug.Log("[MLPipelineSceneAugment] Scanner augmented with Scan/Train tabs. " +
                      "Set the Device Token on MLPipeline.ModelUpdater and " +
                      "MLPipelineUI/TrainPanel.TrainPanel before building.");
        }

        private static void FillTab(SerializedProperty tabProp, string name,
                                    Button button, GameObject panel)
        {
            tabProp.FindPropertyRelative("name").stringValue = name;
            tabProp.FindPropertyRelative("button").objectReferenceValue = button;
            tabProp.FindPropertyRelative("panel").objectReferenceValue = panel;
            tabProp.FindPropertyRelative("buttonBackground").objectReferenceValue =
                button.GetComponent<Image>();
        }

        // -----------------------------------------------------------------
        // Scan panel layout
        // -----------------------------------------------------------------
        private static void BuildScanPanel(Transform root)
        {
            // Top region with class name + confidence
            var top = NewUiNode("Top", root, new Vector2(0, 0.85f), Vector2.one);
            var topBg = top.AddComponent<Image>();
            topBg.color = new Color(0, 0, 0, 0.55f);

            var className = AddText(top.transform, "—", 38);
            className.alignment = TextAnchor.MiddleCenter;
            var classRt = className.GetComponent<RectTransform>();
            classRt.anchorMin = new Vector2(0, 0.5f);
            classRt.anchorMax = Vector2.one;
            classRt.offsetMin = classRt.offsetMax = Vector2.zero;
            className.gameObject.name = "ClassNameText";

            var conf = AddText(top.transform, "—", 20);
            conf.alignment = TextAnchor.MiddleCenter;
            var confRt = conf.GetComponent<RectTransform>();
            confRt.anchorMin = Vector2.zero;
            confRt.anchorMax = new Vector2(1, 0.5f);
            confRt.offsetMin = confRt.offsetMax = Vector2.zero;
            conf.gameObject.name = "ConfidenceText";

            // Confidence bar at the bottom edge of the top region
            var bar = NewUiNode("ConfidenceBar", top.transform,
                new Vector2(0, 0), new Vector2(1, 0.05f));
            var barImg = bar.AddComponent<Image>();
            barImg.color = new Color(0.2f, 0.8f, 0.3f);
            barImg.type = Image.Type.Filled;
            barImg.fillMethod = Image.FillMethod.Horizontal;
            barImg.fillAmount = 0f;

            // Hint text in the middle
            var hint = AddText(root, "Point at a connector", 22);
            hint.alignment = TextAnchor.MiddleCenter;
            hint.color = Color.white;
            var hintRt = hint.GetComponent<RectTransform>();
            hintRt.anchorMin = new Vector2(0, 0.4f);
            hintRt.anchorMax = new Vector2(1, 0.5f);
            hintRt.offsetMin = hintRt.offsetMax = Vector2.zero;
            hint.gameObject.name = "HintText";

            // Wire the ScanPanel component
            var scan = root.gameObject.AddComponent<ScanPanel>();
            SerializeFieldRef(scan, "classNameText", className);
            SerializeFieldRef(scan, "confidenceText", conf);
            SerializeFieldRef(scan, "confidenceBar", barImg);
            SerializeFieldRef(scan, "hintText", hint);
        }

        // -----------------------------------------------------------------
        // Train panel layout
        // -----------------------------------------------------------------
        private static void BuildTrainPanel(Transform root, CameraFrameSource cameraSource, VideoCapture videoCapture)
        {
            var bg = root.gameObject.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.6f);

            // Title
            var title = AddText(root, "Pick connector class", 26);
            title.alignment = TextAnchor.MiddleCenter;
            var titleRt = title.GetComponent<RectTransform>();
            titleRt.anchorMin = new Vector2(0, 0.88f);
            titleRt.anchorMax = new Vector2(1, 0.96f);
            titleRt.offsetMin = titleRt.offsetMax = Vector2.zero;

            // Class buttons grid (2 columns × 4 rows)
            var grid = NewUiNode("Buttons", root,
                new Vector2(0.05f, 0.32f), new Vector2(0.95f, 0.86f));
            var glg = grid.AddComponent<GridLayoutGroup>();
            glg.cellSize = new Vector2(280, 70);
            glg.spacing = new Vector2(8, 8);
            glg.childAlignment = TextAnchor.MiddleCenter;
            glg.constraint = GridLayoutGroup.Constraint.FixedColumnCount;
            glg.constraintCount = 2;

            // Hidden template that TrainPanel instantiates per class
            var template = NewUiNode("ClassButtonTemplate", root,
                Vector2.zero, Vector2.zero);
            template.AddComponent<Image>().color = new Color(0.2f, 0.4f, 0.8f);
            var templateBtn = template.AddComponent<Button>();
            AddText(template.transform, "(class)", 22);
            template.SetActive(false);

            // Status text
            var status = AddText(root, "", 20);
            status.alignment = TextAnchor.MiddleCenter;
            var statusRt = status.GetComponent<RectTransform>();
            statusRt.anchorMin = new Vector2(0.05f, 0.20f);
            statusRt.anchorMax = new Vector2(0.95f, 0.30f);
            statusRt.offsetMin = statusRt.offsetMax = Vector2.zero;
            status.gameObject.name = "StatusText";

            // Progress slider
            var sliderGo = new GameObject("Progress",
                typeof(RectTransform), typeof(Slider), typeof(Image));
            sliderGo.transform.SetParent(root, false);
            var sliderRt = sliderGo.GetComponent<RectTransform>();
            sliderRt.anchorMin = new Vector2(0.1f, 0.16f);
            sliderRt.anchorMax = new Vector2(0.9f, 0.20f);
            sliderRt.offsetMin = sliderRt.offsetMax = Vector2.zero;
            sliderGo.GetComponent<Image>().color = new Color(0.2f, 0.2f, 0.25f);
            var slider = sliderGo.GetComponent<Slider>();
            slider.minValue = 0; slider.maxValue = 1; slider.value = 0;

            var fill = new GameObject("Fill", typeof(RectTransform), typeof(Image));
            fill.transform.SetParent(sliderGo.transform, false);
            var fillRt = fill.GetComponent<RectTransform>();
            fillRt.anchorMin = Vector2.zero;
            fillRt.anchorMax = new Vector2(0, 1);
            fillRt.pivot = new Vector2(0, 0.5f);
            fillRt.offsetMin = fillRt.offsetMax = Vector2.zero;
            fill.GetComponent<Image>().color = new Color(0.2f, 0.8f, 0.3f);

            var fillArea = new GameObject("FillArea", typeof(RectTransform));
            fillArea.transform.SetParent(sliderGo.transform, false);
            var fillAreaRt = fillArea.GetComponent<RectTransform>();
            fillAreaRt.anchorMin = Vector2.zero;
            fillAreaRt.anchorMax = Vector2.one;
            fillAreaRt.offsetMin = fillAreaRt.offsetMax = Vector2.zero;
            fill.transform.SetParent(fillArea.transform, false);

            slider.fillRect = fill.GetComponent<RectTransform>();

            // Record button
            var recordBtnGo = NewUiNode("RecordButton", root,
                new Vector2(0.2f, 0.04f), new Vector2(0.8f, 0.14f));
            var recordImg = recordBtnGo.AddComponent<Image>();
            recordImg.color = new Color(0.85f, 0.25f, 0.25f);
            var recordBtn = recordBtnGo.AddComponent<Button>();
            var recordLabel = AddText(recordBtnGo.transform, "Record", 28);
            recordLabel.color = Color.white;
            recordLabel.alignment = TextAnchor.MiddleCenter;

            // Wire the TrainPanel
            var train = root.gameObject.AddComponent<TrainPanel>();
            SerializeFieldRef(train, "videoCapture", videoCapture);
            SerializeFieldRef(train, "cameraSource", cameraSource);
            SerializeFieldRef(train, "classButtonsContainer", grid.transform);
            SerializeFieldRef(train, "classButtonPrefab", templateBtn);
            SerializeFieldRef(train, "recordButton", recordBtn);
            SerializeFieldRef(train, "recordButtonLabel", recordLabel);
            SerializeFieldRef(train, "statusText", status);
            SerializeFieldRef(train, "progressSlider", slider);
        }

        // -----------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------
        private static Button MakeTabButton(Transform parent, string label)
        {
            var go = new GameObject(label + "Tab",
                typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(parent, false);
            go.GetComponent<Image>().color = new Color(0.15f, 0.15f, 0.2f);
            var btn = go.GetComponent<Button>();
            var t = AddText(go.transform, label, 26);
            t.color = Color.white;
            t.alignment = TextAnchor.MiddleCenter;
            return btn;
        }

        private static GameObject NewUiNode(string name, Transform parent,
                                            Vector2 anchorMin, Vector2 anchorMax)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            return go;
        }

        private static Text AddText(Transform parent, string text, int fontSize)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = rt.offsetMax = Vector2.zero;
            var t = go.GetComponent<Text>();
            t.text = text;
            t.alignment = TextAnchor.MiddleCenter;
            t.color = Color.white;
            t.fontSize = fontSize;
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            return t;
        }

        private static void SerializeFieldRef(Object component, string fieldName, Object reference)
        {
            var so = new SerializedObject(component);
            var prop = so.FindProperty(fieldName);
            if (prop == null)
            {
                Debug.LogWarning($"[MLPipelineSceneAugment] field '{fieldName}' not found on {component.GetType().Name}");
                return;
            }
            prop.objectReferenceValue = reference;
            so.ApplyModifiedProperties();
        }
    }
}
#endif
