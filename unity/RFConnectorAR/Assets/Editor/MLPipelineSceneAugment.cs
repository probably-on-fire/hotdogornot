#if UNITY_EDITOR
using System.IO;
using RFConnectorAR.AR;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;
using RFConnectorAR.UI;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace RFConnectorAR.EditorTools
{
    /// <summary>
    /// One-shot Editor utility that adds the ML pipeline (ModelUpdater +
    /// ClassifierLoop + InlineCorrectionPanel) to the Scanner scene built
    /// by SceneBuilder.
    ///
    /// Idempotent — re-running won't duplicate components. Adds a small
    /// banner + class-picker UI hierarchy under the existing Scanner Canvas.
    ///
    /// Usage:
    ///   menu: "RFConnectorAR/Augment Scanner with ML Pipeline"
    ///   batch:
    ///     Unity.exe -batchmode -projectPath &lt;path&gt; \
    ///         -executeMethod RFConnectorAR.EditorTools.MLPipelineSceneAugment.AugmentScanner -quit
    ///
    /// Run AFTER SceneBuilder.BuildAll has produced Scanner.unity. After
    /// running, set the device token on the ModelUpdater + InlineCorrectionPanel
    /// (Inspector) before building the APK.
    /// </summary>
    public static class MLPipelineSceneAugment
    {
        private const string ScannerScenePath = "Assets/Scenes/Scanner.unity";

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

            // --- Pipeline GameObject ---
            var pipelineGo = GameObject.Find("MLPipeline");
            if (pipelineGo == null)
            {
                pipelineGo = new GameObject("MLPipeline");
                EditorSceneManager.MoveGameObjectToScene(pipelineGo, scene);
            }
            var modelUpdater = pipelineGo.GetComponent<ModelUpdater>() ?? pipelineGo.AddComponent<ModelUpdater>();
            var loop = pipelineGo.GetComponent<ClassifierLoop>() ?? pipelineGo.AddComponent<ClassifierLoop>();

            // Wire CameraFrameSource → ClassifierLoop
            var loopSo = new SerializedObject(loop);
            loopSo.FindProperty("cameraSource").objectReferenceValue = cameraSource;
            loopSo.FindProperty("modelUpdater").objectReferenceValue = modelUpdater;
            // correctionPanel wired below once it exists.

            // --- Banner + picker UI ---
            var panelGo = canvas.transform.Find("InlineCorrectionPanel")?.gameObject;
            if (panelGo == null)
            {
                panelGo = BuildPanel(canvas);
            }
            var panel = panelGo.GetComponent<InlineCorrectionPanel>();
            var panelSo = new SerializedObject(panel);
            panelSo.FindProperty("cameraSource").objectReferenceValue = cameraSource;
            panelSo.FindProperty("modelUpdater").objectReferenceValue = modelUpdater;
            panelSo.ApplyModifiedProperties();

            // Wire the picker back into the loop
            loopSo.FindProperty("correctionPanel").objectReferenceValue = panel;
            loopSo.ApplyModifiedProperties();

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);

            Debug.Log("[MLPipelineSceneAugment] Scanner augmented. Now set the device token "
                      + "on the ModelUpdater + InlineCorrectionPanel inspectors and build the APK.");
        }

        private static GameObject BuildPanel(Canvas canvas)
        {
            // Root with CanvasGroup + InlineCorrectionPanel
            var root = new GameObject("InlineCorrectionPanel",
                typeof(RectTransform), typeof(CanvasGroup), typeof(InlineCorrectionPanel));
            root.transform.SetParent(canvas.transform, worldPositionStays: false);

            var rt = root.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            // Banner (top)
            var banner = NewUiNode("Banner", root.transform, new Vector2(0.05f, 0.85f), new Vector2(0.95f, 0.95f));
            banner.AddComponent<Image>().color = new Color(0, 0, 0, 0.7f);
            var bannerBtn = banner.AddComponent<Button>();
            var bannerText = AddText(banner.transform, "Not sure — tap to help train", 24);

            // Picker (full-screen modal)
            var picker = NewUiNode("Picker", root.transform, Vector2.zero, Vector2.one);
            picker.AddComponent<Image>().color = new Color(0, 0, 0, 0.85f);

            var pickerLabel = AddText(picker.transform, "Pick the connector class", 28);
            var labelRt = pickerLabel.GetComponent<RectTransform>();
            labelRt.anchorMin = new Vector2(0, 0.85f);
            labelRt.anchorMax = new Vector2(1, 0.95f);
            labelRt.offsetMin = labelRt.offsetMax = Vector2.zero;

            // Vertical layout for 8 class buttons
            var buttons = NewUiNode("Buttons", picker.transform, new Vector2(0.1f, 0.15f), new Vector2(0.9f, 0.8f));
            var vlg = buttons.AddComponent<VerticalLayoutGroup>();
            vlg.spacing = 8;
            vlg.childForceExpandWidth = true;
            vlg.childForceExpandHeight = false;
            vlg.childControlWidth = true;
            vlg.childControlHeight = false;

            // Button prefab template (will be instantiated by the panel at runtime)
            var btnTpl = NewUiNode("ClassButtonTemplate", picker.transform, Vector2.zero, Vector2.zero);
            btnTpl.AddComponent<Image>().color = new Color(0.2f, 0.4f, 0.8f);
            var btn = btnTpl.AddComponent<Button>();
            AddText(btnTpl.transform, "(class)", 22);
            btnTpl.SetActive(false);

            // Cancel
            var cancel = NewUiNode("Cancel", picker.transform, new Vector2(0.35f, 0.05f), new Vector2(0.65f, 0.12f));
            cancel.AddComponent<Image>().color = new Color(0.5f, 0.1f, 0.1f);
            var cancelBtn = cancel.AddComponent<Button>();
            AddText(cancel.transform, "Cancel", 22);

            // Result text (top of picker / banner area, fed back after upload)
            var result = NewUiNode("Result", root.transform, new Vector2(0.05f, 0.78f), new Vector2(0.95f, 0.83f));
            var resultText = AddText(result.transform, "", 18);

            // Wire SerializedFields on the panel
            var panelComp = root.GetComponent<InlineCorrectionPanel>();
            var so = new SerializedObject(panelComp);
            so.FindProperty("bannerRoot").objectReferenceValue = banner.GetComponent<RectTransform>();
            so.FindProperty("pickerRoot").objectReferenceValue = picker.GetComponent<RectTransform>();
            so.FindProperty("bannerText").objectReferenceValue = bannerText;
            so.FindProperty("resultText").objectReferenceValue = resultText;
            so.FindProperty("bannerButton").objectReferenceValue = bannerBtn;
            so.FindProperty("cancelButton").objectReferenceValue = cancelBtn;
            so.FindProperty("classButtonsContainer").objectReferenceValue = buttons.transform;
            so.FindProperty("classButtonPrefab").objectReferenceValue = btn;
            so.ApplyModifiedProperties();

            return root;
        }

        private static GameObject NewUiNode(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax)
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
    }
}
#endif
