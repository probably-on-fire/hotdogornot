using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using RFConnectorAR.AR;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Inline-correction UX: when a frame's prediction is uncertain, surface
    /// a small unobtrusive banner. User taps → class picker. User picks →
    /// the frame goes to the relay's /uploads endpoint with the corrected
    /// label, kicking off the continuous-learning loop.
    ///
    /// "Uncertain" means any of:
    ///   - Top class confidence below <see cref="confidenceThreshold"/>
    ///   - No detected class (classifier returned -1 or not yet loaded)
    ///
    /// The current frame is captured at the moment the user taps "help" so
    /// a finger-down moment doesn't let blur creep in. We pull the frame
    /// straight from CameraFrameSource.LatestRgb.
    /// </summary>
    [RequireComponent(typeof(CanvasGroup))]
    public sealed class InlineCorrectionPanel : MonoBehaviour
    {
        [Header("Wiring")]
        [SerializeField] private CameraFrameSource cameraSource;
        [SerializeField] private ModelUpdater modelUpdater;
        [SerializeField] private RectTransform bannerRoot;
        [SerializeField] private RectTransform pickerRoot;
        [SerializeField] private Text bannerText;
        [SerializeField] private Text resultText;
        [SerializeField] private Button bannerButton;
        [SerializeField] private Button cancelButton;
        [SerializeField] private Transform classButtonsContainer;
        [SerializeField] private Button classButtonPrefab;

        [Header("Behavior")]
        [SerializeField] private float confidenceThreshold = 0.6f;
        [SerializeField] private float bannerDebounceSeconds = 1.5f;

        [Header("Relay")]
        [Tooltip("Same instance the ModelUpdater uses, or a separate one if you prefer.")]
        [SerializeField] private string relayBaseUrl = "https://aired.com/rfcai";
        [SerializeField] private string deviceToken = "";
        [SerializeField] private string deviceId = "";

        private static readonly string[] CanonicalClasses = new[]
        {
            "SMA-M", "SMA-F",
            "3.5mm-M", "3.5mm-F",
            "2.92mm-M", "2.92mm-F",
            "2.4mm-M", "2.4mm-F",
        };

        private CanvasGroup _group;
        private RelayClient _relay;
        private byte[] _capturedJpeg;
        private float _lastBannerShownAt;
        private string _predictedClass = "Unknown";

        private void Awake()
        {
            _group = GetComponent<CanvasGroup>();
            HidePicker();
            HideBanner();

            if (string.IsNullOrEmpty(deviceId))
                deviceId = SystemInfo.deviceUniqueIdentifier;
            _relay = new RelayClient(relayBaseUrl, deviceToken, deviceId);

            if (bannerButton != null) bannerButton.onClick.AddListener(OnBannerTapped);
            if (cancelButton != null) cancelButton.onClick.AddListener(HidePicker);

            BuildPicker();
        }

        private void BuildPicker()
        {
            if (classButtonsContainer == null || classButtonPrefab == null) return;
            foreach (Transform t in classButtonsContainer) Destroy(t.gameObject);
            foreach (var cls in CanonicalClasses)
            {
                var btn = Instantiate(classButtonPrefab, classButtonsContainer);
                btn.gameObject.SetActive(true);
                var label = btn.GetComponentInChildren<Text>();
                if (label != null) label.text = cls;
                string captured = cls;
                btn.onClick.AddListener(() => OnClassChosen(captured));
            }
        }

        /// <summary>
        /// Called every frame by the perception loop with the latest result.
        /// We decide whether to surface the banner.
        /// </summary>
        public void NotifyClassification(ClassificationResult result)
        {
            _predictedClass = result.ClassName;
            bool uncertain = result.ClassId < 0 || result.Confidence < confidenceThreshold;
            if (uncertain && (Time.time - _lastBannerShownAt) > bannerDebounceSeconds)
            {
                ShowBanner($"Not sure what this is — tap to help train ({result.ClassName} {result.Confidence:P0})");
            }
            else if (!uncertain)
            {
                HideBanner();
            }
        }

        private void ShowBanner(string message)
        {
            if (bannerRoot == null) return;
            bannerRoot.gameObject.SetActive(true);
            if (bannerText != null) bannerText.text = message;
            _lastBannerShownAt = Time.time;
        }

        private void HideBanner()
        {
            if (bannerRoot != null) bannerRoot.gameObject.SetActive(false);
        }

        private void HidePicker()
        {
            if (pickerRoot != null) pickerRoot.gameObject.SetActive(false);
        }

        private void ShowPicker()
        {
            if (pickerRoot != null) pickerRoot.gameObject.SetActive(true);
        }

        private void OnBannerTapped()
        {
            // Snapshot the live AR frame at the moment of the tap.
            if (cameraSource != null && cameraSource.HasFrame)
            {
                _capturedJpeg = cameraSource.LatestRgb.EncodeToJPG(quality: 88);
            }
            HideBanner();
            ShowPicker();
        }

        private void OnClassChosen(string chosenClass)
        {
            HidePicker();
            if (_capturedJpeg == null || _capturedJpeg.Length == 0)
            {
                SetResult("(no captured frame to send)");
                return;
            }
            string reason = (chosenClass == _predictedClass) ? "auto_confirmed" : "user_corrected";
            StartCoroutine(_relay.UploadFrame(
                _capturedJpeg,
                chosenClass,
                reason,
                (uploadId, err) =>
                {
                    if (err != null) SetResult($"upload failed: {err}");
                    else SetResult($"sent ({chosenClass}) — thanks");
                }));
            _capturedJpeg = null;
        }

        private void SetResult(string msg)
        {
            if (resultText != null) resultText.text = msg;
        }
    }
}
