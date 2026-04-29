using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using RFConnectorAR.AR;
using RFConnectorAR.Reference;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Train tab — three-step flow:
    ///   1. Pick connector class from a list of buttons (radio-style)
    ///   2. Tap "Record" → captures ~20 frames over 5 seconds
    ///   3. Auto-uploads to /rfcai/uploads with the picked class
    ///
    /// While recording, the button is disabled and shows progress.
    /// After upload, status text reports success or failure.
    /// </summary>
    public sealed class TrainPanel : MonoBehaviour
    {
        [Header("Wiring")]
        [SerializeField] private VideoCapture videoCapture;
        [SerializeField] private CameraFrameSource cameraSource;
        [SerializeField] private Transform classButtonsContainer;
        [SerializeField] private Button classButtonPrefab;
        [SerializeField] private Button recordButton;
        [SerializeField] private Text recordButtonLabel;
        [SerializeField] private Text statusText;
        [SerializeField] private Slider progressSlider;

        [Header("Capture")]
        [SerializeField] private float recordDurationSeconds = 5f;
        [SerializeField] private float targetFps = 4f;

        [Header("Relay")]
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

        private RelayClient _relay;
        private string _selectedClass;
        private Button _selectedButton;
        private bool _busy;

        private void Awake()
        {
            if (string.IsNullOrEmpty(deviceId))
                deviceId = SystemInfo.deviceUniqueIdentifier;
            _relay = new RelayClient(relayBaseUrl, deviceToken, deviceId);

            BuildClassPicker();
            if (recordButton != null)
            {
                recordButton.onClick.AddListener(OnRecordTapped);
                SetRecordEnabled(false);
            }
            if (progressSlider != null) progressSlider.value = 0;
            SetStatus("Pick a connector class to begin.");
        }

        private void BuildClassPicker()
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
                Button captBtn = btn;
                btn.onClick.AddListener(() => OnClassSelected(captured, captBtn));
            }
        }

        private void OnClassSelected(string cls, Button btn)
        {
            if (_busy) return;
            _selectedClass = cls;
            // Highlight the selected button by tweaking its image color
            if (_selectedButton != null)
            {
                var prevImg = _selectedButton.GetComponent<Image>();
                if (prevImg != null) prevImg.color = new Color(0.2f, 0.4f, 0.8f);
            }
            _selectedButton = btn;
            var img = btn.GetComponent<Image>();
            if (img != null) img.color = new Color(0.2f, 0.8f, 0.3f);

            SetStatus($"Selected {cls}. Aim at the connector and tap Record.");
            SetRecordEnabled(true);
            if (recordButtonLabel != null) recordButtonLabel.text = "Record";
        }

        private void OnRecordTapped()
        {
            if (_busy || _selectedClass == null) return;
            if (videoCapture == null || cameraSource == null)
            {
                SetStatus("error: camera not ready");
                return;
            }
            if (!cameraSource.HasFrame)
            {
                SetStatus("waiting for camera...");
                return;
            }
            _busy = true;
            SetRecordEnabled(false);
            if (recordButtonLabel != null) recordButtonLabel.text = "Recording...";
            StartCoroutine(DoRecord());
        }

        private IEnumerator DoRecord()
        {
            byte[][] frames = null;
            yield return videoCapture.Record(
                durationSeconds: recordDurationSeconds,
                targetFps: targetFps,
                onComplete: result => frames = result,
                onProgress: (n, total) =>
                {
                    SetStatus($"Recording... {n}/{total} frames");
                    if (progressSlider != null)
                        progressSlider.value = (float)n / Mathf.Max(1, total);
                });

            if (frames == null || frames.Length == 0)
            {
                SetStatus("no frames captured");
                Reset();
                yield break;
            }

            SetStatus($"Uploading {frames.Length} frames...");
            string err = null;
            string uploadId = null;
            yield return _relay.UploadFrames(
                frames, _selectedClass, "manual",
                (id, e) => { uploadId = id; err = e; });

            if (err != null) SetStatus($"upload failed: {err}");
            else SetStatus($"uploaded ({frames.Length} frames as {_selectedClass}) — thanks!");
            Reset();
        }

        private void Reset()
        {
            _busy = false;
            SetRecordEnabled(_selectedClass != null);
            if (recordButtonLabel != null) recordButtonLabel.text = "Record";
            if (progressSlider != null) progressSlider.value = 0;
        }

        private void SetRecordEnabled(bool enabled)
        {
            if (recordButton != null) recordButton.interactable = enabled;
        }

        private void SetStatus(string message)
        {
            if (statusText != null) statusText.text = message;
            Debug.Log($"[TrainPanel] {message}");
        }
    }
}
