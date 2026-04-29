using UnityEngine;
using RFConnectorAR.AR;
using RFConnectorAR.Reference;
using RFConnectorAR.UI;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Drives per-frame classifier inference. Pulls the latest AR camera
    /// frame, runs the active SentisClassifier, pushes the result to the
    /// Scan panel (which renders class + confidence overlay).
    ///
    /// Throttled to a target FPS so we don't run inference every frame —
    /// a phone CPU/GPU appreciates the breathing room and the user doesn't
    /// need 60 Hz classifier updates.
    /// </summary>
    public sealed class ClassifierLoop : MonoBehaviour
    {
        [Header("Inputs")]
        [SerializeField] private CameraFrameSource cameraSource;
        [SerializeField] private ModelUpdater modelUpdater;
        [SerializeField] private ScanPanel scanPanel;

        [Header("Behavior")]
        [Tooltip("Inference target rate (Hz). 4-6 Hz is plenty for live guidance.")]
        [SerializeField] private float targetHz = 5f;

        private float _lastInferenceAt;

        private void Update()
        {
            if (cameraSource == null || modelUpdater == null) return;
            if (!cameraSource.HasFrame) return;
            var classifier = modelUpdater.GetClassifier();
            if (classifier == null) return;

            float minInterval = 1f / Mathf.Max(0.5f, targetHz);
            if (Time.time - _lastInferenceAt < minInterval) return;
            _lastInferenceAt = Time.time;

            var result = classifier.Classify(cameraSource.LatestRgb);
            if (scanPanel != null) scanPanel.NotifyClassification(result);
        }
    }
}
