using System.Collections.Generic;
using UnityEngine;
using RFConnectorAR.AR;
using RFConnectorAR.Reference;
using RFConnectorAR.UI;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Two-stage on-device inference loop: detect connector blobs in the
    /// camera frame, then classify each detected crop. The trained
    /// ResNet-18 was trained on tight crops (via the labeler), so this
    /// matches the train-time distribution and avoids feeding it 95%
    /// background pixels.
    ///
    /// Throttled to a target FPS so the phone CPU/GPU stays responsive.
    /// 5 Hz is plenty for live AR guidance — humans don't perceive
    /// updates faster than ~10 Hz on a video overlay.
    /// </summary>
    public sealed class ClassifierLoop : MonoBehaviour
    {
        [Header("Inputs")]
        [SerializeField] private CameraFrameSource cameraSource;
        [SerializeField] private ModelUpdater modelUpdater;
        [SerializeField] private ScanPanel scanPanel;

        [Header("Behavior")]
        [Tooltip("Inference target rate (Hz). 5 Hz works well; bump up if the phone has headroom.")]
        [SerializeField] private float targetHz = 5f;
        [SerializeField] private int maxDetectionsPerFrame = 4;

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

            var frame = cameraSource.LatestRgb;

            // Stage 1: detect connector positions.
            var detections = ConnectorDetector.Detect(frame, maxDetectionsPerFrame);

            if (detections.Count == 0)
            {
                if (scanPanel != null) scanPanel.NotifyNoDetections(frame.width, frame.height);
                return;
            }

            // Stage 2: classify each crop.
            var results = new List<DetectionResult>(detections.Count);
            foreach (var d in detections)
            {
                Texture2D crop = ConnectorDetector.CropTexture(frame, d.PaddedCrop);
                try
                {
                    var classification = classifier.Classify(crop);
                    results.Add(new DetectionResult
                    {
                        Classification = classification,
                        BBox = d.BBox,
                        PaddedCrop = d.PaddedCrop,
                    });
                }
                finally
                {
                    Object.Destroy(crop);
                }
            }

            if (scanPanel != null) scanPanel.NotifyDetections(results, frame.width, frame.height);
        }
    }

    public struct DetectionResult
    {
        public ClassificationResult Classification;
        public RectInt BBox;
        public RectInt PaddedCrop;
    }
}
