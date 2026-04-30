using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using RFConnectorAR.Perception;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Scan tab — draws bounding boxes + per-detection class labels on top
    /// of the live AR camera feed, plus a top status banner with the most
    /// confident detection.
    ///
    /// Bounding boxes are spawned/recycled from a pool so we don't
    /// allocate per frame. Each pooled BBoxOverlay has a green outline
    /// Image and a Text child for the class label.
    /// </summary>
    public sealed class ScanPanel : MonoBehaviour
    {
        [Header("Top status")]
        [SerializeField] private Text classNameText;
        [SerializeField] private Text confidenceText;
        [SerializeField] private Image confidenceBar;
        [SerializeField] private Text hintText;

        [Header("Overlay")]
        [SerializeField] private RectTransform overlayRoot;
        [SerializeField] private GameObject bboxOverlayPrefab;

        [Header("Thresholds")]
        [SerializeField] private float confidentThreshold = 0.75f;
        [SerializeField] private float borderlineThreshold = 0.5f;

        private static readonly Color HighColor = new(0.2f, 0.8f, 0.3f);
        private static readonly Color MidColor = new(0.95f, 0.75f, 0.1f);
        private static readonly Color LowColor = new(0.85f, 0.25f, 0.25f);

        private readonly List<BBoxOverlay> _pool = new();

        public void NotifyNoDetections(int frameWidth, int frameHeight)
        {
            SetTopStatus("—", 0f, "Point at a connector");
            HideAllBoxes();
        }

        public void NotifyDetections(List<DetectionResult> detections, int frameWidth, int frameHeight)
        {
            // Top status reflects the highest-confidence detection.
            DetectionResult? best = null;
            foreach (var d in detections)
                if (best == null || d.Classification.Confidence > best.Value.Classification.Confidence) best = d;

            if (best.HasValue)
            {
                var c = best.Value.Classification;
                string hint = c.Confidence >= confidentThreshold
                    ? ""
                    : c.Confidence >= borderlineThreshold
                        ? "Move closer or center the connector"
                        : "Tap Train to record this connector";
                SetTopStatus(c.ClassName, c.Confidence, hint);
            }

            // Recycle the pool to draw one box per detection.
            EnsurePoolSize(detections.Count);
            for (int i = 0; i < _pool.Count; i++)
            {
                if (i < detections.Count)
                {
                    _pool[i].gameObject.SetActive(true);
                    _pool[i].SetForDetection(detections[i], frameWidth, frameHeight);
                }
                else
                {
                    _pool[i].gameObject.SetActive(false);
                }
            }
        }

        private void HideAllBoxes()
        {
            foreach (var o in _pool) o.gameObject.SetActive(false);
        }

        private void EnsurePoolSize(int n)
        {
            while (_pool.Count < n)
            {
                if (bboxOverlayPrefab == null || overlayRoot == null) return;
                var go = Instantiate(bboxOverlayPrefab, overlayRoot);
                var o = go.GetComponent<BBoxOverlay>();
                if (o == null) o = go.AddComponent<BBoxOverlay>();
                _pool.Add(o);
            }
        }

        private void SetTopStatus(string className, float confidence, string hint)
        {
            if (classNameText != null) classNameText.text = className;
            if (confidenceText != null) confidenceText.text = confidence > 0 ? $"{confidence:P0}" : "—";
            Color c = confidence >= confidentThreshold ? HighColor :
                      confidence >= borderlineThreshold ? MidColor : LowColor;
            if (classNameText != null) classNameText.color = c;
            if (confidenceBar != null)
            {
                confidenceBar.color = c;
                confidenceBar.fillAmount = Mathf.Clamp01(confidence);
            }
            if (hintText != null) hintText.text = hint;
        }
    }
}
