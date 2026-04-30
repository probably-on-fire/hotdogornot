using UnityEngine;
using UnityEngine.UI;
using RFConnectorAR.Perception;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Single bounding-box overlay element. Re-positioned each frame by
    /// ScanPanel based on the latest detection. Coordinates are mapped
    /// from camera-frame pixel space to the overlay RectTransform's
    /// normalized space.
    ///
    /// Expected children (built by the augment script if missing):
    ///   - Outline (Image, transparent fill, sliced sprite for the border)
    ///   - Label (Text, anchored to the top-left of the box)
    /// </summary>
    public sealed class BBoxOverlay : MonoBehaviour
    {
        [SerializeField] private RectTransform outline;
        [SerializeField] private Text label;

        private RectTransform _rt;

        private void Awake()
        {
            _rt = (RectTransform)transform;
        }

        public void SetForDetection(DetectionResult d, int frameWidth, int frameHeight)
        {
            // Use the padded-crop region for the visible box (matches what
            // the classifier actually saw). Convert from camera-frame
            // pixels to the parent RectTransform's normalized [0,1] space.
            float x0 = d.PaddedCrop.x / (float)frameWidth;
            float y0 = d.PaddedCrop.y / (float)frameHeight;
            float x1 = (d.PaddedCrop.x + d.PaddedCrop.width) / (float)frameWidth;
            float y1 = (d.PaddedCrop.y + d.PaddedCrop.height) / (float)frameHeight;

            // AR camera frames typically have y-flipped coords vs the UI;
            // flip here so the box draws over the right spot on screen.
            float fy0 = 1f - y1;
            float fy1 = 1f - y0;

            _rt.anchorMin = new Vector2(x0, fy0);
            _rt.anchorMax = new Vector2(x1, fy1);
            _rt.offsetMin = Vector2.zero;
            _rt.offsetMax = Vector2.zero;

            if (label != null)
            {
                label.text = $"{d.Classification.ClassName} {d.Classification.Confidence:P0}";
                Color c = d.Classification.Confidence >= 0.75f
                    ? new Color(0.2f, 0.85f, 0.35f)
                    : d.Classification.Confidence >= 0.5f
                        ? new Color(0.95f, 0.75f, 0.1f)
                        : new Color(0.85f, 0.3f, 0.3f);
                label.color = c;
                if (outline != null)
                {
                    var img = outline.GetComponent<Image>();
                    if (img != null) img.color = c;
                }
            }
        }
    }
}
