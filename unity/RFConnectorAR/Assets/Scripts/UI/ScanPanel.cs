using UnityEngine;
using UnityEngine.UI;
using RFConnectorAR.Perception;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Scan tab — overlays the live classifier prediction on top of the
    /// AR camera feed. Updates whenever ClassifierLoop calls
    /// <see cref="NotifyClassification"/>.
    ///
    /// Color-codes confidence: green = confident, yellow = borderline,
    /// red = uncertain (suggest the user switch to the Train tab).
    /// </summary>
    public sealed class ScanPanel : MonoBehaviour
    {
        [SerializeField] private Text classNameText;
        [SerializeField] private Text confidenceText;
        [SerializeField] private Image confidenceBar;
        [SerializeField] private Text hintText;

        [Header("Thresholds")]
        [SerializeField] private float confidentThreshold = 0.75f;
        [SerializeField] private float borderlineThreshold = 0.5f;

        private static readonly Color HighColor = new(0.2f, 0.8f, 0.3f);
        private static readonly Color MidColor = new(0.95f, 0.75f, 0.1f);
        private static readonly Color LowColor = new(0.85f, 0.25f, 0.25f);

        public void NotifyClassification(ClassificationResult result)
        {
            if (classNameText != null) classNameText.text = result.ClassName;
            if (confidenceText != null) confidenceText.text = $"{result.Confidence:P0}";

            Color c;
            string hint;
            if (result.Confidence >= confidentThreshold)
            {
                c = HighColor;
                hint = "";
            }
            else if (result.Confidence >= borderlineThreshold)
            {
                c = MidColor;
                hint = "Move closer or center the connector";
            }
            else
            {
                c = LowColor;
                hint = "Tap Train to record this connector";
            }

            if (confidenceBar != null)
            {
                confidenceBar.color = c;
                confidenceBar.fillAmount = Mathf.Clamp01(result.Confidence);
            }
            if (classNameText != null) classNameText.color = c;
            if (hintText != null) hintText.text = hint;
        }
    }
}
