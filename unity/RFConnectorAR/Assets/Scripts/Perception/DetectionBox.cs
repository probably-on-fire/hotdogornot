using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// A single detector hit: a normalized bounding box (0..1 in image coords)
    /// with a confidence score.
    /// </summary>
    public readonly struct DetectionBox
    {
        public readonly Rect NormalizedRect;
        public readonly float Score;

        public DetectionBox(Rect normalizedRect, float score)
        {
            NormalizedRect = normalizedRect;
            Score = score;
        }
    }
}
