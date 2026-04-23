using UnityEngine;

namespace RFConnectorAR.Perception
{
    public enum ConfidenceLevel
    {
        Unknown = 0,
        Low = 1,
        Medium = 2,
        High = 3,
    }

    public sealed class Verdict
    {
        public string ClassName { get; init; } = "Unknown";
        public int ClassId { get; init; } = -1;
        public ConfidenceLevel Confidence { get; init; } = ConfidenceLevel.Unknown;
        public float Score { get; init; } = 0f;
        public float? MeasuredDiameterMm { get; init; }
        public DetectionBox DetectionBox { get; init; }
        public Vector3? WorldPosition { get; init; }
    }
}
