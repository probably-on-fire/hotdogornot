using UnityEngine;

namespace RFConnectorAR.Perception
{
    public sealed class StubDetector : IDetector
    {
        private readonly float _score;
        private readonly Rect _box;

        public StubDetector(float score = 0.9f)
            : this(score, new Rect(0.35f, 0.35f, 0.3f, 0.3f)) { }

        public StubDetector(float score, Rect normalizedBox)
        {
            _score = score;
            _box = normalizedBox;
        }

        public DetectionBox[] Detect(Texture2D frame)
        {
            _ = frame;
            return new[] { new DetectionBox(_box, _score) };
        }
    }
}
