using UnityEngine;

namespace RFConnectorAR.Perception
{
    public sealed class StubMeasurer : IMeasurer
    {
        private readonly float? _diameter;

        public StubMeasurer(float? diameterMm = 2.92f) { _diameter = diameterMm; }

        public float? MeasureInnerPinDiameterMm(Texture2D rgbCrop, Texture2D depthCrop)
        {
            _ = rgbCrop; _ = depthCrop;
            return _diameter;
        }
    }
}
