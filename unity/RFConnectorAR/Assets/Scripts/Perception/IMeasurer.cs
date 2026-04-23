using UnityEngine;

namespace RFConnectorAR.Perception
{
    public interface IMeasurer
    {
        float? MeasureInnerPinDiameterMm(Texture2D rgbCrop, Texture2D depthCrop);
    }
}
