using UnityEngine;

namespace RFConnectorAR.Perception
{
    public interface IDetector
    {
        DetectionBox[] Detect(Texture2D frame);
    }
}
