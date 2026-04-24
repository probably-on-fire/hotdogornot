using UnityEngine;

namespace RFConnectorAR.Perception
{
    public interface IEmbedder
    {
        float[] Embed(Texture2D rgbCrop, Texture2D depthCrop);
    }
}
