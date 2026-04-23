namespace RFConnectorAR.Perception
{
    public sealed class StubEmbedder : IEmbedder
    {
        private readonly float[] _vector;

        public StubEmbedder(int dim = 128)
        {
            _vector = new float[dim];
            _vector[0] = 1f;
        }

        public StubEmbedder(float[] vector)
        {
            _vector = (float[])vector.Clone();
        }

        public float[] Embed(UnityEngine.Texture2D rgbCrop, UnityEngine.Texture2D depthCrop)
        {
            _ = rgbCrop; _ = depthCrop;
            return (float[])_vector.Clone();
        }
    }
}
