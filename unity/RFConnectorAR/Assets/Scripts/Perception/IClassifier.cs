using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Image classifier — input a (likely cropped) RGB texture, output a class
    /// index plus full per-class softmax. The Python pipeline trains a ResNet-18
    /// on the labeled connector classes; SentisClassifier wraps the resulting
    /// ONNX model. A stub implementation may also be used for tests / pre-model
    /// states.
    /// </summary>
    public interface IClassifier
    {
        ClassificationResult Classify(Texture2D rgb);

        /// <summary>Class label strings indexed by class id.</summary>
        string[] ClassNames { get; }
    }

    public readonly struct ClassificationResult
    {
        public readonly int ClassId;
        public readonly string ClassName;
        public readonly float Confidence;
        public readonly float[] Probabilities;

        public ClassificationResult(int classId, string className, float confidence, float[] probabilities)
        {
            ClassId = classId;
            ClassName = className;
            Confidence = confidence;
            Probabilities = probabilities;
        }
    }
}
