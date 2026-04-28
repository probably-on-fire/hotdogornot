using System;
using System.IO;
// Sentis was rebranded to Inference Engine in com.unity.ai.inference 2.2.
// The com.unity.sentis package is now a shim. Namespace + asmdef are
// Unity.InferenceEngine; the API is the same as Sentis 2.x.
using Unity.InferenceEngine;
using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Inference Engine (formerly Sentis) ResNet-18 classifier.
    ///
    /// The Inference Engine runtime can only load its own .sentis FlatBuffer
    /// format — it can't parse raw .onnx at runtime. Two valid load paths:
    ///
    ///   - <see cref="LoadFromModelAsset"/> — preferred, used at startup
    ///     when the model is bundled in Assets/Resources/ (Unity's editor
    ///     ONNX importer converts to a ModelAsset at build time).
    ///   - <see cref="LoadFromBytes"/> — used when OTA-downloading a
    ///     pre-serialized .sentis file from the relay. NOT for raw ONNX.
    ///
    /// The exporter on the Python side (export_onnx.py) bakes ImageNet
    /// normalization into the graph, so we feed raw [0, 1] float NCHW
    /// pixels and get logits back. Softmax is applied in C# to give the
    /// caller calibrated probabilities.
    /// </summary>
    public sealed class SentisClassifier : IClassifier, IDisposable
    {
        private const int InputSize = 224;
        private const int Channels = 3;

        private readonly Worker _worker;
        private readonly TensorShape _inputShape;
        private readonly string[] _classNames;
        private bool _disposed;

        public string[] ClassNames => _classNames;

        private SentisClassifier(Model model, string[] classNames)
        {
            _classNames = classNames;
            _inputShape = new TensorShape(1, Channels, InputSize, InputSize);
            // GPUCompute is great on phones with capable GPUs; CPU is the safe
            // fallback. Inference Engine picks the best available backend
            // automatically when we pass GPUCompute.
            _worker = new Worker(model, BackendType.GPUCompute);
        }

        /// <summary>
        /// Load from a Unity-imported ModelAsset. The editor ONNX importer
        /// produces these at build time when an .onnx file lives under
        /// Assets/Resources/ (or any other discoverable folder).
        /// </summary>
        public static SentisClassifier LoadFromModelAsset(ModelAsset modelAsset, string[] classNames)
        {
            if (modelAsset == null) throw new ArgumentNullException(nameof(modelAsset));
            if (classNames == null || classNames.Length == 0)
                throw new ArgumentException("classNames is empty", nameof(classNames));
            var model = ModelLoader.Load(modelAsset);
            return new SentisClassifier(model, classNames);
        }

        /// <summary>
        /// Load from a serialized .sentis byte buffer (e.g. OTA-downloaded
        /// from the relay). This format is produced by ModelWriter.Save —
        /// raw ONNX bytes will NOT work here.
        /// </summary>
        public static SentisClassifier LoadFromBytes(byte[] sentisBytes, string[] classNames)
        {
            if (sentisBytes == null || sentisBytes.Length == 0)
                throw new ArgumentException("sentisBytes is empty", nameof(sentisBytes));
            if (classNames == null || classNames.Length == 0)
                throw new ArgumentException("classNames is empty", nameof(classNames));
            using var ms = new MemoryStream(sentisBytes);
            var model = ModelLoader.Load(ms);
            return new SentisClassifier(model, classNames);
        }

        public ClassificationResult Classify(Texture2D rgb)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(SentisClassifier));
            if (rgb == null) throw new ArgumentNullException(nameof(rgb));

            // Center-crop on CPU to a square Texture2D, then let
            // InferenceEngine's TextureConverter do the resize + NCHW pack
            // on the GPU. Avoids per-element tensor indexing concerns.
            var square = CenterCropToSquare(rgb);
            using var input = TextureConverter.ToTensor(
                square, width: InputSize, height: InputSize, channels: 3);
            UnityEngine.Object.Destroy(square);

            _worker.Schedule(input);
            using var logits = _worker.PeekOutput() as Tensor<float>;
            if (logits == null)
                throw new InvalidOperationException("model output was not Tensor<float>");

            using var cpuLogits = logits.ReadbackAndClone();
            var raw = cpuLogits.DownloadToArray();
            var probs = Softmax(raw);

            int top = 0;
            for (int i = 1; i < probs.Length; i++) if (probs[i] > probs[top]) top = i;
            string name = top < _classNames.Length ? _classNames[top] : $"class_{top}";
            return new ClassificationResult(top, name, probs[top], probs);
        }

        private static Texture2D CenterCropToSquare(Texture2D src)
        {
            int side = Mathf.Min(src.width, src.height);
            int cropX = (src.width - side) / 2;
            int cropY = (src.height - side) / 2;
            var pixels = src.GetPixels(cropX, cropY, side, side);
            var square = new Texture2D(side, side, TextureFormat.RGB24, false);
            square.SetPixels(pixels);
            square.Apply();
            return square;
        }

        private static float[] Softmax(float[] logits)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
            var exp = new float[logits.Length];
            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = Mathf.Exp(logits[i] - max);
                sum += exp[i];
            }
            for (int i = 0; i < exp.Length; i++) exp[i] /= sum;
            return exp;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _worker?.Dispose();
        }
    }
}
