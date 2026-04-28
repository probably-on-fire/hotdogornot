using System;
using System.IO;
using Unity.Sentis;
using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Sentis-backed ResNet-18 classifier.
    ///
    /// Loads the ONNX model produced by <c>rfconnectorai.classifier.export_onnx</c>.
    /// The exporter bakes ImageNet normalization into the graph, so we feed
    /// raw [0, 1] float NCHW pixels and get logits back. Softmax is applied
    /// in C# (cheap; one tensor) so the on-device caller gets calibrated
    /// per-class probabilities.
    ///
    /// Two construction paths:
    ///   - <see cref="LoadFromBytes"/> — for OTA-downloaded weights buffered in memory
    ///   - <see cref="LoadFromStreamingAssets"/> — for the bundled startup model
    ///
    /// Reload behavior: the class is created fresh per model version. The
    /// caller (ModelUpdater) disposes the old instance and constructs a new
    /// one when a newer model arrives — keeps lifecycle obvious vs. mutating
    /// internal state.
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
            // fallback. Sentis picks the best available backend automatically
            // when we pass GPUCompute and falls back to CPU as needed.
            _worker = new Worker(model, BackendType.GPUCompute);
        }

        public static SentisClassifier LoadFromBytes(byte[] onnxBytes, string[] classNames)
        {
            if (onnxBytes == null || onnxBytes.Length == 0)
                throw new ArgumentException("onnxBytes is empty", nameof(onnxBytes));
            if (classNames == null || classNames.Length == 0)
                throw new ArgumentException("classNames is empty", nameof(classNames));

            using var ms = new MemoryStream(onnxBytes);
            var model = ModelLoader.Load(ms);
            return new SentisClassifier(model, classNames);
        }

        /// <summary>
        /// Load the ONNX file at <paramref name="streamingAssetsRelativePath"/>
        /// (e.g. "model/connector_classifier.onnx"). On Android this is read
        /// out of the APK; on iOS it's a regular file on disk.
        /// </summary>
        public static SentisClassifier LoadFromStreamingAssets(string streamingAssetsRelativePath, string[] classNames)
        {
            var path = Path.Combine(Application.streamingAssetsPath, streamingAssetsRelativePath);
            byte[] bytes;
            if (path.Contains("://"))
            {
                // Android: APK contents need UnityWebRequest. Caller should
                // typically pre-stage StreamingAssets via the ModelUpdater
                // (downloads to persistentDataPath) so this path stays
                // synchronous. We fall back to File.ReadAllBytes if the URL
                // happens to be a real file URL on iOS.
                throw new InvalidOperationException(
                    "On Android, copy StreamingAssets into persistentDataPath at " +
                    "boot via UnityWebRequest, then call LoadFromBytes(...)");
            }
            bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes, classNames);
        }

        public ClassificationResult Classify(Texture2D rgb)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(SentisClassifier));
            if (rgb == null) throw new ArgumentNullException(nameof(rgb));

            // Resize + center-crop to 224×224 in CPU. For prod we'd do this on
            // GPU (RenderTexture + Graphics.Blit) to avoid the readback; for
            // the first build the synchronous CPU path is simpler and correct.
            using var input = TextureToTensor(rgb);

            _worker.Schedule(input);
            using var logits = _worker.PeekOutput() as Tensor<float>;
            if (logits == null)
                throw new InvalidOperationException("model output was not Tensor<float>");

            // Sentis tensors live on GPU by default; download to CPU before reading.
            using var cpuLogits = logits.ReadbackAndClone();
            var raw = cpuLogits.DownloadToArray();
            var probs = Softmax(raw);

            int top = 0;
            for (int i = 1; i < probs.Length; i++) if (probs[i] > probs[top]) top = i;
            string name = top < _classNames.Length ? _classNames[top] : $"class_{top}";
            return new ClassificationResult(top, name, probs[top], probs);
        }

        private Tensor<float> TextureToTensor(Texture2D src)
        {
            // Center-crop the source to a square, then resize to InputSize.
            int side = Mathf.Min(src.width, src.height);
            int cropX = (src.width - side) / 2;
            int cropY = (src.height - side) / 2;
            var pixels = src.GetPixels(cropX, cropY, side, side);
            var square = new Texture2D(side, side, TextureFormat.RGB24, false);
            square.SetPixels(pixels);
            square.Apply();

            // Now resample to InputSize × InputSize via RenderTexture.
            var rt = RenderTexture.GetTemporary(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(square, rt);
            var resized = new Texture2D(InputSize, InputSize, TextureFormat.RGB24, false);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            resized.ReadPixels(new Rect(0, 0, InputSize, InputSize), 0, 0);
            resized.Apply();
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);

            // Pack into NCHW [1, 3, H, W] in [0, 1]. Normalization is baked
            // into the ONNX graph so we don't apply mean/std here.
            var tensor = new Tensor<float>(_inputShape);
            var px = resized.GetPixels32();
            for (int y = 0; y < InputSize; y++)
            {
                for (int x = 0; x < InputSize; x++)
                {
                    var p = px[(InputSize - 1 - y) * InputSize + x]; // flip Y for image-coord origin
                    tensor[0, 0, y, x] = p.r / 255f;
                    tensor[0, 1, y, x] = p.g / 255f;
                    tensor[0, 2, y, x] = p.b / 255f;
                }
            }

            // Cleanup intermediates.
            UnityEngine.Object.Destroy(square);
            UnityEngine.Object.Destroy(resized);
            return tensor;
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
