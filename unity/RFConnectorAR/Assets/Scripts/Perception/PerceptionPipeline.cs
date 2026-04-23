using UnityEngine;

namespace RFConnectorAR.Perception
{
    public sealed class PerceptionPipeline
    {
        private readonly IDetector _detector;
        private readonly IEmbedder _embedder;
        private readonly IMatcher _matcher;
        private readonly IMeasurer _measurer;
        private readonly ConfidenceFuser _fuser;

        public PerceptionPipeline(
            IDetector detector,
            IEmbedder embedder,
            IMatcher matcher,
            IMeasurer measurer,
            ConfidenceFuser fuser)
        {
            _detector = detector;
            _embedder = embedder;
            _matcher = matcher;
            _measurer = measurer;
            _fuser = fuser;
        }

        public Verdict[] RunFrame(Texture2D rgbFrame, Texture2D depthFrame)
        {
            var detections = _detector.Detect(rgbFrame);
            if (detections.Length == 0) return System.Array.Empty<Verdict>();

            var verdicts = new Verdict[detections.Length];
            for (int i = 0; i < detections.Length; i++)
            {
                var box = detections[i];
                var rgbCrop = CropTexture(rgbFrame, box.NormalizedRect);
                var depthCrop = depthFrame != null ? CropTexture(depthFrame, box.NormalizedRect) : null;

                var embedding = _embedder.Embed(rgbCrop, depthCrop);
                var match = _matcher.MatchTop1(embedding);
                var measured = _measurer.MeasureInnerPinDiameterMm(rgbCrop, depthCrop);

                verdicts[i] = _fuser.Fuse(
                    match: match,
                    measuredDiameterMm: measured,
                    detectionBox: box,
                    worldPosition: null);
            }
            return verdicts;
        }

        private static Texture2D CropTexture(Texture2D src, Rect normalized)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(normalized.xMin * src.width), 0, src.width - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt(normalized.yMin * src.height), 0, src.height - 1);
            int w = Mathf.Clamp(Mathf.RoundToInt(normalized.width * src.width), 1, src.width - x);
            int h = Mathf.Clamp(Mathf.RoundToInt(normalized.height * src.height), 1, src.height - y);

            var pixels = src.GetPixels(x, y, w, h);
            var crop = new Texture2D(w, h, src.format, false);
            crop.SetPixels(pixels);
            crop.Apply();
            return crop;
        }
    }
}
