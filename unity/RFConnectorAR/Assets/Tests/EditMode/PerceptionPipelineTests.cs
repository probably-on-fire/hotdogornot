using NUnit.Framework;
using RFConnectorAR.Perception;
using UnityEngine;

namespace RFConnectorAR.Tests.EditMode
{
    public class PerceptionPipelineTests
    {
        private static Texture2D Black() => new Texture2D(8, 8);

        [Test]
        public void RunFrame_StubStack_ProducesOneVerdictPerDetection()
        {
            var pipeline = new PerceptionPipeline(
                new StubDetector(),
                new StubEmbedder(),
                new StubMatcher(classId: 1, className: "SMA-F", cosine: 0.9f),
                new StubMeasurer(diameterMm: null),
                new ConfidenceFuser());

            var verdicts = pipeline.RunFrame(Black(), Black());

            Assert.AreEqual(1, verdicts.Length);
            Assert.AreEqual("SMA-F", verdicts[0].ClassName);
            Assert.AreEqual(ConfidenceLevel.High, verdicts[0].Confidence);
        }

        [Test]
        public void RunFrame_PrecisionWithAgreement_HighVerdict()
        {
            var pipeline = new PerceptionPipeline(
                new StubDetector(),
                new StubEmbedder(),
                new StubMatcher(classId: 4, className: "2.92mm-M", cosine: 0.85f),
                new StubMeasurer(diameterMm: 1.27f),
                new ConfidenceFuser());

            var verdicts = pipeline.RunFrame(Black(), Black());

            Assert.AreEqual("2.92mm-M", verdicts[0].ClassName);
            Assert.AreEqual(ConfidenceLevel.High, verdicts[0].Confidence);
            Assert.AreEqual(1.27f, verdicts[0].MeasuredDiameterMm);
        }

        [Test]
        public void RunFrame_NoDetections_EmptyArray()
        {
            var pipeline = new PerceptionPipeline(
                detector: new NoDetectionsDetector(),
                embedder: new StubEmbedder(),
                matcher: new StubMatcher(),
                measurer: new StubMeasurer(null),
                fuser: new ConfidenceFuser());

            var verdicts = pipeline.RunFrame(Black(), Black());
            Assert.AreEqual(0, verdicts.Length);
        }

        private sealed class NoDetectionsDetector : IDetector
        {
            public DetectionBox[] Detect(Texture2D frame) => System.Array.Empty<DetectionBox>();
        }
    }
}
