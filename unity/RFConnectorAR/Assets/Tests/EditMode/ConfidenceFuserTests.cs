using NUnit.Framework;
using RFConnectorAR.Perception;

namespace RFConnectorAR.Tests.EditMode
{
    public class ConfidenceFuserTests
    {
        private static readonly ConfidenceFuser Fuser = new();

        [Test]
        public void SmaClass_NoMeasurement_HighConfidenceWhenScoreAboveThreshold()
        {
            var verdict = Fuser.Fuse(
                match: new Match(classId: 1, className: "SMA-F", cosineSimilarity: 0.92f),
                measuredDiameterMm: null,
                detectionBox: default,
                worldPosition: null);

            Assert.AreEqual(ConfidenceLevel.High, verdict.Confidence);
            Assert.AreEqual("SMA-F", verdict.ClassName);
        }

        [Test]
        public void SmaClass_LowScore_MediumOrLow()
        {
            var verdict = Fuser.Fuse(
                match: new Match(1, "SMA-F", cosineSimilarity: 0.55f),
                measuredDiameterMm: null,
                detectionBox: default,
                worldPosition: null);

            Assert.That(
                verdict.Confidence,
                Is.EqualTo(ConfidenceLevel.Medium).Or.EqualTo(ConfidenceLevel.Low));
        }

        [Test]
        public void PrecisionClass_MeasurementAgrees_HighConfidence()
        {
            var verdict = Fuser.Fuse(
                match: new Match(4, "2.92mm-M", 0.88f),
                measuredDiameterMm: 1.27f,
                detectionBox: default,
                worldPosition: null);

            Assert.AreEqual(ConfidenceLevel.High, verdict.Confidence);
            Assert.AreEqual("2.92mm-M", verdict.ClassName);
            Assert.AreEqual(1.27f, verdict.MeasuredDiameterMm);
        }

        [Test]
        public void PrecisionClass_MeasurementDisagrees_LowConfidence()
        {
            var verdict = Fuser.Fuse(
                match: new Match(6, "2.4mm-M", 0.91f),
                measuredDiameterMm: 1.52f,
                detectionBox: default,
                worldPosition: null);

            Assert.AreEqual(ConfidenceLevel.Low, verdict.Confidence);
        }

        [Test]
        public void PrecisionClass_NoDepth_MediumConfidence()
        {
            var verdict = Fuser.Fuse(
                match: new Match(4, "2.92mm-M", 0.88f),
                measuredDiameterMm: null,
                detectionBox: default,
                worldPosition: null);

            Assert.AreEqual(ConfidenceLevel.Medium, verdict.Confidence);
        }

        [Test]
        public void UnknownMatch_ReturnsUnknownVerdict()
        {
            var verdict = Fuser.Fuse(
                match: new Match(-1, "Unknown", 0.2f),
                measuredDiameterMm: null,
                detectionBox: default,
                worldPosition: null);

            Assert.AreEqual(ConfidenceLevel.Unknown, verdict.Confidence);
        }
    }
}
