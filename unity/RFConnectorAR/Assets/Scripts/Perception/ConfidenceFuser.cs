using System.Collections.Generic;
using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// Combines ML classifier + physical measurement into a verdict.
    ///
    /// Philosophy (from the design spec): honesty over false precision.
    /// Precision connectors require both ML and measurement agreement to
    /// commit to a HIGH verdict. Missing depth degrades gracefully to MEDIUM.
    /// Clearly-disagreeing measurement demotes to LOW.
    /// </summary>
    public sealed class ConfidenceFuser
    {
        private const float ScoreHighCutoff = 0.80f;
        private const float ScoreMediumCutoff = 0.60f;
        private const float ScoreUnknownCutoff = 0.40f;

        private static readonly Dictionary<string, float> ExpectedPinMm = new()
        {
            { "SMA-M",    0.91f },
            { "SMA-F",    1.27f },
            { "3.5mm-M",  1.52f },
            { "3.5mm-F",  1.52f },
            { "2.92mm-M", 1.27f },
            { "2.92mm-F", 1.27f },
            { "2.4mm-M",  1.04f },
            { "2.4mm-F",  1.04f },
        };

        private const float AgreementToleranceMm = 0.25f;

        public Verdict Fuse(
            Match match,
            float? measuredDiameterMm,
            DetectionBox detectionBox,
            Vector3? worldPosition)
        {
            float score01 = (match.CosineSimilarity + 1f) * 0.5f;

            if (match.ClassId < 0 || score01 < ScoreUnknownCutoff)
            {
                return new Verdict
                {
                    ClassName = "Unknown",
                    ClassId = -1,
                    Confidence = ConfidenceLevel.Unknown,
                    Score = score01,
                    MeasuredDiameterMm = measuredDiameterMm,
                    DetectionBox = detectionBox,
                    WorldPosition = worldPosition,
                };
            }

            bool isPrecision = match.ClassName.Contains("mm");

            ConfidenceLevel confidence;
            if (isPrecision)
            {
                if (measuredDiameterMm is float mm
                    && ExpectedPinMm.TryGetValue(match.ClassName, out float expected))
                {
                    float delta = Mathf.Abs(mm - expected);
                    confidence = delta <= AgreementToleranceMm
                        ? ConfidenceLevel.High
                        : ConfidenceLevel.Low;
                }
                else
                {
                    confidence = ConfidenceLevel.Medium;
                }
            }
            else
            {
                confidence =
                    score01 >= ScoreHighCutoff   ? ConfidenceLevel.High :
                    score01 >= ScoreMediumCutoff ? ConfidenceLevel.Medium :
                                                   ConfidenceLevel.Low;
            }

            return new Verdict
            {
                ClassName = match.ClassName,
                ClassId = match.ClassId,
                Confidence = confidence,
                Score = score01,
                MeasuredDiameterMm = measuredDiameterMm,
                DetectionBox = detectionBox,
                WorldPosition = worldPosition,
            };
        }
    }
}
