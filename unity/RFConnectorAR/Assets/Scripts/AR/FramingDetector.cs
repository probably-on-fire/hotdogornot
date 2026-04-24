using UnityEngine;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Fast "is there an object centred in the frame?" detector.
    ///
    /// This is a pragmatic stand-in for a real hex detector. The Python
    /// prototype at `rfconnectorai/measurement/hex_detector.py` is more
    /// accurate but needs OpenCV. Until we port contour-finding + polygon
    /// approximation to managed C# (or run a Sentis ONNX hex detector), this
    /// cheap heuristic does the job of "gate commit unless user is pointing
    /// at something centrally framed."
    ///
    /// Algorithm:
    ///   1. Downsample to 64×64 luminance on the CPU (small, fast).
    ///   2. Compute mean luminance of the outer ring (outside central ~40%)
    ///      and the central disc.
    ///   3. Framed when:
    ///        - outer ring is fairly uniform (low variance → clean background)
    ///        - central disc differs from ring by a clear margin (object vs bg)
    ///        - central disc occupies a reasonable fraction of the frame
    ///          (neither a tiny dot nor the whole image)
    ///
    /// This correctly separates "connector held at arm's length against a
    /// clean desk" from "user pointed camera at the floor" and from "full
    /// black screen / paper tight in frame" — the two failure modes we care
    /// about rejecting in the capture gate.
    /// </summary>
    public static class FramingDetector
    {
        public struct Result
        {
            public bool IsFramed;
            public float Score;           // 0..1, informal confidence
            public string Reason;
        }

        // Analysis thresholds — tuned empirically; may need revisiting after
        // real-device testing.
        private const int AnalysisSize = 64;
        // Center region as fraction of half-width. At 0.6 the center disc is
        // ~60% of the image radius — large enough that a typical frontal
        // connector shot fits inside it, leaving a thin clean-background ring
        // for the std-dev measurement.
        private const float CenterRadiusFrac = 0.6f;
        private const float MinContrast = 30f;            // luminance units (0..255)
        private const float MaxBgStdDev = 40f;            // outer ring max variance
        private const float MinObjectCoverage = 0.08f;    // fraction of center region that's "object"
        private const float MaxObjectCoverage = 0.95f;

        public static Result Analyze(Texture2D texture)
        {
            if (texture == null)
            {
                return new Result { IsFramed = false, Score = 0f, Reason = "null texture" };
            }

            var lumen = Downsample(texture, AnalysisSize);
            float centerR = AnalysisSize * 0.5f * CenterRadiusFrac;
            float centerR2 = centerR * centerR;
            float halfSide = AnalysisSize * 0.5f;

            float sumIn = 0, sumSqIn = 0; int nIn = 0;
            float sumOut = 0, sumSqOut = 0; int nOut = 0;

            for (int y = 0; y < AnalysisSize; y++)
            {
                for (int x = 0; x < AnalysisSize; x++)
                {
                    float dx = x - halfSide;
                    float dy = y - halfSide;
                    float r2 = dx * dx + dy * dy;
                    float v = lumen[y * AnalysisSize + x];
                    if (r2 <= centerR2)
                    {
                        sumIn += v; sumSqIn += v * v; nIn++;
                    }
                    else
                    {
                        sumOut += v; sumSqOut += v * v; nOut++;
                    }
                }
            }

            if (nIn == 0 || nOut == 0)
            {
                return new Result { IsFramed = false, Score = 0f, Reason = "degenerate regions" };
            }

            float meanIn = sumIn / nIn;
            float meanOut = sumOut / nOut;
            float varOut = (sumSqOut / nOut) - (meanOut * meanOut);
            float stdOut = Mathf.Sqrt(Mathf.Max(varOut, 0f));

            float contrast = Mathf.Abs(meanIn - meanOut);

            // Coverage: fraction of center pixels that are on the object side
            // of (meanIn+meanOut)/2 threshold.
            float midPoint = (meanIn + meanOut) * 0.5f;
            bool darkerInside = meanIn < meanOut;
            int objectPixels = 0;
            for (int y = 0; y < AnalysisSize; y++)
            {
                for (int x = 0; x < AnalysisSize; x++)
                {
                    float dx = x - halfSide;
                    float dy = y - halfSide;
                    float r2 = dx * dx + dy * dy;
                    if (r2 > centerR2) continue;
                    float v = lumen[y * AnalysisSize + x];
                    bool isObj = darkerInside ? (v < midPoint) : (v > midPoint);
                    if (isObj) objectPixels++;
                }
            }
            float coverage = (float)objectPixels / nIn;

            bool framed = contrast >= MinContrast
                       && stdOut <= MaxBgStdDev
                       && coverage >= MinObjectCoverage
                       && coverage <= MaxObjectCoverage;

            // Informal 0..1 score: weighted combination of the three signals.
            float contrastScore = Mathf.Clamp01(contrast / 120f);
            float stdScore = Mathf.Clamp01(1f - stdOut / MaxBgStdDev);
            float coverageScore = Mathf.Clamp01(
                (coverage - MinObjectCoverage) / (MaxObjectCoverage - MinObjectCoverage));
            float score = (contrastScore + stdScore + coverageScore) / 3f;

            string reason = framed ? "framed" :
                (contrast < MinContrast)       ? $"low contrast ({contrast:F0})" :
                (stdOut > MaxBgStdDev)         ? $"noisy background (std={stdOut:F0})" :
                (coverage < MinObjectCoverage) ? $"object too small (cov={coverage:F2})" :
                                                 $"object too large (cov={coverage:F2})";

            return new Result { IsFramed = framed, Score = score, Reason = reason };
        }

        /// <summary>
        /// Box-filter downsample texture → small grayscale byte array, in luminance.
        /// Avoids allocating multiple intermediate Texture2Ds.
        /// </summary>
        private static float[] Downsample(Texture2D src, int outSize)
        {
            var srcPixels = src.GetPixels32();
            int sw = src.width, sh = src.height;
            var outBuf = new float[outSize * outSize];

            for (int oy = 0; oy < outSize; oy++)
            {
                int sy0 = oy * sh / outSize;
                int sy1 = (oy + 1) * sh / outSize;
                for (int ox = 0; ox < outSize; ox++)
                {
                    int sx0 = ox * sw / outSize;
                    int sx1 = (ox + 1) * sw / outSize;
                    float sum = 0; int n = 0;
                    for (int yy = sy0; yy < sy1; yy++)
                    {
                        int rowBase = yy * sw;
                        for (int xx = sx0; xx < sx1; xx++)
                        {
                            var c = srcPixels[rowBase + xx];
                            // Rec. 601 luminance
                            sum += 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
                            n++;
                        }
                    }
                    outBuf[oy * outSize + ox] = (n > 0) ? sum / n : 0f;
                }
            }
            return outBuf;
        }
    }
}
