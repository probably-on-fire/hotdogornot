using NUnit.Framework;
using RFConnectorAR.AR;
using UnityEngine;

namespace RFConnectorAR.Tests.EditMode
{
    public class FramingDetectorTests
    {
        /// <summary>Make a grayscale texture with a dark square of given size centred on it.</summary>
        private static Texture2D MakeCenteredDark(int size, int darkSidePx, byte darkVal = 40, byte bgVal = 220)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
            var pixels = new Color32[size * size];
            int lo = (size - darkSidePx) / 2;
            int hi = lo + darkSidePx;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    byte v = (x >= lo && x < hi && y >= lo && y < hi) ? darkVal : bgVal;
                    pixels[y * size + x] = new Color32(v, v, v, 255);
                }
            tex.SetPixels32(pixels);
            tex.Apply();
            return tex;
        }

        /// <summary>Make a uniform-color texture (no object in view).</summary>
        private static Texture2D MakeUniform(int size, byte val = 220)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
            var pixels = new Color32[size * size];
            for (int i = 0; i < pixels.Length; i++) pixels[i] = new Color32(val, val, val, 255);
            tex.SetPixels32(pixels);
            tex.Apply();
            return tex;
        }

        [Test]
        public void Analyze_ReportsFramedTrueForCenteredObject()
        {
            var tex = MakeCenteredDark(size: 256, darkSidePx: 120);
            var result = FramingDetector.Analyze(tex);
            Assert.IsTrue(result.IsFramed,
                $"Expected IsFramed=true, got score={result.Score} reason={result.Reason}");
            Assert.Greater(result.Score, 0.5f);
        }

        [Test]
        public void Analyze_ReportsFramedFalseForUniformBackground()
        {
            var tex = MakeUniform(size: 256);
            var result = FramingDetector.Analyze(tex);
            Assert.IsFalse(result.IsFramed);
            Assert.Less(result.Score, 0.5f);
        }

        [Test]
        public void Analyze_ReportsFramedFalseWhenObjectTooSmall()
        {
            // A tiny dark dot doesn't look like a centred connector face.
            var tex = MakeCenteredDark(size: 256, darkSidePx: 20);
            var result = FramingDetector.Analyze(tex);
            Assert.IsFalse(result.IsFramed);
        }

        [Test]
        public void Analyze_ReportsFramedFalseWhenObjectFillsEntireFrame()
        {
            // A dark texture filling the whole frame is "all object, no bg" — reject.
            var tex = MakeCenteredDark(size: 256, darkSidePx: 250);
            var result = FramingDetector.Analyze(tex);
            Assert.IsFalse(result.IsFramed);
        }

        [Test]
        public void Analyze_RejectsNullTexture()
        {
            var result = FramingDetector.Analyze(null);
            Assert.IsFalse(result.IsFramed);
            Assert.AreEqual(0f, result.Score);
        }
    }
}
