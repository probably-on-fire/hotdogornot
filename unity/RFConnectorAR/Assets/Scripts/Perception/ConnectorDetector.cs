using System.Collections.Generic;
using UnityEngine;

namespace RFConnectorAR.Perception
{
    /// <summary>
    /// On-device connector blob detector. C# port of the Python
    /// <c>detect_connector_crops</c> function, using only Unity-native
    /// pixel operations (no OpenCV dependency).
    ///
    /// Algorithm:
    ///   1. Downsample the camera frame to ~640px on the long side for speed.
    ///   2. Compute a global Otsu threshold on the grayscale image.
    ///   3. Flood-fill / connected-components on the binary mask to find
    ///      bright blobs.
    ///   4. Filter blobs by area (drops noise + the wood background) and
    ///      aspect ratio (drops wood-grain line artifacts).
    ///   5. Return per-blob square padded crops, scaled back to original
    ///      frame coordinates.
    ///
    /// Designed to run at 5-10 Hz on a phone. The classifier feeds on the
    /// returned crops, not the full camera frame — same train/inference
    /// distribution as the labeler produces.
    /// </summary>
    public static class ConnectorDetector
    {
        public struct Detection
        {
            public RectInt BBox;          // contour bounding box on the source frame
            public RectInt PaddedCrop;    // padded square crop region
            public Vector2Int Center;
            public int AreaPx;
        }

        private const int DetectMaxLongSide = 640;
        private const float MinAreaFrac = 0.001f;
        private const float MaxAreaFrac = 0.10f;
        private const float PadFrac = 0.35f;
        private const float MaxAspect = 2.5f;
        private const int DefaultMaxDetections = 4;

        public static List<Detection> Detect(Texture2D src, int maxDetections = DefaultMaxDetections)
        {
            int w = src.width, h = src.height;
            int longSide = Mathf.Max(w, h);
            float scale = longSide > DetectMaxLongSide
                ? (float)DetectMaxLongSide / longSide
                : 1f;
            int dw = Mathf.Max(1, Mathf.RoundToInt(w * scale));
            int dh = Mathf.Max(1, Mathf.RoundToInt(h * scale));

            // Downsample to grayscale array via simple nearest-neighbor.
            byte[] gray = DownsampleGray(src, dw, dh);

            // Otsu threshold.
            int threshold = OtsuThreshold(gray);

            // Binary mask: 1 if pixel >= threshold else 0.
            byte[] mask = new byte[gray.Length];
            for (int i = 0; i < gray.Length; i++)
                mask[i] = (byte)(gray[i] >= threshold ? 1 : 0);

            // Connected-components labelling (4-neighbor) to find blobs.
            var blobs = ConnectedComponents(mask, dw, dh);

            int totalArea = dw * dh;
            int minArea = Mathf.RoundToInt(totalArea * MinAreaFrac);
            int maxArea = Mathf.RoundToInt(totalArea * MaxAreaFrac);

            var detections = new List<Detection>();
            foreach (var b in blobs)
            {
                if (b.area < minArea || b.area > maxArea) continue;
                float aspect = Mathf.Max(b.width, b.height) /
                               (float)Mathf.Max(1, Mathf.Min(b.width, b.height));
                if (aspect > MaxAspect) continue;

                // Map back to original coords.
                int origX = Mathf.RoundToInt(b.x / scale);
                int origY = Mathf.RoundToInt(b.y / scale);
                int origW = Mathf.RoundToInt(b.width / scale);
                int origH = Mathf.RoundToInt(b.height / scale);
                int origArea = Mathf.RoundToInt(b.area / (scale * scale));

                int side = Mathf.RoundToInt(Mathf.Max(origW, origH) * (1 + 2 * PadFrac));
                int cx = origX + origW / 2;
                int cy = origY + origH / 2;
                int x0 = Mathf.Clamp(cx - side / 2, 0, Mathf.Max(0, w - 1));
                int y0 = Mathf.Clamp(cy - side / 2, 0, Mathf.Max(0, h - 1));
                int x1 = Mathf.Clamp(x0 + side, 0, w);
                int y1 = Mathf.Clamp(y0 + side, 0, h);
                if (x1 - x0 < side) x0 = Mathf.Max(0, x1 - side);
                if (y1 - y0 < side) y0 = Mathf.Max(0, y1 - side);

                detections.Add(new Detection
                {
                    BBox = new RectInt(origX, origY, origW, origH),
                    PaddedCrop = new RectInt(x0, y0, x1 - x0, y1 - y0),
                    Center = new Vector2Int(cx, cy),
                    AreaPx = origArea,
                });
            }

            // Keep largest N
            detections.Sort((a, b) => b.AreaPx.CompareTo(a.AreaPx));
            if (detections.Count > maxDetections)
                detections.RemoveRange(maxDetections, detections.Count - maxDetections);
            return detections;
        }

        /// <summary>
        /// Crop the given source texture to the detection's padded region.
        /// Caller is responsible for destroying the returned texture.
        /// </summary>
        public static Texture2D CropTexture(Texture2D src, RectInt region)
        {
            var pixels = src.GetPixels(region.x, region.y, region.width, region.height);
            var crop = new Texture2D(region.width, region.height, TextureFormat.RGB24, false);
            crop.SetPixels(pixels);
            crop.Apply();
            return crop;
        }

        // -------- Helpers --------

        private static byte[] DownsampleGray(Texture2D src, int dw, int dh)
        {
            var pixels = src.GetPixels32();
            int sw = src.width, sh = src.height;
            byte[] gray = new byte[dw * dh];
            for (int y = 0; y < dh; y++)
            {
                int srcY = (int)((y + 0.5f) * sh / dh);
                if (srcY >= sh) srcY = sh - 1;
                for (int x = 0; x < dw; x++)
                {
                    int srcX = (int)((x + 0.5f) * sw / dw);
                    if (srcX >= sw) srcX = sw - 1;
                    var p = pixels[srcY * sw + srcX];
                    // Rec.709 luma
                    gray[y * dw + x] = (byte)((p.r * 54 + p.g * 183 + p.b * 19) >> 8);
                }
            }
            return gray;
        }

        private static int OtsuThreshold(byte[] gray)
        {
            int[] hist = new int[256];
            for (int i = 0; i < gray.Length; i++) hist[gray[i]]++;
            int total = gray.Length;
            float sum = 0;
            for (int i = 0; i < 256; i++) sum += i * hist[i];

            float sumB = 0, wB = 0, varMax = 0;
            int threshold = 127;
            for (int t = 0; t < 256; t++)
            {
                wB += hist[t];
                if (wB == 0) continue;
                float wF = total - wB;
                if (wF == 0) break;
                sumB += (float)t * hist[t];
                float mB = sumB / wB;
                float mF = (sum - sumB) / wF;
                float between = wB * wF * (mB - mF) * (mB - mF);
                if (between > varMax)
                {
                    varMax = between;
                    threshold = t;
                }
            }
            return threshold;
        }

        private struct BlobBounds
        {
            public int x, y, width, height, area;
        }

        private static List<BlobBounds> ConnectedComponents(byte[] mask, int w, int h)
        {
            // Iterative flood-fill with 4-connectivity.
            int[] labels = new int[mask.Length];
            var blobs = new List<BlobBounds>();
            var stack = new Stack<int>();
            int nextLabel = 1;
            for (int i = 0; i < mask.Length; i++)
            {
                if (mask[i] == 0 || labels[i] != 0) continue;
                stack.Clear();
                stack.Push(i);
                int minX = w, minY = h, maxX = 0, maxY = 0, area = 0;
                while (stack.Count > 0)
                {
                    int idx = stack.Pop();
                    if (idx < 0 || idx >= mask.Length) continue;
                    if (mask[idx] == 0 || labels[idx] != 0) continue;
                    labels[idx] = nextLabel;
                    int px = idx % w, py = idx / w;
                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (py < minY) minY = py;
                    if (py > maxY) maxY = py;
                    area++;
                    if (px > 0)     stack.Push(idx - 1);
                    if (px < w - 1) stack.Push(idx + 1);
                    if (py > 0)     stack.Push(idx - w);
                    if (py < h - 1) stack.Push(idx + w);
                }
                blobs.Add(new BlobBounds
                {
                    x = minX, y = minY,
                    width = maxX - minX + 1,
                    height = maxY - minY + 1,
                    area = area,
                });
                nextLabel++;
            }
            return blobs;
        }
    }
}
