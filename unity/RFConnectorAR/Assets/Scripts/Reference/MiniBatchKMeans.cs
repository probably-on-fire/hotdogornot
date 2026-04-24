using System;
using System.Collections.Generic;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Mini-batch k-means for clustering enrollment embeddings into per-class
    /// prototypes. Operates on raw float[] vectors; uses Euclidean distance
    /// (appropriate for L2-normalized embeddings from RGBDEmbedder — on the
    /// unit hypersphere Euclidean distance is monotone with cosine distance).
    /// </summary>
    public static class MiniBatchKMeans
    {
        public static float[][] Cluster(
            float[][] data, int k, int maxIters = 50, int batchSize = 32, int seed = 0)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("data must be non-empty");
            if (k <= 0) throw new ArgumentException("k must be > 0");

            int n = data.Length;
            int dim = data[0].Length;
            int actualK = Math.Min(k, n);

            var rng = new Random(seed);
            var indices = new List<int>(n);
            for (int i = 0; i < n; i++) indices.Add(i);
            for (int i = 0; i < actualK; i++)
            {
                int j = rng.Next(i, n);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            var centroids = new float[actualK][];
            for (int i = 0; i < actualK; i++)
            {
                centroids[i] = new float[dim];
                Array.Copy(data[indices[i]], centroids[i], dim);
            }

            var counts = new int[actualK];

            for (int iter = 0; iter < maxIters; iter++)
            {
                int bs = Math.Min(batchSize, n);
                for (int b = 0; b < bs; b++)
                {
                    int sampleIdx = rng.Next(n);
                    var x = data[sampleIdx];
                    int nearest = NearestCentroidIndex(x, centroids);
                    counts[nearest]++;
                    float lr = 1f / counts[nearest];
                    var c = centroids[nearest];
                    for (int d = 0; d < dim; d++)
                    {
                        c[d] = c[d] + lr * (x[d] - c[d]);
                    }
                }
            }

            return centroids;
        }

        private static int NearestCentroidIndex(float[] x, float[][] centroids)
        {
            int best = 0;
            float bestDist = float.MaxValue;
            for (int i = 0; i < centroids.Length; i++)
            {
                float d = SquaredEuclidean(x, centroids[i]);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            return best;
        }

        private static float SquaredEuclidean(float[] a, float[] b)
        {
            float sum = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }
    }
}
