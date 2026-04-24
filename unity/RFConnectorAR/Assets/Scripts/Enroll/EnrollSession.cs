using System;
using System.Collections.Generic;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Enroll
{
    /// <summary>
    /// Stateful capture buffer for one enrollment. Caller pushes embeddings
    /// (typically one per frame from EnrollController), the session caps at
    /// `targetFrames`, then `Finalize()` clusters into K prototypes.
    /// </summary>
    public sealed class EnrollSession
    {
        public int TargetFrames { get; }
        public int K { get; }
        private readonly List<float[]> _embeddings;

        public EnrollSession(int targetFrames, int k)
        {
            if (targetFrames <= 0) throw new ArgumentException("targetFrames must be > 0");
            if (k <= 0) throw new ArgumentException("k must be > 0");
            TargetFrames = targetFrames;
            K = k;
            _embeddings = new List<float[]>(targetFrames);
        }

        public int CapturedCount => _embeddings.Count;
        public bool IsComplete => _embeddings.Count >= TargetFrames;
        public float Progress01 => (float)_embeddings.Count / TargetFrames;

        public void Push(float[] embedding)
        {
            if (IsComplete) return;
            _embeddings.Add(embedding);
        }

        public float[][] Finalize()
        {
            if (_embeddings.Count == 0)
                throw new InvalidOperationException("Finalize called on empty session");
            return MiniBatchKMeans.Cluster(_embeddings.ToArray(), k: K, maxIters: 50, seed: 0);
        }
    }
}
