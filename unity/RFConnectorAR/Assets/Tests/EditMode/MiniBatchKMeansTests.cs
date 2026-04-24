using System;
using NUnit.Framework;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Tests.EditMode
{
    public class MiniBatchKMeansTests
    {
        [Test]
        public void Cluster_ReturnsRequestedK()
        {
            var rng = new System.Random(1);
            var data = new float[60][];
            for (int i = 0; i < 60; i++)
            {
                data[i] = new float[8];
                for (int j = 0; j < 8; j++) data[i][j] = (float)rng.NextDouble();
            }
            var k = 3;
            var protos = MiniBatchKMeans.Cluster(data, k: k, maxIters: 30, seed: 42);
            Assert.AreEqual(k, protos.Length);
            foreach (var p in protos) Assert.AreEqual(8, p.Length);
        }

        [Test]
        public void Cluster_RecoversObviousClusters()
        {
            var rng = new System.Random(7);
            var data = new float[60][];
            int idx = 0;
            float[][] centers = { new[] { 0f, 0f }, new[] { 5f, 5f }, new[] { -5f, 5f } };
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < 20; i++)
                {
                    data[idx++] = new float[]
                    {
                        centers[c][0] + (float)(rng.NextDouble() - 0.5) * 0.2f,
                        centers[c][1] + (float)(rng.NextDouble() - 0.5) * 0.2f,
                    };
                }
            }

            var protos = MiniBatchKMeans.Cluster(data, k: 3, maxIters: 50, seed: 1);

            foreach (var center in centers)
            {
                bool matched = false;
                foreach (var p in protos)
                {
                    float dx = p[0] - center[0], dy = p[1] - center[1];
                    if (Math.Sqrt(dx * dx + dy * dy) < 0.5f) { matched = true; break; }
                }
                Assert.IsTrue(matched, $"No prototype near ({center[0]},{center[1]})");
            }
        }

        [Test]
        public void Cluster_KGreaterThanData_ReturnsAllPointsAsPrototypes()
        {
            var data = new[]
            {
                new[] { 1f, 0f },
                new[] { 0f, 1f },
            };
            var protos = MiniBatchKMeans.Cluster(data, k: 5, maxIters: 10, seed: 0);
            Assert.AreEqual(2, protos.Length);
        }

        [Test]
        public void Cluster_RejectsEmptyInput()
        {
            Assert.Throws<ArgumentException>(
                () => MiniBatchKMeans.Cluster(new float[0][], k: 3, maxIters: 10, seed: 0));
        }
    }
}
