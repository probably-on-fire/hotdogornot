# Unity Enroll + Curate Implementation Plan (Plan 2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the on-device enroll architecture to the Unity scanner. A tech can teach the app any connector by holding it up in front of the camera; the app captures frames, embeds them, clusters into per-class prototypes, and writes the local reference database. A curate scene lets the tech browse, delete, and re-enroll. Identification mode (the existing scanner from Plan 2) reads from the same on-device DB.

**Architecture:** Three Unity scenes — Scanner (existing), Enroll (new), Curate (new) — share a single `OnDeviceReferenceStore` that persists the reference database to `Application.persistentDataPath`. Enroll captures frames at 30 FPS for 5 seconds, runs each through the perception pipeline (stub initially, real ONNX in Plan 3) to get embeddings, and clusters with mini-batch k-means (K=3 by default) into per-class prototypes. The RFCE binary format gains a v2 variant with `vectors_per_class > 1`. The Plan 2 `ReferenceDatabase` reader is extended to handle both v1 and v2.

**Tech Stack:** Unity 6.0 LTS, AR Foundation 6.x, Sentis 2.x, Unity Test Framework. Spec references: `docs/superpowers/specs/2026-04-23-rf-connector-ar-design.md` + amendment `docs/superpowers/specs/2026-04-24-on-device-enroll-amendment.md`. Previous plan: `docs/superpowers/plans/2026-04-23-unity-scanner-mvp.md`.

---

## File Structure

```
unity/RFConnectorAR/
├── Assets/
│   ├── Scenes/
│   │   ├── Scanner.unity        (existing)
│   │   ├── Enroll.unity         (new)
│   │   └── Curate.unity         (new)
│   ├── Scripts/
│   │   ├── Reference/
│   │   │   ├── ReferenceDatabase.cs           (extend: support v2 format)
│   │   │   ├── ReferenceDatabaseWriter.cs     (new: write RFCE v2)
│   │   │   ├── OnDeviceReferenceStore.cs      (new: read+write+update DB on disk)
│   │   │   └── MiniBatchKMeans.cs             (new: cluster embeddings → prototypes)
│   │   ├── Enroll/
│   │   │   ├── EnrollSession.cs               (new: capture + embed loop)
│   │   │   ├── EnrollController.cs            (new: scene controller / state machine)
│   │   │   └── EnrollHUD.cs                   (new: progress + guidance UI)
│   │   ├── Curate/
│   │   │   ├── CurateController.cs            (new: list classes, browse references)
│   │   │   └── CurateHUD.cs                   (new: list + delete UI)
│   │   ├── App/
│   │   │   └── ModeRouter.cs                  (new: switch between Scanner/Enroll/Curate)
│   │   └── ...                                (existing files unchanged)
│   └── Tests/
│       └── EditMode/
│           ├── ReferenceDatabaseV2Tests.cs    (new)
│           ├── MiniBatchKMeansTests.cs        (new)
│           └── OnDeviceReferenceStoreTests.cs (new)
```

---

## Task 1: Extend RFCE binary format (v2 with multiple prototypes per class)

The Plan 1 / Plan 2 reader handles `FORMAT_VERSION = 1` with one mean vector per class. v2 stores N vectors per class (default N=3 for enrolled classes). Backward compatible: v1 files still load (each class has 1 prototype).

**Files:**
- Modify: `Assets/Scripts/Reference/ReferenceDatabase.cs`
- Create: `Assets/Tests/EditMode/ReferenceDatabaseV2Tests.cs`

- [ ] **Step 1: Write failing test `Assets/Tests/EditMode/ReferenceDatabaseV2Tests.cs`**

```csharp
using System;
using System.IO;
using NUnit.Framework;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Tests.EditMode
{
    public class ReferenceDatabaseV2Tests
    {
        // Writes a v2 binary: per-class records have a u32 count followed by
        // count × dim float32 prototype vectors.
        private static string WriteV2(int dim, params (int id, string name, float[][] vecs)[] entries)
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            using var w = new BinaryWriter(File.Create(path));
            w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
            w.Write((uint)2);                       // version
            w.Write((uint)entries.Length);
            w.Write((uint)dim);
            foreach (var e in entries)
            {
                w.Write(e.id);
                var name = new byte[64];
                var src = System.Text.Encoding.UTF8.GetBytes(e.name);
                Array.Copy(src, name, Math.Min(src.Length, 64));
                w.Write(name);
                w.Write((uint)e.vecs.Length);
                foreach (var v in e.vecs)
                    foreach (var f in v) w.Write(f);
            }
            return path;
        }

        [Test]
        public void Load_V2_WithMultiplePrototypesPerClass()
        {
            var path = WriteV2(
                dim: 3,
                (0, "SMA-M", new[] {
                    new[] { 1f, 0f, 0f },
                    new[] { 0.99f, 0.1f, 0f },   // slight variant of SMA-M
                }),
                (1, "SMA-F", new[] {
                    new[] { 0f, 1f, 0f },
                }));

            var db = ReferenceDatabase.Load(path);
            Assert.AreEqual(2, db.Count);
            Assert.AreEqual(3, db.EmbeddingDim);
        }

        [Test]
        public void MatchTop1_V2_PicksBestPrototypeAcrossClasses()
        {
            var path = WriteV2(
                dim: 3,
                (0, "SMA-M", new[] {
                    new[] { 1f, 0f, 0f },
                    new[] { 0.5f, 0.5f, 0.7f },  // a "side view" of SMA-M
                }),
                (1, "SMA-F", new[] {
                    new[] { 0f, 1f, 0f },
                }));

            var db = ReferenceDatabase.Load(path);

            // Query close to the second SMA-M prototype:
            var match = db.MatchTop1(new[] { 0.5f, 0.45f, 0.7f });
            Assert.AreEqual(0, match.ClassId);
            Assert.AreEqual("SMA-M", match.ClassName);
        }

        [Test]
        public void Load_V1_StillWorks()
        {
            // v1: same structure but no per-class count, exactly one vector each
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            using (var w = new BinaryWriter(File.Create(path)))
            {
                w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
                w.Write((uint)1);
                w.Write((uint)1);
                w.Write((uint)2);
                w.Write(0);
                var name = new byte[64];
                var src = System.Text.Encoding.UTF8.GetBytes("SMA-M");
                Array.Copy(src, name, src.Length);
                w.Write(name);
                w.Write(1f); w.Write(0f);
            }

            var db = ReferenceDatabase.Load(path);
            Assert.AreEqual(1, db.Count);
            var match = db.MatchTop1(new[] { 1f, 0f });
            Assert.AreEqual("SMA-M", match.ClassName);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```
"/e/unity/6000.0.73f1/Editor/Unity.exe" -batchmode -projectPath "E:\anduril\unity\RFConnectorAR" -runTests -testPlatform EditMode -testResults "E:\anduril\unity\RFConnectorAR\test-results.xml" -logFile - -nographics
```

Expected: the v2 tests fail (the loader only knows v1). The existing v1 test still passes.

- [ ] **Step 3: Update `Assets/Scripts/Reference/ReferenceDatabase.cs` to support both v1 and v2**

Replace the current implementation with:

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using RFConnectorAR.Perception;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Reads RFCE-format reference embeddings. Supports format v1 (one mean
    /// vector per class) and v2 (N prototypes per class).
    ///
    /// File format (little-endian):
    ///   magic  : 4 bytes ASCII "RFCE"
    ///   ver    : u32  (1 or 2)
    ///   count  : u32 number of classes
    ///   dim    : u32 embedding dimension
    ///   then `count` records:
    ///     v1: id (i32) + name (64 bytes UTF-8 null-padded) + vector (dim × f32)
    ///     v2: id (i32) + name (64 bytes UTF-8 null-padded)
    ///         + n_prototypes (u32) + n_prototypes × (dim × f32)
    /// </summary>
    public sealed class ReferenceDatabase : IMatcher
    {
        private readonly int[] _ids;
        private readonly string[] _names;
        // For v2 each class can have multiple prototype vectors. For v1, _vectors[i]
        // has exactly one entry.
        private readonly float[][][] _vectors;

        public int Count => _ids.Length;
        public int EmbeddingDim { get; }

        private ReferenceDatabase(int[] ids, string[] names, float[][][] vectors, int dim)
        {
            _ids = ids; _names = names; _vectors = vectors; EmbeddingDim = dim;
        }

        public static ReferenceDatabase Load(string path)
        {
            using var f = File.OpenRead(path);
            using var r = new BinaryReader(f);

            var magic = r.ReadBytes(4);
            if (magic.Length != 4 ||
                magic[0] != 'R' || magic[1] != 'F' ||
                magic[2] != 'C' || magic[3] != 'E')
            {
                throw new InvalidDataException($"Not an RFCE reference file: {path}");
            }

            uint version = r.ReadUInt32();
            if (version != 1 && version != 2)
            {
                throw new InvalidDataException($"Unsupported RFCE version {version} in {path}");
            }

            uint count = r.ReadUInt32();
            int dim = (int)r.ReadUInt32();

            var ids = new int[count];
            var names = new string[count];
            var vectors = new float[count][][];

            for (int i = 0; i < count; i++)
            {
                ids[i] = r.ReadInt32();
                var nameBytes = r.ReadBytes(64);
                int nameLen = Array.IndexOf<byte>(nameBytes, 0);
                if (nameLen < 0) nameLen = 64;
                names[i] = System.Text.Encoding.UTF8.GetString(nameBytes, 0, nameLen);

                int nProto = (version == 2) ? (int)r.ReadUInt32() : 1;
                vectors[i] = new float[nProto][];
                for (int p = 0; p < nProto; p++)
                {
                    var v = new float[dim];
                    for (int j = 0; j < dim; j++) v[j] = r.ReadSingle();
                    vectors[i][p] = v;
                }
            }

            return new ReferenceDatabase(ids, names, vectors, dim);
        }

        public Match MatchTop1(float[] embedding)
        {
            if (embedding == null || embedding.Length != EmbeddingDim)
            {
                throw new ArgumentException(
                    $"embedding length {(embedding?.Length ?? 0)} != DB dim {EmbeddingDim}");
            }

            int bestIdx = 0;
            float bestScore = float.MinValue;
            for (int i = 0; i < _vectors.Length; i++)
            {
                foreach (var proto in _vectors[i])
                {
                    float s = CosineSimilarity(embedding, proto);
                    if (s > bestScore)
                    {
                        bestScore = s;
                        bestIdx = i;
                    }
                }
            }

            return new Match(_ids[bestIdx], _names[bestIdx], bestScore);
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0f, na = 0f, nb = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            float denom = (float)Math.Sqrt(na) * (float)Math.Sqrt(nb);
            if (denom <= 0f) return 0f;
            return dot / denom;
        }
    }
}
```

- [ ] **Step 4: Run test; verify all pass (existing 17 + 3 new = 20)**

- [ ] **Step 5: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Reference/ReferenceDatabase.cs unity/RFConnectorAR/Assets/Tests/EditMode/ReferenceDatabaseV2Tests.cs unity/RFConnectorAR/Assets/Tests/EditMode/ReferenceDatabaseV2Tests.cs.meta
git commit -m "feat(unity): RFCE v2 binary format with per-class prototypes"
```

---

## Task 2: ReferenceDatabaseWriter

Writes RFCE v2 binaries. Used by the on-device store when persisting enrollments.

**Files:**
- Create: `Assets/Scripts/Reference/ReferenceDatabaseWriter.cs`
- Create: `Assets/Tests/EditMode/ReferenceDatabaseWriterTests.cs`

- [ ] **Step 1: Write failing test `Assets/Tests/EditMode/ReferenceDatabaseWriterTests.cs`**

```csharp
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Tests.EditMode
{
    public class ReferenceDatabaseWriterTests
    {
        [Test]
        public void Write_RoundTripsThroughLoad()
        {
            var path = Path.Combine(Path.GetTempPath(), System.Guid.NewGuid().ToString("N") + ".bin");

            var entries = new List<EnrolledClass>
            {
                new EnrolledClass(id: 0, name: "SMA-M", prototypes: new[] {
                    new[] { 1f, 0f, 0f },
                    new[] { 0.95f, 0.31f, 0f },
                }),
                new EnrolledClass(id: 1, name: "SMA-F", prototypes: new[] {
                    new[] { 0f, 1f, 0f },
                }),
            };
            ReferenceDatabaseWriter.Write(path, entries, embeddingDim: 3);

            var db = ReferenceDatabase.Load(path);
            Assert.AreEqual(2, db.Count);
            Assert.AreEqual(3, db.EmbeddingDim);

            var smaMatch = db.MatchTop1(new[] { 1f, 0f, 0f });
            Assert.AreEqual("SMA-M", smaMatch.ClassName);

            var smaFMatch = db.MatchTop1(new[] { 0f, 1f, 0f });
            Assert.AreEqual("SMA-F", smaFMatch.ClassName);
        }

        [Test]
        public void Write_RejectsMismatchedDim()
        {
            var path = Path.Combine(Path.GetTempPath(), System.Guid.NewGuid().ToString("N") + ".bin");
            var entries = new List<EnrolledClass>
            {
                new EnrolledClass(0, "A", new[] { new[] { 1f, 2f, 3f } }),
                new EnrolledClass(1, "B", new[] { new[] { 1f, 2f } }),  // wrong dim
            };
            Assert.Throws<System.ArgumentException>(
                () => ReferenceDatabaseWriter.Write(path, entries, embeddingDim: 3));
        }

        [Test]
        public void Write_RejectsTooLongClassName()
        {
            var path = Path.Combine(Path.GetTempPath(), System.Guid.NewGuid().ToString("N") + ".bin");
            var entries = new List<EnrolledClass>
            {
                new EnrolledClass(0, new string('x', 100), new[] { new[] { 1f } }),
            };
            Assert.Throws<System.ArgumentException>(
                () => ReferenceDatabaseWriter.Write(path, entries, embeddingDim: 1));
        }
    }
}
```

- [ ] **Step 2: Run tests; confirm failure**

- [ ] **Step 3: Implement `Assets/Scripts/Reference/ReferenceDatabaseWriter.cs`**

```csharp
using System;
using System.Collections.Generic;
using System.IO;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Logical record for one enrolled class: id, display name, and one or more
    /// prototype vectors clustered from the enrollment capture.
    /// </summary>
    public sealed class EnrolledClass
    {
        public int Id { get; }
        public string Name { get; }
        public float[][] Prototypes { get; }

        public EnrolledClass(int id, string name, float[][] prototypes)
        {
            Id = id;
            Name = name;
            Prototypes = prototypes;
        }
    }

    public static class ReferenceDatabaseWriter
    {
        public static void Write(string path, IReadOnlyList<EnrolledClass> classes, int embeddingDim)
        {
            foreach (var c in classes)
            {
                var nameBytes = System.Text.Encoding.UTF8.GetBytes(c.Name);
                if (nameBytes.Length > 64)
                {
                    throw new ArgumentException(
                        $"Class name too long ({nameBytes.Length} bytes > 64): {c.Name}");
                }
                foreach (var p in c.Prototypes)
                {
                    if (p.Length != embeddingDim)
                    {
                        throw new ArgumentException(
                            $"Prototype length {p.Length} for class {c.Name} != embeddingDim {embeddingDim}");
                    }
                }
            }

            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            using var f = File.Create(path);
            using var w = new BinaryWriter(f);
            w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
            w.Write((uint)2);                          // version
            w.Write((uint)classes.Count);
            w.Write((uint)embeddingDim);

            foreach (var c in classes)
            {
                w.Write(c.Id);
                var nameBuf = new byte[64];
                var src = System.Text.Encoding.UTF8.GetBytes(c.Name);
                Array.Copy(src, nameBuf, src.Length);
                w.Write(nameBuf);
                w.Write((uint)c.Prototypes.Length);
                foreach (var p in c.Prototypes)
                    foreach (var v in p) w.Write(v);
            }
        }
    }
}
```

- [ ] **Step 4: Run tests; confirm pass**

- [ ] **Step 5: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Reference/ReferenceDatabaseWriter.cs unity/RFConnectorAR/Assets/Scripts/Reference/ReferenceDatabaseWriter.cs.meta unity/RFConnectorAR/Assets/Tests/EditMode/ReferenceDatabaseWriterTests.cs unity/RFConnectorAR/Assets/Tests/EditMode/ReferenceDatabaseWriterTests.cs.meta
git commit -m "feat(unity): ReferenceDatabaseWriter for RFCE v2"
```

---

## Task 3: MiniBatchKMeans clustering

Clusters a buffer of embeddings (typically ~150 from one enrollment) into K prototypes (typically 3). On-device, in pure C#, fast. Mini-batch variant rather than classical k-means because it converges faster and is friendlier to the variable-size buffers we'll see.

**Files:**
- Create: `Assets/Scripts/Reference/MiniBatchKMeans.cs`
- Create: `Assets/Tests/EditMode/MiniBatchKMeansTests.cs`

- [ ] **Step 1: Write failing test `Assets/Tests/EditMode/MiniBatchKMeansTests.cs`**

```csharp
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
            // Three well-separated clusters in 2D. Each prototype should sit
            // close to the centroid of one of them.
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

            // Each true centre should have a prototype within 0.5 units.
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
```

- [ ] **Step 2: Run tests; confirm failure**

- [ ] **Step 3: Implement `Assets/Scripts/Reference/MiniBatchKMeans.cs`**

```csharp
using System;
using System.Collections.Generic;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Mini-batch k-means for clustering enrollment embeddings into per-class
    /// prototypes. Operates on raw float[] vectors; uses Euclidean distance
    /// after L2 normalization (so it's effectively cosine clustering on the
    /// unit hypersphere — appropriate for L2-normalized embeddings from the
    /// RGBDEmbedder).
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

            // Initialize centroids with k random distinct samples.
            var rng = new Random(seed);
            var indices = new List<int>(n);
            for (int i = 0; i < n; i++) indices.Add(i);
            // Fisher-Yates partial shuffle for first actualK
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

            // Per-centroid sample counts for the running-average update.
            var counts = new int[actualK];

            for (int iter = 0; iter < maxIters; iter++)
            {
                // Pick a mini-batch.
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
```

- [ ] **Step 4: Run tests; confirm pass**

- [ ] **Step 5: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Reference/MiniBatchKMeans.cs unity/RFConnectorAR/Assets/Scripts/Reference/MiniBatchKMeans.cs.meta unity/RFConnectorAR/Assets/Tests/EditMode/MiniBatchKMeansTests.cs unity/RFConnectorAR/Assets/Tests/EditMode/MiniBatchKMeansTests.cs.meta
git commit -m "feat(unity): mini-batch k-means clustering for enrollment prototypes"
```

---

## Task 4: OnDeviceReferenceStore

Reads/writes the on-device reference DB, exposes mutation operations (enroll a class, delete a class, list enrolled classes). Persistent store lives at `Application.persistentDataPath/references.bin`.

**Files:**
- Create: `Assets/Scripts/Reference/OnDeviceReferenceStore.cs`
- Create: `Assets/Tests/EditMode/OnDeviceReferenceStoreTests.cs`

- [ ] **Step 1: Write failing test `Assets/Tests/EditMode/OnDeviceReferenceStoreTests.cs`**

```csharp
using System.IO;
using NUnit.Framework;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Tests.EditMode
{
    public class OnDeviceReferenceStoreTests
    {
        private string _tmpDir;

        [SetUp]
        public void Setup()
        {
            _tmpDir = Path.Combine(Path.GetTempPath(), System.Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tmpDir);
        }

        [TearDown]
        public void Teardown()
        {
            if (Directory.Exists(_tmpDir)) Directory.Delete(_tmpDir, recursive: true);
        }

        [Test]
        public void Empty_StoreHasZeroClasses()
        {
            var store = new OnDeviceReferenceStore(Path.Combine(_tmpDir, "refs.bin"), embeddingDim: 4);
            Assert.AreEqual(0, store.Count);
        }

        [Test]
        public void Enroll_AddsClassAndPersistsAcrossInstances()
        {
            var path = Path.Combine(_tmpDir, "refs.bin");
            var s1 = new OnDeviceReferenceStore(path, embeddingDim: 4);
            s1.Enroll("SMA-M", new[] { new[] { 1f, 0f, 0f, 0f }, new[] { 0.9f, 0.1f, 0f, 0f } });
            Assert.AreEqual(1, s1.Count);

            var s2 = new OnDeviceReferenceStore(path, embeddingDim: 4);
            Assert.AreEqual(1, s2.Count);

            var match = s2.Database.MatchTop1(new[] { 1f, 0f, 0f, 0f });
            Assert.AreEqual("SMA-M", match.ClassName);
        }

        [Test]
        public void Enroll_TwiceForSameClassReplacesPrototypes()
        {
            var path = Path.Combine(_tmpDir, "refs.bin");
            var store = new OnDeviceReferenceStore(path, embeddingDim: 4);
            store.Enroll("SMA-M", new[] { new[] { 1f, 0f, 0f, 0f } });
            store.Enroll("SMA-M", new[] { new[] { 0f, 1f, 0f, 0f } });

            Assert.AreEqual(1, store.Count);
            var match = store.Database.MatchTop1(new[] { 0f, 1f, 0f, 0f });
            Assert.AreEqual("SMA-M", match.ClassName);
        }

        [Test]
        public void Delete_RemovesClass()
        {
            var path = Path.Combine(_tmpDir, "refs.bin");
            var store = new OnDeviceReferenceStore(path, embeddingDim: 4);
            store.Enroll("SMA-M", new[] { new[] { 1f, 0f, 0f, 0f } });
            store.Enroll("SMA-F", new[] { new[] { 0f, 1f, 0f, 0f } });
            Assert.AreEqual(2, store.Count);

            store.Delete("SMA-M");
            Assert.AreEqual(1, store.Count);
            CollectionAssert.AreEqual(new[] { "SMA-F" }, store.ClassNames);
        }
    }
}
```

- [ ] **Step 2: Run tests; confirm failure**

- [ ] **Step 3: Implement `Assets/Scripts/Reference/OnDeviceReferenceStore.cs`**

```csharp
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Mutable on-device reference store. Persists to a single RFCE v2 file.
    /// Thread model: assume single-threaded access from the Unity main thread.
    /// </summary>
    public sealed class OnDeviceReferenceStore
    {
        private readonly string _path;
        private readonly int _embeddingDim;
        private readonly List<EnrolledClass> _classes = new();
        private int _nextId = 0;

        public OnDeviceReferenceStore(string path, int embeddingDim)
        {
            _path = path;
            _embeddingDim = embeddingDim;
            if (File.Exists(path))
            {
                LoadFromDisk();
            }
        }

        public int Count => _classes.Count;
        public IReadOnlyList<string> ClassNames => _classes.Select(c => c.Name).ToList();

        public ReferenceDatabase Database
        {
            get
            {
                // Re-read from disk so we're always handing out a fresh snapshot.
                if (!File.Exists(_path))
                {
                    // Build a transient empty DB by writing & loading.
                    ReferenceDatabaseWriter.Write(_path, _classes, _embeddingDim);
                }
                return ReferenceDatabase.Load(_path);
            }
        }

        public void Enroll(string className, float[][] prototypes)
        {
            // Replace if exists, otherwise append.
            int existing = _classes.FindIndex(c => c.Name == className);
            int id = existing >= 0 ? _classes[existing].Id : _nextId++;
            var entry = new EnrolledClass(id, className, prototypes);
            if (existing >= 0) _classes[existing] = entry;
            else _classes.Add(entry);
            Persist();
        }

        public void Delete(string className)
        {
            _classes.RemoveAll(c => c.Name == className);
            Persist();
        }

        private void LoadFromDisk()
        {
            var db = ReferenceDatabase.Load(_path);
            // Re-derive _classes by reading raw file again to recover prototype vectors.
            // (ReferenceDatabase keeps them internal; cheap to reread for the small
            // sizes we expect — ~10 classes × ~3 vectors × 128 floats = ~16 KB.)
            using var f = File.OpenRead(_path);
            using var r = new BinaryReader(f);
            r.ReadBytes(4);                  // magic
            uint version = r.ReadUInt32();
            uint count = r.ReadUInt32();
            int dim = (int)r.ReadUInt32();
            for (int i = 0; i < count; i++)
            {
                int id = r.ReadInt32();
                if (id >= _nextId) _nextId = id + 1;
                var nameBytes = r.ReadBytes(64);
                int nameLen = System.Array.IndexOf<byte>(nameBytes, 0);
                if (nameLen < 0) nameLen = 64;
                string name = System.Text.Encoding.UTF8.GetString(nameBytes, 0, nameLen);
                int nProto = (version == 2) ? (int)r.ReadUInt32() : 1;
                var protos = new float[nProto][];
                for (int p = 0; p < nProto; p++)
                {
                    var v = new float[dim];
                    for (int j = 0; j < dim; j++) v[j] = r.ReadSingle();
                    protos[p] = v;
                }
                _classes.Add(new EnrolledClass(id, name, protos));
            }
        }

        private void Persist()
        {
            ReferenceDatabaseWriter.Write(_path, _classes, _embeddingDim);
        }
    }
}
```

- [ ] **Step 4: Run tests; confirm pass**

- [ ] **Step 5: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Reference/OnDeviceReferenceStore.cs unity/RFConnectorAR/Assets/Scripts/Reference/OnDeviceReferenceStore.cs.meta unity/RFConnectorAR/Assets/Tests/EditMode/OnDeviceReferenceStoreTests.cs unity/RFConnectorAR/Assets/Tests/EditMode/OnDeviceReferenceStoreTests.cs.meta
git commit -m "feat(unity): OnDeviceReferenceStore for persistent enrolled classes"
```

---

## Task 5: EnrollSession (capture + embed loop)

Pure C# state machine that runs an enrollment: capture frames at a target rate, embed each crop, accumulate into a buffer, finish, cluster, return prototypes. Owned by `EnrollController`; testable independently.

**Files:**
- Create: `Assets/Scripts/Enroll/EnrollSession.cs`
- Create: `Assets/Tests/EditMode/EnrollSessionTests.cs`

- [ ] **Step 1: Write failing test `Assets/Tests/EditMode/EnrollSessionTests.cs`**

```csharp
using NUnit.Framework;
using RFConnectorAR.Enroll;
using RFConnectorAR.Perception;
using UnityEngine;

namespace RFConnectorAR.Tests.EditMode
{
    public class EnrollSessionTests
    {
        // A stub embedder that returns a fixed vector per call (constructor-set).
        private sealed class CountingStubEmbedder : IEmbedder
        {
            public int Calls { get; private set; }
            private readonly float[] _vec;
            public CountingStubEmbedder(int dim = 4)
            {
                _vec = new float[dim]; _vec[0] = 1f;
            }
            public float[] Embed(Texture2D rgb, Texture2D depth)
            {
                Calls++;
                return (float[])_vec.Clone();
            }
        }

        [Test]
        public void NewSession_StartsEmpty()
        {
            var sess = new EnrollSession(targetFrames: 50, k: 3);
            Assert.AreEqual(0, sess.CapturedCount);
            Assert.IsFalse(sess.IsComplete);
        }

        [Test]
        public void Push_IncrementsCount_UntilTargetReached()
        {
            var sess = new EnrollSession(targetFrames: 5, k: 2);
            for (int i = 0; i < 5; i++)
            {
                Assert.IsFalse(sess.IsComplete);
                sess.Push(new[] { (float)i, 0f, 0f, 0f });
            }
            Assert.IsTrue(sess.IsComplete);
            Assert.AreEqual(5, sess.CapturedCount);
        }

        [Test]
        public void Finalize_ProducesKPrototypes()
        {
            var sess = new EnrollSession(targetFrames: 30, k: 3);
            var rng = new System.Random(0);
            for (int i = 0; i < 30; i++)
            {
                sess.Push(new[]
                {
                    (float)rng.NextDouble(), (float)rng.NextDouble(),
                    (float)rng.NextDouble(), (float)rng.NextDouble(),
                });
            }
            var protos = sess.Finalize();
            Assert.AreEqual(3, protos.Length);
            foreach (var p in protos) Assert.AreEqual(4, p.Length);
        }

        [Test]
        public void Push_AfterCompleteIsNoop()
        {
            var sess = new EnrollSession(targetFrames: 2, k: 1);
            sess.Push(new[] { 1f, 0f });
            sess.Push(new[] { 0f, 1f });
            sess.Push(new[] { 0.5f, 0.5f });  // ignored
            Assert.AreEqual(2, sess.CapturedCount);
            Assert.IsTrue(sess.IsComplete);
        }
    }
}
```

- [ ] **Step 2: Run tests; confirm failure**

- [ ] **Step 3: Implement `Assets/Scripts/Enroll/EnrollSession.cs`**

```csharp
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
```

- [ ] **Step 4: Run tests; confirm pass**

- [ ] **Step 5: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Enroll/ unity/RFConnectorAR/Assets/Scripts/Enroll.meta unity/RFConnectorAR/Assets/Tests/EditMode/EnrollSessionTests.cs unity/RFConnectorAR/Assets/Tests/EditMode/EnrollSessionTests.cs.meta
git commit -m "feat(unity): EnrollSession buffer + finalize-with-clustering"
```

---

## Task 6: ModeRouter — switch between Scanner / Enroll / Curate

Single MonoBehaviour that loads the correct scene based on user-selected mode. UI for mode selection: three buttons or a tabbed bar at the bottom of every scene.

**Files:**
- Create: `Assets/Scripts/App/ModeRouter.cs`

(No test — pure scene-loading wrapper.)

- [ ] **Step 1: Implement `Assets/Scripts/App/ModeRouter.cs`**

```csharp
using UnityEngine;
using UnityEngine.SceneManagement;

namespace RFConnectorAR.App
{
    /// <summary>
    /// Static entry-points to navigate between the three top-level scenes.
    /// Wire these to UI buttons via UnityEvents in the inspector.
    /// </summary>
    public sealed class ModeRouter : MonoBehaviour
    {
        public void GoToScanner() => SceneManager.LoadScene("Scanner");
        public void GoToEnroll()  => SceneManager.LoadScene("Enroll");
        public void GoToCurate()  => SceneManager.LoadScene("Curate");
    }
}
```

- [ ] **Step 2: Verify project compiles in Unity**

Run: `Unity -batchmode -projectPath unity/RFConnectorAR -runTests -testPlatform EditMode -testResults test-results.xml -logFile - -nographics`. Expect existing tests still pass.

- [ ] **Step 3: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/App/ModeRouter.cs unity/RFConnectorAR/Assets/Scripts/App/ModeRouter.cs.meta
git commit -m "feat(unity): ModeRouter for Scanner/Enroll/Curate scene switching"
```

---

## Task 7: EnrollController + EnrollHUD + Enroll scene (editor-heavy)

Sets up the Enroll scene: AR camera, perception pipeline (stub embedder), an EnrollSession driven from the scene's Update loop, and an enrollment-progress HUD.

**Files:**
- Create: `Assets/Scripts/Enroll/EnrollController.cs`
- Create: `Assets/Scripts/Enroll/EnrollHUD.cs`
- Create: `Assets/Scenes/Enroll.unity`

- [ ] **Step 1: Implement `Assets/Scripts/Enroll/EnrollHUD.cs`**

```csharp
using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.Enroll
{
    public sealed class EnrollHUD : MonoBehaviour
    {
        [SerializeField] private InputField _classNameInput;
        [SerializeField] private Button _startButton;
        [SerializeField] private Text _progressText;
        [SerializeField] private Slider _progressBar;
        [SerializeField] private Text _hint;

        public string ClassName => _classNameInput != null ? _classNameInput.text : "";

        public void OnStartClicked(System.Action<string> onStart)
        {
            if (_startButton != null)
            {
                _startButton.onClick.RemoveAllListeners();
                _startButton.onClick.AddListener(() => onStart?.Invoke(ClassName));
            }
        }

        public void SetIdle() => SetStatus("Type a class name and press Start.", 0f);
        public void SetCapturing(int captured, int total)
            => SetStatus($"Capturing… {captured} / {total}", (float)captured / total);
        public void SetComplete(int captured, int prototypes)
            => SetStatus($"Done. {captured} frames → {prototypes} prototypes saved.", 1f);
        public void SetError(string msg) => SetStatus(msg, 0f);

        private void SetStatus(string text, float progress)
        {
            if (_progressText != null) _progressText.text = text;
            if (_progressBar != null) _progressBar.value = progress;
        }
    }
}
```

- [ ] **Step 2: Implement `Assets/Scripts/Enroll/EnrollController.cs`**

```csharp
using RFConnectorAR.AR;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;
using UnityEngine;

namespace RFConnectorAR.Enroll
{
    public sealed class EnrollController : MonoBehaviour
    {
        [SerializeField] private CameraFrameSource _cameraFrameSource;
        [SerializeField] private EnrollHUD _hud;

        [SerializeField] private int _targetFrames = 150;
        [SerializeField] private int _prototypesPerClass = 3;
        [SerializeField] private int _embeddingDim = 128;

        private IDetector _detector;
        private IEmbedder _embedder;
        private IMeasurer _measurer;
        private OnDeviceReferenceStore _store;
        private EnrollSession _session;
        private string _activeClassName;

        private void Awake()
        {
            // Plan 2 stubs; Plan 3 swaps in real Sentis-backed implementations.
            _detector = new StubDetector(score: 0.9f);
            _embedder = new StubEmbedder(dim: _embeddingDim);
            _measurer = new StubMeasurer(diameterMm: null);

            string path = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
            _store = new OnDeviceReferenceStore(path, _embeddingDim);
        }

        private void Start()
        {
            _hud?.SetIdle();
            _hud?.OnStartClicked(StartEnrollment);
        }

        private void StartEnrollment(string className)
        {
            if (string.IsNullOrWhiteSpace(className))
            {
                _hud?.SetError("Pick a class name first.");
                return;
            }
            _activeClassName = className.Trim();
            _session = new EnrollSession(_targetFrames, _prototypesPerClass);
            _hud?.SetCapturing(0, _targetFrames);
        }

        private void Update()
        {
            if (_session == null || _session.IsComplete) return;
            if (_cameraFrameSource == null || !_cameraFrameSource.HasFrame) return;

            var rgb = _cameraFrameSource.LatestRgb;
            var detections = _detector.Detect(rgb);
            if (detections.Length == 0) return;

            // Use the top-scoring detection.
            var box = detections[0];
            var crop = CropTexture(rgb, box.NormalizedRect);
            var emb = _embedder.Embed(crop, depthCrop: null);
            _session.Push(emb);

            _hud?.SetCapturing(_session.CapturedCount, _session.TargetFrames);

            if (_session.IsComplete)
            {
                var protos = _session.Finalize();
                _store.Enroll(_activeClassName, protos);
                _hud?.SetComplete(_session.CapturedCount, protos.Length);
                _session = null;
            }
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
```

- [ ] **Step 3: Create the Enroll scene in the Unity Editor**

In Unity: **File → New Scene → Basic (Built-in)**. Save as `Assets/Scenes/Enroll.unity`.

Mirror the Scanner scene setup:

- Delete default Main Camera and Directional Light.
- Right-click hierarchy → **XR → AR Session**.
- Right-click hierarchy → **XR → XR Origin (AR)**.
- Select `XR Origin → Camera Offset → Main Camera` → **Add Component → CameraFrameSource**.
- Right-click → **UI → Canvas**, named `UI`.
- Inside `UI`, add:
  - **UI → Legacy → InputField** named `ClassNameInput` (top of screen)
  - **UI → Legacy → Button** named `StartButton` (top right)
  - **UI → Legacy → Text** named `ProgressText` (centered)
  - **UI → Legacy → Slider** named `ProgressBar` (just below the progress text)
- Create empty `App` GameObject. Add `EnrollHUD` component to the `UI` Canvas. Drag the four UI children into the EnrollHUD inspector slots.
- Add `EnrollController` component to `App`. Drag `Main Camera`'s `CameraFrameSource` and the `UI`'s `EnrollHUD` into its slots.

Add `Enroll.unity` to **File → Build Settings → Scenes In Build** at index 1 (after Scanner).

**Verify:** Press Play. The Enroll scene loads. Type a class name, press Start; the progress bar animates as the stub detector fires every frame, and a Done message appears after 150 frames (~5 seconds). Then `Application.persistentDataPath/references.bin` exists.

- [ ] **Step 4: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Enroll/EnrollHUD.cs unity/RFConnectorAR/Assets/Scripts/Enroll/EnrollHUD.cs.meta unity/RFConnectorAR/Assets/Scripts/Enroll/EnrollController.cs unity/RFConnectorAR/Assets/Scripts/Enroll/EnrollController.cs.meta unity/RFConnectorAR/Assets/Scenes/Enroll.unity unity/RFConnectorAR/Assets/Scenes/Enroll.unity.meta
git commit -m "feat(unity): Enroll scene with capture + embed + cluster + persist"
```

---

## Task 8: CurateController + CurateHUD + Curate scene (editor-heavy)

Lists enrolled classes with delete buttons. Lets the tech remove a bad enrollment so they can re-enroll cleanly.

**Files:**
- Create: `Assets/Scripts/Curate/CurateController.cs`
- Create: `Assets/Scripts/Curate/CurateHUD.cs`
- Create: `Assets/Scenes/Curate.unity`

- [ ] **Step 1: Implement `Assets/Scripts/Curate/CurateHUD.cs`**

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.Curate
{
    public sealed class CurateHUD : MonoBehaviour
    {
        [SerializeField] private RectTransform _listRoot;
        [SerializeField] private GameObject _rowPrefab;        // a row with class name + Delete button
        [SerializeField] private Text _emptyMessage;

        public Action<string> OnDeleteClicked;

        public void Render(IReadOnlyList<string> classNames)
        {
            // Clear existing children.
            for (int i = _listRoot.childCount - 1; i >= 0; i--)
            {
                Destroy(_listRoot.GetChild(i).gameObject);
            }

            if (classNames.Count == 0)
            {
                if (_emptyMessage != null) _emptyMessage.gameObject.SetActive(true);
                return;
            }
            if (_emptyMessage != null) _emptyMessage.gameObject.SetActive(false);

            foreach (var name in classNames)
            {
                var row = Instantiate(_rowPrefab, _listRoot);
                var label = row.GetComponentInChildren<Text>();
                if (label != null) label.text = name;
                var deleteBtn = row.GetComponentInChildren<Button>();
                if (deleteBtn != null)
                {
                    string captured = name;
                    deleteBtn.onClick.AddListener(() => OnDeleteClicked?.Invoke(captured));
                }
            }
        }
    }
}
```

- [ ] **Step 2: Implement `Assets/Scripts/Curate/CurateController.cs`**

```csharp
using RFConnectorAR.Reference;
using UnityEngine;

namespace RFConnectorAR.Curate
{
    public sealed class CurateController : MonoBehaviour
    {
        [SerializeField] private CurateHUD _hud;
        [SerializeField] private int _embeddingDim = 128;

        private OnDeviceReferenceStore _store;

        private void Awake()
        {
            string path = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
            _store = new OnDeviceReferenceStore(path, _embeddingDim);
        }

        private void Start()
        {
            if (_hud != null)
            {
                _hud.OnDeleteClicked = HandleDelete;
                _hud.Render(_store.ClassNames);
            }
        }

        private void HandleDelete(string className)
        {
            _store.Delete(className);
            _hud?.Render(_store.ClassNames);
        }
    }
}
```

- [ ] **Step 3: Create the Curate scene + row prefab in the Unity Editor**

In Unity: **File → New Scene → Basic (Built-in)**. Save as `Assets/Scenes/Curate.unity`.

- Delete default Main Camera and Directional Light.
- Add a **UI → Canvas**, render mode Screen Space - Overlay.
- Inside the canvas: **UI → Legacy → Scroll View** named `ListScroll`. Inside its Content child, add a **VerticalLayoutGroup** component.
- Create a row prefab:
  - In the scene, **UI → Panel** named `RowTemplate`. Inside, **UI → Legacy → Text** named `Label` and **UI → Legacy → Button** named `DeleteButton` with text "Delete".
  - Drag `RowTemplate` to `Assets/Prefabs/EnrolledClassRow.prefab`. Delete the in-scene copy.
- Add a **UI → Legacy → Text** named `EmptyMessage` with text "No enrolled classes yet — go to Enroll to teach the app."
- Create empty `App` GameObject. Add `CurateHUD` to the Canvas. Drag the Scroll View's Content into `_listRoot`, the prefab into `_rowPrefab`, and the EmptyMessage Text into `_emptyMessage`.
- Add `CurateController` to `App`. Drag the Canvas's `CurateHUD` into its slot.

Add to Build Settings at index 2.

**Verify:** Press Play. If `Application.persistentDataPath/references.bin` has any classes (from Task 7), they appear in the list with Delete buttons. Clicking Delete removes them.

- [ ] **Step 4: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/Curate/ unity/RFConnectorAR/Assets/Scripts/Curate.meta unity/RFConnectorAR/Assets/Scenes/Curate.unity unity/RFConnectorAR/Assets/Scenes/Curate.unity.meta unity/RFConnectorAR/Assets/Prefabs/EnrolledClassRow.prefab unity/RFConnectorAR/Assets/Prefabs/EnrolledClassRow.prefab.meta
git commit -m "feat(unity): Curate scene with enrolled-class list + delete"
```

---

## Task 9: Wire ModeRouter buttons into all three scenes (editor-heavy)

Add a small bottom-bar with three buttons (Scan / Enroll / Curate) to each of the three scenes so the tech can switch modes from anywhere.

**Files:**
- Modify: `Assets/Scenes/Scanner.unity`
- Modify: `Assets/Scenes/Enroll.unity`
- Modify: `Assets/Scenes/Curate.unity`

- [ ] **Step 1: Add the bottom-bar to each scene**

In each scene's Canvas:

- Right-click → **UI → Panel** named `ModeBar`. Anchor bottom, height 60.
- Add a `HorizontalLayoutGroup` component.
- Add three child **UI → Legacy → Button** GameObjects: `ScannerBtn`, `EnrollBtn`, `CurateBtn`. Set their text accordingly.
- Add a `ModeRouter` script to the Canvas (or scene's `App` GameObject).
- Wire each button's OnClick to the appropriate `ModeRouter.GoTo*` method (set via inspector).

**Verify:** From any scene, pressing each button switches to the matching scene without errors.

- [ ] **Step 2: Commit**

```
git add unity/RFConnectorAR/Assets/Scenes/Scanner.unity unity/RFConnectorAR/Assets/Scenes/Enroll.unity unity/RFConnectorAR/Assets/Scenes/Curate.unity
git commit -m "feat(unity): mode-bar for Scanner/Enroll/Curate navigation"
```

---

## Task 10: Update Scanner to use OnDeviceReferenceStore as its matcher

Currently AppBootstrap (Plan 2 Task 12) constructs a `StubMatcher`. Swap to a matcher backed by `OnDeviceReferenceStore.Database`. This means the Scanner now identifies whatever the tech enrolled.

**Files:**
- Modify: `Assets/Scripts/App/AppBootstrap.cs`

- [ ] **Step 1: Modify `AppBootstrap.Awake` to use the on-device store**

Replace the matcher line in `_pipeline = new PerceptionPipeline(...)` with:

```csharp
string refPath = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
var store = new RFConnectorAR.Reference.OnDeviceReferenceStore(refPath, embeddingDim: 128);

IMatcher matcher;
try
{
    matcher = store.Database;  // ReferenceDatabase implements IMatcher
}
catch (System.IO.FileNotFoundException)
{
    matcher = new StubMatcher(classId: -1, className: "Unknown", cosine: 0.0f);
}

_pipeline = new PerceptionPipeline(
    detector: new StubDetector(score: 0.9f),
    embedder: new StubEmbedder(),
    matcher: matcher,
    measurer: new StubMeasurer(diameterMm: null),
    fuser: new ConfidenceFuser());
```

(Add `using RFConnectorAR.Reference;` at the top if not present.)

- [ ] **Step 2: Run EditMode tests; verify all still pass**

- [ ] **Step 3: Commit**

```
git add unity/RFConnectorAR/Assets/Scripts/App/AppBootstrap.cs
git commit -m "feat(unity): Scanner reads from OnDeviceReferenceStore"
```

---

## Task 11: Run full test suite + manual editor walkthrough + README update

**Files:**
- Modify: `unity/RFConnectorAR/README.md`

- [ ] **Step 1: Run full EditMode test suite**

```
"/e/unity/6000.0.73f1/Editor/Unity.exe" -batchmode -projectPath "E:\anduril\unity\RFConnectorAR" -runTests -testPlatform EditMode -testResults "E:\anduril\unity\RFConnectorAR\test-results.xml" -logFile - -nographics
```

Expected: existing 17 + new (3 v2 + 3 writer + 4 kmeans + 4 store + 4 session) = 35 tests, all passing.

- [ ] **Step 2: Manual end-to-end editor test**

In Unity Editor:

1. Press Play in the Scanner scene. Verify it shows "Unknown" for all detections (no classes enrolled yet).
2. Use the bottom-bar to switch to Enroll. Type "TestConnector" and press Start. Wait ~5s for completion.
3. Switch back to Scanner. Verify it now shows "TestConnector" labels (since stub detector + stub embedder always returns the same vector, all detections will match).
4. Switch to Curate. Verify "TestConnector" appears with a Delete button. Click Delete.
5. Switch back to Scanner. Verify it returns to "Unknown."

- [ ] **Step 3: Append to `unity/RFConnectorAR/README.md`**

```markdown
## On-device enroll architecture (Plan 2b)

The app ships with no enrolled connectors. Three modes:

- **Scanner** — identifies connectors against the on-device reference DB.
- **Enroll** — type a class name, hold the connector in front of the camera,
  press Start. ~5 seconds of capture, then the embedding cluster is saved
  to `Application.persistentDataPath/references.bin`.
- **Curate** — list of enrolled classes, with delete buttons.

Reference DB is stored as RFCE format v2 (multiple prototypes per class)
and is fully on-device. There is no server dependency.

In Plan 2 / Plan 2b, perception runs on stub models (always returns one
canned detection + a fixed embedding). Real Sentis-backed models land in
Plan 3.
```

- [ ] **Step 4: Commit**

```
git add unity/RFConnectorAR/README.md
git commit -m "docs(unity): document on-device enroll architecture in README"
```

---

## Plan Self-Review

Spec coverage:
- On-device persistent reference DB → Tasks 1, 2, 4
- Multiple prototypes per class via clustering → Tasks 1, 3, 5
- Capture + embed loop → Task 5, 7
- Enroll / Curate / Scanner mode switching → Task 6, 9
- Scanner reads on-device DB → Task 10

Placeholder scan: every step has explicit code + commands; no TBDs.

Type consistency: `EnrolledClass(int id, string name, float[][] prototypes)` is the same shape across the writer (Task 2), the store (Task 4), and the file format (Task 1). `MiniBatchKMeans.Cluster(float[][], int k, int maxIters, int batchSize, int seed)` signature consistent across `EnrollSession.Finalize` (Task 5) and tests (Task 3). `OnDeviceReferenceStore` API surface is consistent across Enroll (Task 7), Curate (Task 8), and the scanner refactor (Task 10).

Scope: 11 tasks, of which 5 are pure-C# TDD tasks (Tasks 1–5) and 6 are editor-heavy (Tasks 6–11). Each task produces a focused commit. Plan delivers a fully working enroll-and-identify loop with stub perception; Plan 3 swaps in real models.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-unity-enroll-curate.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Code-only tasks (1–5, 10) get dispatched to subagents; editor tasks (6 done by me, 7–9, 11) you do in the Unity Editor.

**2. Inline Execution** — I do code tasks directly; you do the editor tasks. Same overall split, less subagent ceremony.

Which approach?
