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
                    new[] { 0.99f, 0.1f, 0f },
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
                    new[] { 0.5f, 0.5f, 0.7f },
                }),
                (1, "SMA-F", new[] {
                    new[] { 0f, 1f, 0f },
                }));

            var db = ReferenceDatabase.Load(path);
            var match = db.MatchTop1(new[] { 0.5f, 0.45f, 0.7f });
            Assert.AreEqual(0, match.ClassId);
            Assert.AreEqual("SMA-M", match.ClassName);
        }

        [Test]
        public void Load_V1_StillWorks()
        {
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
