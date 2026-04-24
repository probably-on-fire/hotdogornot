using System;
using System.IO;
using NUnit.Framework;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;

namespace RFConnectorAR.Tests.EditMode
{
    public class ReferenceDatabaseTests
    {
        private static string WriteSyntheticBinary(params (int id, string name, float[] vec)[] entries)
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            using var f = File.Create(path);
            using var w = new BinaryWriter(f);

            int dim = entries[0].vec.Length;
            w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
            w.Write((uint)1);
            w.Write((uint)entries.Length);
            w.Write((uint)dim);

            foreach (var e in entries)
            {
                w.Write(e.id);
                var nameBytes = new byte[64];
                var src = System.Text.Encoding.UTF8.GetBytes(e.name);
                Array.Copy(src, nameBytes, Math.Min(src.Length, 64));
                w.Write(nameBytes);
                foreach (var v in e.vec) w.Write(v);
            }

            return path;
        }

        [Test]
        public void Load_ReadsMagicAndClassMetadata()
        {
            var path = WriteSyntheticBinary(
                (0, "SMA-M", new[] { 1f, 0f, 0f }),
                (1, "SMA-F", new[] { 0f, 1f, 0f }));

            var db = ReferenceDatabase.Load(path);

            Assert.AreEqual(2, db.Count);
            Assert.AreEqual(3, db.EmbeddingDim);
        }

        [Test]
        public void MatchTop1_ReturnsExactMatchForIdenticalVector()
        {
            var path = WriteSyntheticBinary(
                (0, "SMA-M", new[] { 1f, 0f, 0f }),
                (1, "SMA-F", new[] { 0f, 1f, 0f }));
            var db = ReferenceDatabase.Load(path);

            var match = db.MatchTop1(new[] { 1f, 0f, 0f });

            Assert.AreEqual(0, match.ClassId);
            Assert.AreEqual("SMA-M", match.ClassName);
            Assert.That(match.CosineSimilarity, Is.EqualTo(1f).Within(1e-5f));
        }

        [Test]
        public void MatchTop1_PicksHigherCosineSimilarity()
        {
            var path = WriteSyntheticBinary(
                (0, "SMA-M", new[] { 1f, 0f, 0f }),
                (1, "SMA-F", new[] { 0.707f, 0.707f, 0f }));
            var db = ReferenceDatabase.Load(path);

            var match = db.MatchTop1(new[] { 0.9f, 0.4359f, 0f });
            Assert.AreEqual(1, match.ClassId);
            Assert.AreEqual("SMA-F", match.ClassName);
        }

        [Test]
        public void Load_ThrowsOnBadMagic()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 0, 0, 0, 0 });
            Assert.Throws<InvalidDataException>(() => ReferenceDatabase.Load(path));
        }

        [Test]
        public void Load_ThrowsOnUnsupportedVersion()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            using (var f = File.Create(path))
            using (var w = new BinaryWriter(f))
            {
                w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
                w.Write((uint)999);
                w.Write((uint)0);
                w.Write((uint)0);
            }
            Assert.Throws<InvalidDataException>(() => ReferenceDatabase.Load(path));
        }
    }
}
