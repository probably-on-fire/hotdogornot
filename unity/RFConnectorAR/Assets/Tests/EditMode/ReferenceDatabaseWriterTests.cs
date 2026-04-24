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
                new EnrolledClass(1, "B", new[] { new[] { 1f, 2f } }),
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
