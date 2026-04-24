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
