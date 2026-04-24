using NUnit.Framework;
using RFConnectorAR.Enroll;
using UnityEngine;

namespace RFConnectorAR.Tests.EditMode
{
    public class EnrollSessionTests
    {
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
            sess.Push(new[] { 0.5f, 0.5f });
            Assert.AreEqual(2, sess.CapturedCount);
            Assert.IsTrue(sess.IsComplete);
        }
    }
}
