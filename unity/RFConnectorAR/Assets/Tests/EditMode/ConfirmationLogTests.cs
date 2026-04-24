using System.IO;
using NUnit.Framework;
using RFConnectorAR.Learning;

namespace RFConnectorAR.Tests.EditMode
{
    public class ConfirmationLogTests
    {
        [Test]
        public void Append_WritesJsonLineToFile()
        {
            var path = Path.Combine(Path.GetTempPath(), "rfconfirm_" + System.Guid.NewGuid().ToString("N") + ".jsonl");
            var log = new ConfirmationLog(path);

            log.Append(new SignalRecord
            {
                Timestamp = "2026-04-23T12:00:00Z",
                ModelVersion = "v0",
                ReferenceDbVersion = "v0",
                PredictedClassName = "SMA-F",
                Score = 0.9f,
                UserAction = "confirmed",
            });

            var lines = File.ReadAllLines(path);
            Assert.AreEqual(1, lines.Length);
            StringAssert.Contains("\"PredictedClassName\":\"SMA-F\"", lines[0]);
            StringAssert.Contains("\"UserAction\":\"confirmed\"", lines[0]);
        }

        [Test]
        public void Append_AppendsAcrossCalls()
        {
            var path = Path.Combine(Path.GetTempPath(), "rfconfirm_" + System.Guid.NewGuid().ToString("N") + ".jsonl");
            var log = new ConfirmationLog(path);

            log.Append(new SignalRecord { Timestamp = "t1", PredictedClassName = "SMA-F", UserAction = "confirmed" });
            log.Append(new SignalRecord { Timestamp = "t2", PredictedClassName = "SMA-M", UserAction = "corrected_to_SMA-F" });

            var lines = File.ReadAllLines(path);
            Assert.AreEqual(2, lines.Length);
        }

        [Test]
        public void Count_ReadsLineCount()
        {
            var path = Path.Combine(Path.GetTempPath(), "rfconfirm_" + System.Guid.NewGuid().ToString("N") + ".jsonl");
            var log = new ConfirmationLog(path);
            log.Append(new SignalRecord { Timestamp = "t1", PredictedClassName = "X", UserAction = "confirmed" });
            log.Append(new SignalRecord { Timestamp = "t2", PredictedClassName = "Y", UserAction = "confirmed" });
            log.Append(new SignalRecord { Timestamp = "t3", PredictedClassName = "Z", UserAction = "confirmed" });

            Assert.AreEqual(3, log.Count());
        }
    }
}
