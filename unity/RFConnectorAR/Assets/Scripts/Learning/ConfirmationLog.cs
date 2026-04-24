using System;
using System.IO;
using UnityEngine;

namespace RFConnectorAR.Learning
{
    [Serializable]
    public class SignalRecord
    {
        public string Timestamp;
        public string DeviceId;
        public string AppVersion;
        public string ModelVersion;
        public string ReferenceDbVersion;
        public string PredictedClassName;
        public int PredictedClassId;
        public float Score;
        public float? MeasuredDiameterMm;
        public string FusionConfidence;
        public string UserAction;
    }

    public sealed class ConfirmationLog
    {
        private readonly string _path;

        public ConfirmationLog(string path)
        {
            _path = path;
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }

        public static ConfirmationLog AtPersistentDataPath(string filename = "confirmations.jsonl")
        {
            return new ConfirmationLog(Path.Combine(Application.persistentDataPath, filename));
        }

        public void Append(SignalRecord record)
        {
            string line = JsonUtility.ToJson(record);
            File.AppendAllText(_path, line + "\n");
        }

        public int Count()
        {
            if (!File.Exists(_path)) return 0;
            int n = 0;
            foreach (var _ in File.ReadLines(_path)) n++;
            return n;
        }
    }
}
