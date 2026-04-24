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
                if (!File.Exists(_path))
                {
                    ReferenceDatabaseWriter.Write(_path, _classes, _embeddingDim);
                }
                return ReferenceDatabase.Load(_path);
            }
        }

        public void Enroll(string className, float[][] prototypes)
        {
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
            using var f = File.OpenRead(_path);
            using var r = new BinaryReader(f);
            r.ReadBytes(4);
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
