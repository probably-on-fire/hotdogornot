using System;
using System.IO;
using RFConnectorAR.Perception;

namespace RFConnectorAR.Reference
{
    public sealed class ReferenceDatabase : IMatcher
    {
        private readonly int[] _ids;
        private readonly string[] _names;
        private readonly float[][] _vectors;

        public int Count => _ids.Length;
        public int EmbeddingDim => _vectors.Length == 0 ? 0 : _vectors[0].Length;

        private ReferenceDatabase(int[] ids, string[] names, float[][] vectors)
        {
            _ids = ids; _names = names; _vectors = vectors;
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
            if (version != 1)
            {
                throw new InvalidDataException($"Unsupported RFCE version {version} in {path}");
            }

            uint count = r.ReadUInt32();
            uint dim = r.ReadUInt32();

            var ids = new int[count];
            var names = new string[count];
            var vectors = new float[count][];

            for (int i = 0; i < count; i++)
            {
                ids[i] = r.ReadInt32();
                var nameBytes = r.ReadBytes(64);
                int nameLen = Array.IndexOf<byte>(nameBytes, 0);
                if (nameLen < 0) nameLen = 64;
                names[i] = System.Text.Encoding.UTF8.GetString(nameBytes, 0, nameLen);

                var v = new float[dim];
                for (int j = 0; j < dim; j++) v[j] = r.ReadSingle();
                vectors[i] = v;
            }

            return new ReferenceDatabase(ids, names, vectors);
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
                float s = CosineSimilarity(embedding, _vectors[i]);
                if (s > bestScore)
                {
                    bestScore = s;
                    bestIdx = i;
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
