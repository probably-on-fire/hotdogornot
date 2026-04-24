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
