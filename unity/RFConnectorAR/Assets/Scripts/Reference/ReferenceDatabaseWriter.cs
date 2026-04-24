using System;
using System.Collections.Generic;
using System.IO;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Logical record for one enrolled class: id, display name, and one or more
    /// prototype vectors clustered from the enrollment capture.
    /// </summary>
    public sealed class EnrolledClass
    {
        public int Id { get; }
        public string Name { get; }
        public float[][] Prototypes { get; }

        public EnrolledClass(int id, string name, float[][] prototypes)
        {
            Id = id;
            Name = name;
            Prototypes = prototypes;
        }
    }

    public static class ReferenceDatabaseWriter
    {
        public static void Write(string path, IReadOnlyList<EnrolledClass> classes, int embeddingDim)
        {
            foreach (var c in classes)
            {
                var nameBytes = System.Text.Encoding.UTF8.GetBytes(c.Name);
                if (nameBytes.Length > 64)
                {
                    throw new ArgumentException(
                        $"Class name too long ({nameBytes.Length} bytes > 64): {c.Name}");
                }
                foreach (var p in c.Prototypes)
                {
                    if (p.Length != embeddingDim)
                    {
                        throw new ArgumentException(
                            $"Prototype length {p.Length} for class {c.Name} != embeddingDim {embeddingDim}");
                    }
                }
            }

            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            using var f = File.Create(path);
            using var w = new BinaryWriter(f);
            w.Write(new byte[] { (byte)'R', (byte)'F', (byte)'C', (byte)'E' });
            w.Write((uint)2);
            w.Write((uint)classes.Count);
            w.Write((uint)embeddingDim);

            foreach (var c in classes)
            {
                w.Write(c.Id);
                var nameBuf = new byte[64];
                var src = System.Text.Encoding.UTF8.GetBytes(c.Name);
                Array.Copy(src, nameBuf, src.Length);
                w.Write(nameBuf);
                w.Write((uint)c.Prototypes.Length);
                foreach (var p in c.Prototypes)
                    foreach (var v in p) w.Write(v);
            }
        }
    }
}
