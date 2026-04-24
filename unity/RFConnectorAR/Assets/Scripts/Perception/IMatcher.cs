namespace RFConnectorAR.Perception
{
    public readonly struct Match
    {
        public readonly int ClassId;
        public readonly string ClassName;
        public readonly float CosineSimilarity;

        public Match(int classId, string className, float cosineSimilarity)
        {
            ClassId = classId;
            ClassName = className;
            CosineSimilarity = cosineSimilarity;
        }
    }

    public interface IMatcher
    {
        Match MatchTop1(float[] embedding);
    }
}
