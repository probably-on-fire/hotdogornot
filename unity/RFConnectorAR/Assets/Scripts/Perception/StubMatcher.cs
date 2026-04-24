namespace RFConnectorAR.Perception
{
    public sealed class StubMatcher : IMatcher
    {
        private readonly Match _match;

        public StubMatcher(int classId = 1, string className = "SMA-F", float cosine = 0.85f)
        {
            _match = new Match(classId, className, cosine);
        }

        public Match MatchTop1(float[] embedding)
        {
            _ = embedding;
            return _match;
        }
    }
}
