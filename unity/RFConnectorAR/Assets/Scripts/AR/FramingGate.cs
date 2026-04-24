using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Runs FramingDetector on the latest camera frame every `_analyzeInterval`
    /// seconds and exposes an IsFramed property. Optionally drives the color of
    /// a UI reticle so the operator can see when they're framed.
    ///
    /// Attach to the `App` GameObject. Consumers (AppBootstrap, EnrollController)
    /// check `IsFramed` before committing a verdict or pushing an embedding.
    /// </summary>
    public sealed class FramingGate : MonoBehaviour
    {
        [SerializeField] private CameraFrameSource _cameraFrameSource;
        [SerializeField] private Image _reticle;
        [SerializeField] private float _analyzeInterval = 0.1f;   // 10 Hz

        [SerializeField] private Color _framedColor = new Color(0.2f, 0.9f, 0.4f);
        [SerializeField] private Color _unframedColor = new Color(0.7f, 0.7f, 0.7f, 0.6f);

        private float _nextAnalyzeTime;
        private FramingDetector.Result _lastResult;

        public bool IsFramed => _lastResult.IsFramed;
        public float Score => _lastResult.Score;
        public string Reason => _lastResult.Reason;

        private void Update()
        {
            if (Time.time < _nextAnalyzeTime) return;
            _nextAnalyzeTime = Time.time + _analyzeInterval;

            if (_cameraFrameSource == null || !_cameraFrameSource.HasFrame)
            {
                _lastResult = new FramingDetector.Result { IsFramed = false, Reason = "no frame" };
                UpdateReticle();
                return;
            }

            _lastResult = FramingDetector.Analyze(_cameraFrameSource.LatestRgb);
            UpdateReticle();
        }

        private void UpdateReticle()
        {
            if (_reticle != null)
            {
                _reticle.color = _lastResult.IsFramed ? _framedColor : _unframedColor;
            }
        }
    }
}
