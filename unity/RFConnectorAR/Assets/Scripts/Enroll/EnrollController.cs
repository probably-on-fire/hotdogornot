using RFConnectorAR.AR;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;
using UnityEngine;

namespace RFConnectorAR.Enroll
{
    public sealed class EnrollController : MonoBehaviour
    {
        [SerializeField] private CameraFrameSource _cameraFrameSource;
        [SerializeField] private EnrollHUD _hud;
        [SerializeField] private FramingGate _framingGate;

        [SerializeField] private int _targetFrames = 150;
        [SerializeField] private int _prototypesPerClass = 3;
        [SerializeField] private int _embeddingDim = 128;

        private IDetector _detector;
        private IEmbedder _embedder;
        private IMeasurer _measurer;
        private OnDeviceReferenceStore _store;
        private EnrollSession _session;
        private string _activeClassName;

        private void Awake()
        {
            _detector = new StubDetector(score: 0.9f);
            _embedder = new StubEmbedder(dim: _embeddingDim);
            _measurer = new StubMeasurer(diameterMm: null);

            string path = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
            _store = new OnDeviceReferenceStore(path, _embeddingDim);
        }

        private void Start()
        {
            _hud?.SetIdle();
            _hud?.OnStartClicked(StartEnrollment);
        }

        private void StartEnrollment(string className)
        {
            if (string.IsNullOrWhiteSpace(className))
            {
                _hud?.SetError("Pick a class name first.");
                return;
            }
            _activeClassName = className.Trim();
            _session = new EnrollSession(_targetFrames, _prototypesPerClass);
            _hud?.SetCapturing(0, _targetFrames);
        }

        private void Update()
        {
            if (_session == null || _session.IsComplete) return;
            if (_cameraFrameSource == null || !_cameraFrameSource.HasFrame) return;

            // Skip enrollment frames when the framing gate is red. This
            // prevents bad frames (user's hand covering the connector, camera
            // pointing away, etc.) from polluting the embedding buffer.
            if (_framingGate != null && !_framingGate.IsFramed) return;

            var rgb = _cameraFrameSource.LatestRgb;
            var detections = _detector.Detect(rgb);
            if (detections.Length == 0) return;

            var box = detections[0];
            var crop = CropTexture(rgb, box.NormalizedRect);
            var emb = _embedder.Embed(crop, depthCrop: null);
            _session.Push(emb);

            _hud?.SetCapturing(_session.CapturedCount, _session.TargetFrames);

            if (_session.IsComplete)
            {
                var protos = _session.Finalize();
                _store.Enroll(_activeClassName, protos);
                _hud?.SetComplete(_session.CapturedCount, protos.Length);
                _session = null;
            }
        }

        private static Texture2D CropTexture(Texture2D src, Rect normalized)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(normalized.xMin * src.width), 0, src.width - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt(normalized.yMin * src.height), 0, src.height - 1);
            int w = Mathf.Clamp(Mathf.RoundToInt(normalized.width * src.width), 1, src.width - x);
            int h = Mathf.Clamp(Mathf.RoundToInt(normalized.height * src.height), 1, src.height - y);
            var pixels = src.GetPixels(x, y, w, h);
            var crop = new Texture2D(w, h, src.format, false);
            crop.SetPixels(pixels);
            crop.Apply();
            return crop;
        }
    }
}
