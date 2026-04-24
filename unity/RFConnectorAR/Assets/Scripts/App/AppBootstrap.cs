using RFConnectorAR.AR;
using RFConnectorAR.Learning;
using RFConnectorAR.Perception;
using RFConnectorAR.Reference;
using RFConnectorAR.UI;
using UnityEngine;

namespace RFConnectorAR.App
{
    public sealed class AppBootstrap : MonoBehaviour
    {
        [SerializeField] private CameraFrameSource _cameraFrameSource;
        [SerializeField] private OverlayController _overlay;
        [SerializeField] private ScannerHUD _hud;

        private PerceptionPipeline _pipeline;
        private ConfirmationLog _log;
        private string _modelVersion = "stub-v0";

        private void Awake()
        {
            string refPath = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
            var store = new RFConnectorAR.Reference.OnDeviceReferenceStore(refPath, embeddingDim: 128);

            IMatcher matcher;
            if (store.Count == 0)
            {
                matcher = new StubMatcher(classId: -1, className: "Unknown", cosine: 0.0f);
            }
            else
            {
                matcher = store.Database;
            }

            _pipeline = new PerceptionPipeline(
                detector: new StubDetector(score: 0.9f),
                embedder: new StubEmbedder(),
                matcher: matcher,
                measurer: new StubMeasurer(diameterMm: null),
                fuser: new ConfidenceFuser());

            _log = ConfirmationLog.AtPersistentDataPath();
        }

        private void Update()
        {
            if (_hud != null) _hud.SetStatus(_modelVersion, _log.Count());

            if (_cameraFrameSource == null || !_cameraFrameSource.HasFrame)
            {
                _hud?.SetState(ScannerHUD.ScanState.Searching);
                return;
            }

            var verdicts = _pipeline.RunFrame(_cameraFrameSource.LatestRgb, depthFrame: null);

            if (verdicts.Length == 0)
            {
                _hud?.SetState(ScannerHUD.ScanState.Searching);
                _overlay?.UpdateVerdicts(System.Array.Empty<Verdict>());
                return;
            }

            var cam = Camera.main;
            var camPos = cam != null ? cam.transform.position : Vector3.zero;
            var camFwd = cam != null ? cam.transform.forward : Vector3.forward;

            var positioned = new Verdict[verdicts.Length];
            for (int i = 0; i < verdicts.Length; i++)
            {
                var v = verdicts[i];
                positioned[i] = new Verdict
                {
                    ClassName = v.ClassName,
                    ClassId = v.ClassId,
                    Confidence = v.Confidence,
                    Score = v.Score,
                    MeasuredDiameterMm = v.MeasuredDiameterMm,
                    DetectionBox = v.DetectionBox,
                    WorldPosition = camPos + camFwd * 0.3f,
                };
            }

            _overlay?.UpdateVerdicts(positioned);

            var v0 = positioned[0];
            _hud?.SetState(
                ScannerHUD.ScanState.Identified,
                v0.MeasuredDiameterMm is float mm
                    ? $"{v0.ClassName}  ({mm:F2} mm)  [{v0.Confidence}]"
                    : $"{v0.ClassName}  [{v0.Confidence}]");
        }
    }
}
