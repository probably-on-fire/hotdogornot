using System;
using System.Collections;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using RFConnectorAR.Perception;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Owns the live SentisClassifier instance.
    ///
    /// Boot path:
    ///   1. Load the bundled ModelAsset from Assets/Resources/Models/
    ///      (Unity's editor ONNX importer creates the asset at build time).
    ///   2. Construct a SentisClassifier from it; expose via GetClassifier().
    ///
    /// OTA path (DEFERRED):
    ///   The Inference Engine runtime only reads its own .sentis FlatBuffer
    ///   format — it cannot parse raw ONNX at runtime. To OTA-update the
    ///   bundled model we'd need the relay to serve a pre-serialized .sentis
    ///   file (produced by ModelWriter.Save in an Editor batchmode pass).
    ///   The Python training pipeline doesn't produce .sentis today, so OTA
    ///   updates are stubbed out for now: we still poll the relay version
    ///   and log when a newer one is available, but we don't try to load it.
    ///   When the .sentis pipeline is wired up, switch the download URL +
    ///   re-enable LoadFromBytes(...) in OnVersionResponse.
    /// </summary>
    public sealed class ModelUpdater : MonoBehaviour
    {
        [Header("Bundled model")]
        [SerializeField] private string resourcesModelPath = "Models/connector_classifier";
        [SerializeField] private string resourcesLabelsPath = "Models/labels";

        [Header("Relay (OTA — deferred until .sentis serving is wired up)")]
        [SerializeField] private string relayBaseUrl = "https://aired.com/rfcai";
        [SerializeField] private string deviceToken = "";
        [SerializeField] private string deviceId = "";
        [SerializeField] private float versionCheckIntervalSeconds = 600f;
        [Tooltip("If false, the OTA poll loop is disabled entirely.")]
        [SerializeField] private bool enableOtaPolling = true;

        private RelayClient _client;
        private IClassifier _classifier;
        private int _localVersion = 0;

        public IClassifier GetClassifier() => _classifier;
        public int CurrentVersion => _localVersion;

        private void Awake()
        {
            if (string.IsNullOrEmpty(deviceId))
                deviceId = SystemInfo.deviceUniqueIdentifier;
            _client = new RelayClient(relayBaseUrl, deviceToken, deviceId);
        }

        private IEnumerator Start()
        {
            yield return null;
            LoadBundledClassifier();
            if (enableOtaPolling)
                StartCoroutine(VersionPollLoop());
        }

        private void LoadBundledClassifier()
        {
            var modelAsset = Resources.Load<ModelAsset>(resourcesModelPath);
            if (modelAsset == null)
            {
                Debug.LogError($"[ModelUpdater] no ModelAsset at Resources/{resourcesModelPath}. " +
                               "Did you place the .onnx under Assets/Resources/Models/?");
                return;
            }

            string[] classNames = LoadBundledLabels();
            if (classNames == null || classNames.Length == 0)
            {
                Debug.LogError($"[ModelUpdater] no labels at Resources/{resourcesLabelsPath} — classifier disabled.");
                return;
            }

            try
            {
                if (_classifier is IDisposable old) old.Dispose();
                _classifier = SentisClassifier.LoadFromModelAsset(modelAsset, classNames);
                _localVersion = 0;   // bundled has no published version
                Debug.Log($"[ModelUpdater] loaded bundled classifier (classes: {classNames.Length})");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUpdater] failed to load classifier: {e.Message}");
            }
        }

        private string[] LoadBundledLabels()
        {
            var ta = Resources.Load<TextAsset>(resourcesLabelsPath);
            if (ta == null) return null;
            try
            {
                var blob = JsonUtility.FromJson<RelayClient.LabelsBlob>(ta.text);
                return blob?.class_names;
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUpdater] labels JSON parse failed: {e.Message}");
                return null;
            }
        }

        private IEnumerator VersionPollLoop()
        {
            while (true)
            {
                yield return _client.GetVersion(OnVersionResponse);
                yield return new WaitForSeconds(versionCheckIntervalSeconds);
            }
        }

        private void OnVersionResponse(int remoteVersion, string err)
        {
            if (err != null)
            {
                Debug.LogWarning($"[ModelUpdater] version check failed: {err}");
                return;
            }
            if (remoteVersion <= _localVersion) return;

            // OTA download path is deferred — see class docstring. For now,
            // log the available version so we know the relay is reachable
            // and the model has moved forward.
            Debug.Log($"[ModelUpdater] relay reports newer model v{remoteVersion} " +
                      $"(local v{_localVersion}). OTA load skipped (needs .sentis serving). " +
                      "Rebuild the APK with the new bundled ONNX to pick it up.");
        }

        private void OnDestroy()
        {
            if (_classifier is IDisposable old) old.Dispose();
        }
    }
}
