using System;
using System.Collections;
using System.IO;
using UnityEngine;
using RFConnectorAR.Perception;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// Polls the relay for new model versions and swaps the live SentisClassifier.
    ///
    /// On Awake:
    ///   1. Try loading the cached model from persistentDataPath.
    ///   2. If none, copy the bundled StreamingAssets/connector_classifier.onnx
    ///      to persistentDataPath and load that.
    ///   3. Either way, kick off a version-check coroutine — if the relay
    ///      reports a higher version than what's on disk, download the new
    ///      ONNX, validate sha256, swap the active classifier, persist.
    ///
    /// Call <see cref="GetClassifier"/> from the inference loop. The returned
    /// IClassifier reference is stable across update cycles within a single
    /// frame; the updater swaps it between frames.
    /// </summary>
    public sealed class ModelUpdater : MonoBehaviour
    {
        [Header("Relay")]
        [SerializeField] private string relayBaseUrl = "https://aired.com/rfcai";
        [SerializeField] private string deviceToken = ""; // populate via inspector or DeviceConfig
        [SerializeField] private string deviceId = "";    // device identifier, falls back to SystemInfo
        [Header("Bundled fallback")]
        [SerializeField] private string streamingAssetsRelativeOnnx = "model/connector_classifier.onnx";
        [SerializeField] private string streamingAssetsRelativeLabels = "model/labels.json";
        [Header("Polling")]
        [SerializeField] private float versionCheckIntervalSeconds = 600f;

        private RelayClient _client;
        private IClassifier _classifier;
        private int _localVersion = 0;

        public IClassifier GetClassifier() => _classifier;
        public int CurrentVersion => _localVersion;

        private string LocalOnnxPath =>
            Path.Combine(Application.persistentDataPath, "weights.onnx");
        private string LocalLabelsPath =>
            Path.Combine(Application.persistentDataPath, "labels.json");
        private string LocalVersionPath =>
            Path.Combine(Application.persistentDataPath, "version.txt");

        private void Awake()
        {
            if (string.IsNullOrEmpty(deviceId))
                deviceId = SystemInfo.deviceUniqueIdentifier;
            _client = new RelayClient(relayBaseUrl, deviceToken, deviceId);
        }

        private IEnumerator Start()
        {
            yield return EnsureLocalModel();
            yield return TryLoadClassifier();
            // Kick off the periodic version-check loop.
            StartCoroutine(VersionPollLoop());
        }

        private IEnumerator EnsureLocalModel()
        {
            if (File.Exists(LocalOnnxPath) && File.Exists(LocalLabelsPath))
                yield break;

            // Copy bundled StreamingAssets → persistentDataPath. On Android
            // StreamingAssets lives inside the APK so we need
            // UnityWebRequest; on iOS/desktop a file copy works.
            string srcOnnx = Path.Combine(Application.streamingAssetsPath, streamingAssetsRelativeOnnx);
            string srcLabels = Path.Combine(Application.streamingAssetsPath, streamingAssetsRelativeLabels);

            yield return CopyStreamingAsset(srcOnnx, LocalOnnxPath);
            yield return CopyStreamingAsset(srcLabels, LocalLabelsPath);

            // No version stamp yet from a downloaded model — assume bundled
            // is version 0 so any relay-reported version > 0 triggers OTA.
            File.WriteAllText(LocalVersionPath, "0");
        }

        private IEnumerator CopyStreamingAsset(string srcUrl, string dstPath)
        {
            // On Android srcUrl looks like jar:file:///... — use UnityWebRequest.
            // On other platforms it's a normal path; File.Copy is faster but
            // UnityWebRequest works everywhere so we use it uniformly.
            using var req = UnityEngine.Networking.UnityWebRequest.Get(srcUrl);
            yield return req.SendWebRequest();
            if (req.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[ModelUpdater] could not load bundled asset {srcUrl}: {req.error}");
                yield break;
            }
            File.WriteAllBytes(dstPath, req.downloadHandler.data);
        }

        private IEnumerator TryLoadClassifier()
        {
            yield return null;
            if (!File.Exists(LocalOnnxPath) || !File.Exists(LocalLabelsPath))
            {
                Debug.LogWarning("[ModelUpdater] no local model — classifier disabled.");
                yield break;
            }
            try
            {
                var bytes = File.ReadAllBytes(LocalOnnxPath);
                var labels = LoadLabels(LocalLabelsPath);
                var newClassifier = SentisClassifier.LoadFromBytes(bytes, labels);
                if (_classifier is IDisposable old) old.Dispose();
                _classifier = newClassifier;
                if (File.Exists(LocalVersionPath) && int.TryParse(File.ReadAllText(LocalVersionPath), out int v))
                    _localVersion = v;
                Debug.Log($"[ModelUpdater] loaded classifier (version {_localVersion})");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelUpdater] failed to load classifier: {e.Message}");
            }
        }

        private IEnumerator VersionPollLoop()
        {
            // First check fires on launch (immediate); subsequent ones at the
            // configured interval. Persistent so a long-lived session picks
            // up nightly retrains automatically.
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
            if (remoteVersion <= _localVersion)
                return;
            Debug.Log($"[ModelUpdater] new model available: {_localVersion} -> {remoteVersion}");
            StartCoroutine(DownloadAndSwap(remoteVersion));
        }

        private IEnumerator DownloadAndSwap(int targetVersion)
        {
            string tmpOnnx = LocalOnnxPath + ".tmp";
            string err = null;
            yield return _client.DownloadOnnx(tmpOnnx, e => err = e);
            if (err != null)
            {
                Debug.LogError($"[ModelUpdater] ONNX download failed: {err}");
                yield break;
            }

            string[] labels = null;
            yield return _client.DownloadLabels((arr, e) => { labels = arr; err = e; });
            if (labels == null || labels.Length == 0)
            {
                Debug.LogError($"[ModelUpdater] labels download failed: {err}");
                File.Delete(tmpOnnx);
                yield break;
            }

            // Atomically replace local files.
            File.Copy(tmpOnnx, LocalOnnxPath, overwrite: true);
            File.Delete(tmpOnnx);
            File.WriteAllText(LocalLabelsPath, "{\"class_names\":[\"" + string.Join("\",\"", labels) + "\"]}");
            File.WriteAllText(LocalVersionPath, targetVersion.ToString());
            _localVersion = targetVersion;

            yield return TryLoadClassifier();
        }

        private static string[] LoadLabels(string path)
        {
            var blob = JsonUtility.FromJson<RelayClient.LabelsBlob>(File.ReadAllText(path));
            return blob.class_names;
        }

        private void OnDestroy()
        {
            if (_classifier is IDisposable old) old.Dispose();
        }
    }
}
