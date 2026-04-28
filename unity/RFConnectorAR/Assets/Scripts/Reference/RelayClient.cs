using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace RFConnectorAR.Reference
{
    /// <summary>
    /// HTTP client for the rfcai relay (default https://aired.com/rfcai).
    /// Coroutine-based so it plays well with Unity's main thread.
    ///
    /// Endpoint coverage:
    ///   - GET /model/version          (no auth)
    ///   - GET /model/latest           (auth — returns manifest)
    ///   - GET /model/weights.onnx     (auth — bytes)
    ///   - GET /model/labels           (auth — JSON)
    ///   - POST /uploads               (auth — multipart frames + class)
    ///
    /// Auth: an X-Device-Token header derived from a per-device shared secret.
    /// In production this token is provisioned at app install time; for now
    /// it's read from a config asset (DeviceConfig.token).
    /// </summary>
    public sealed class RelayClient
    {
        public string BaseUrl { get; }
        public string DeviceToken { get; }
        public string DeviceId { get; }

        public RelayClient(string baseUrl, string deviceToken, string deviceId)
        {
            BaseUrl = baseUrl.TrimEnd('/');
            DeviceToken = deviceToken;
            DeviceId = deviceId;
        }

        public IEnumerator GetVersion(Action<int, string> onResult)
        {
            using var req = UnityWebRequest.Get($"{BaseUrl}/model/version");
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                onResult?.Invoke(-1, req.error);
                yield break;
            }
            try
            {
                var blob = JsonUtility.FromJson<VersionBlob>(req.downloadHandler.text);
                onResult?.Invoke(blob.version, null);
            }
            catch (Exception e)
            {
                onResult?.Invoke(-1, e.Message);
            }
        }

        public IEnumerator GetLatestManifest(Action<ModelManifest, string> onResult)
        {
            using var req = UnityWebRequest.Get($"{BaseUrl}/model/latest");
            req.SetRequestHeader("X-Device-Token", DeviceToken);
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                onResult?.Invoke(null, $"{req.responseCode}: {req.error}");
                yield break;
            }
            try
            {
                var manifest = JsonUtility.FromJson<ModelManifest>(req.downloadHandler.text);
                onResult?.Invoke(manifest, null);
            }
            catch (Exception e)
            {
                onResult?.Invoke(null, e.Message);
            }
        }

        public IEnumerator DownloadOnnx(string savePath, Action<string> onComplete)
        {
            using var req = UnityWebRequest.Get($"{BaseUrl}/model/weights.onnx");
            req.SetRequestHeader("X-Device-Token", DeviceToken);
            req.downloadHandler = new DownloadHandlerFile(savePath);
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                onComplete?.Invoke($"download failed: {req.responseCode} {req.error}");
                yield break;
            }
            onComplete?.Invoke(null);
        }

        public IEnumerator DownloadLabels(Action<string[], string> onResult)
        {
            using var req = UnityWebRequest.Get($"{BaseUrl}/model/labels");
            req.SetRequestHeader("X-Device-Token", DeviceToken);
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                onResult?.Invoke(null, $"{req.responseCode}: {req.error}");
                yield break;
            }
            try
            {
                var blob = JsonUtility.FromJson<LabelsBlob>(req.downloadHandler.text);
                onResult?.Invoke(blob.class_names, null);
            }
            catch (Exception e)
            {
                onResult?.Invoke(null, e.Message);
            }
        }

        /// <summary>
        /// POST /uploads — submits a JPEG-encoded frame with the user's
        /// claimed class. Used by the inline-correction flow.
        /// </summary>
        public IEnumerator UploadFrame(byte[] jpegBytes, string claimedClass, string captureReason,
                                       Action<string, string> onResult)
        {
            var form = new List<IMultipartFormSection>
            {
                new MultipartFormDataSection("claimed_class", claimedClass),
                new MultipartFormDataSection("device_id", DeviceId),
                new MultipartFormDataSection("capture_reason", captureReason),
                new MultipartFormFileSection("frames", jpegBytes, "frame.jpg", "image/jpeg"),
            };
            using var req = UnityWebRequest.Post($"{BaseUrl}/uploads", form);
            req.SetRequestHeader("X-Device-Token", DeviceToken);
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                onResult?.Invoke(null, $"{req.responseCode}: {req.error}");
                yield break;
            }
            try
            {
                var blob = JsonUtility.FromJson<UploadBlob>(req.downloadHandler.text);
                onResult?.Invoke(blob.upload_id, null);
            }
            catch (Exception e)
            {
                onResult?.Invoke(null, e.Message);
            }
        }

        // ---- DTOs --------------------------------------------------------

        [Serializable]
        public class VersionBlob { public int version; }

        [Serializable]
        public class ModelManifest
        {
            public int version;
            public string weights_filename;
            public string weights_onnx_filename;
            public string labels_filename;
            public string weights_sha256;
            public string weights_onnx_sha256;
            public string labels_sha256;
            public string trained_at;
            public string weights_url;
            public string weights_onnx_url;
            public string labels_url;
        }

        [Serializable]
        public class LabelsBlob
        {
            public string[] class_names;
            public int input_size;
            public string architecture;
        }

        [Serializable]
        private class UploadBlob
        {
            public string upload_id;
            public int n_frames_received;
            public string claimed_class;
        }
    }
}
