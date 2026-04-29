using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Records a short burst of frames from <see cref="CameraFrameSource"/>
    /// for upload. Not actually a video file (mp4) — we just grab N JPEG
    /// frames at a target FPS and ship them as a multipart upload to the
    /// relay's /uploads endpoint, which expects 1-200 frames.
    ///
    /// Why not real video: the backend's ingestion daemon already extracts
    /// frames from videos via the Process Video page; doing the extraction
    /// on-device saves the upload bandwidth of an mp4 wrapper and avoids a
    /// codec dependency on the phone (which Android can be picky about).
    ///
    /// Recommended usage: 5 seconds × 4 fps = 20 frames, ~1-2 MB total.
    /// </summary>
    public sealed class VideoCapture : MonoBehaviour
    {
        [SerializeField] private CameraFrameSource cameraSource;

        public bool IsRecording { get; private set; }
        public int FramesCaptured { get; private set; }

        /// <summary>
        /// Capture <paramref name="durationSeconds"/> seconds at <paramref name="targetFps"/>.
        /// Calls <paramref name="onComplete"/> with the JPEG-encoded frames
        /// once done (or empty array on failure).
        /// </summary>
        public IEnumerator Record(float durationSeconds, float targetFps,
                                  Action<byte[][]> onComplete,
                                  Action<int, int> onProgress = null)
        {
            if (IsRecording)
            {
                Debug.LogWarning("[VideoCapture] already recording");
                onComplete?.Invoke(Array.Empty<byte[]>());
                yield break;
            }
            if (cameraSource == null)
            {
                Debug.LogError("[VideoCapture] no CameraFrameSource assigned");
                onComplete?.Invoke(Array.Empty<byte[]>());
                yield break;
            }

            IsRecording = true;
            FramesCaptured = 0;

            int totalFrames = Mathf.Max(1, Mathf.RoundToInt(durationSeconds * targetFps));
            float interval = 1f / Mathf.Max(0.5f, targetFps);
            var captured = new List<byte[]>(totalFrames);

            float elapsed = 0f;
            float nextCaptureAt = 0f;
            while (elapsed < durationSeconds && captured.Count < totalFrames)
            {
                if (elapsed >= nextCaptureAt && cameraSource.HasFrame)
                {
                    var jpeg = cameraSource.LatestRgb.EncodeToJPG(quality: 88);
                    if (jpeg != null && jpeg.Length > 0)
                    {
                        captured.Add(jpeg);
                        FramesCaptured = captured.Count;
                        onProgress?.Invoke(captured.Count, totalFrames);
                    }
                    nextCaptureAt = elapsed + interval;
                }
                yield return null;
                elapsed += Time.deltaTime;
            }

            IsRecording = false;
            onComplete?.Invoke(captured.ToArray());
        }
    }
}
