using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Fetches the current AR camera frame as a Texture2D. Unity's AR Foundation
    /// exposes a native camera image via ARCameraManager.TryAcquireLatestCpuImage;
    /// we convert it to RGBA32 for downstream CPU-side stages. A real-time
    /// optimization later (Plan 3) keeps the image on GPU.
    /// </summary>
    [RequireComponent(typeof(ARCameraManager))]
    public sealed class CameraFrameSource : MonoBehaviour
    {
        private ARCameraManager _cameraManager;
        private Texture2D _rgb;

        public Texture2D LatestRgb => _rgb;

        public bool HasFrame => _rgb != null;

        private void Awake()
        {
            _cameraManager = GetComponent<ARCameraManager>();
        }

        private void OnEnable()
        {
            _cameraManager.frameReceived += OnFrameReceived;
        }

        private void OnDisable()
        {
            _cameraManager.frameReceived -= OnFrameReceived;
        }

        private void OnFrameReceived(ARCameraFrameEventArgs _)
        {
            if (!_cameraManager.TryAcquireLatestCpuImage(out var image)) return;
            try
            {
                if (_rgb == null || _rgb.width != image.width || _rgb.height != image.height)
                {
                    if (_rgb != null) Destroy(_rgb);
                    _rgb = new Texture2D(image.width, image.height, TextureFormat.RGBA32, false);
                }

                var conversionParams = new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(0, 0, image.width, image.height),
                    outputDimensions = new Vector2Int(image.width, image.height),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.None,
                };

                var data = _rgb.GetRawTextureData<byte>();
                unsafe
                {
                    image.Convert(
                        conversionParams,
                        new IntPtr(NativeArrayUnsafeUtility.GetUnsafePtr(data)),
                        data.Length);
                }
                _rgb.Apply();
            }
            finally
            {
                image.Dispose();
            }
        }
    }
}
