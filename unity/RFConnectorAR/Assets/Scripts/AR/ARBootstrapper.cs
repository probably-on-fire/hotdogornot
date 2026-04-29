using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// First-launch AR bootstrap. Handles three things the user shouldn't
    /// have to think about:
    ///
    ///   1. Camera permission — AR Foundation requests it automatically
    ///      when the session enables, but we explicitly poll so we can
    ///      show a friendly UI if the user denies.
    ///   2. ARCore Services for AR availability — separate Play Store
    ///      package; if missing, AR Foundation can prompt install via
    ///      ARSession.InstallAsync. (For Play Store distribution this is
    ///      automatic, but sideloaded APKs need to drive the prompt
    ///      manually.)
    ///   3. ARSession start — once both above are green, enable the
    ///      session GameObject so the camera feed comes alive.
    ///
    /// Drop this MonoBehaviour anywhere in the scene; assign the
    /// ARSession reference. It runs once on Start and handles everything.
    ///
    /// Optional: assign a status Text so the user sees what's happening
    /// (e.g. "Installing AR services..." instead of a black screen).
    /// </summary>
    public sealed class ARBootstrapper : MonoBehaviour
    {
        [SerializeField] private ARSession session;
        [SerializeField] private GameObject overlayRoot;       // shown during bootstrap
        [SerializeField] private Text statusText;              // optional

        private void Awake()
        {
            ShowOverlay(true);
            SetStatus("Initializing AR...");
        }

        private IEnumerator Start()
        {
            yield return null;   // let Awake settle

            // Step 1: Check ARCore availability. ARSession.CheckAvailability
            // is a static API that works whether or not the local session is
            // enabled, so we don't need to gate the session start on this.
            yield return ARSession.CheckAvailability();

            switch (ARSession.state)
            {
                case ARSessionState.Unsupported:
                    SetStatus("This device doesn't support AR. The classifier still works on tap-to-capture, but the live AR feed is disabled.");
                    // Camera will not work. Leave overlay up so user knows why.
                    yield break;

                case ARSessionState.NeedsInstall:
                    SetStatus("Installing AR services from Google Play... (one-time setup)");
                    yield return ARSession.Install();
                    break;

                case ARSessionState.Ready:
                case ARSessionState.SessionInitializing:
                case ARSessionState.SessionTracking:
                    // Already good to go.
                    break;
            }

            // Step 2: Re-check after install attempt.
            if (ARSession.state == ARSessionState.NeedsInstall || ARSession.state == ARSessionState.Unsupported)
            {
                SetStatus(
                    "AR services couldn't be installed. Open Google Play and search for " +
                    "\"Google Play Services for AR\" to install manually.");
                yield break;
            }

            // Step 3: Ensure the session is enabled. AR Foundation requests
            // camera permission as part of session start; if the user
            // denies, the session moves to a NoCamera state.
            SetStatus("Starting camera...");
            if (session != null && !session.enabled) session.enabled = true;

            // Wait up to 10 s for tracking or a definitive failure.
            float deadline = Time.time + 10f;
            while (Time.time < deadline)
            {
                if (ARSession.state == ARSessionState.SessionTracking)
                {
                    SetStatus("");
                    ShowOverlay(false);
                    yield break;
                }
                if (ARSession.state == ARSessionState.Unsupported ||
                    ARSession.state == ARSessionState.NeedsInstall)
                {
                    SetStatus($"AR session failed: {ARSession.state}");
                    yield break;
                }
                yield return null;
            }

            // Tracking didn't kick in within 10s — usually means permission
            // denied or hardware not available. Show a hint but leave the
            // session enabled in case it recovers.
            SetStatus(
                "Camera not available. If you denied camera permission, " +
                "go to Settings → Apps → RF Connector → Permissions → Camera → Allow.");
        }

        private void SetStatus(string message)
        {
            if (statusText != null) statusText.text = message;
            if (!string.IsNullOrEmpty(message)) Debug.Log($"[ARBootstrapper] {message}");
        }

        private void ShowOverlay(bool visible)
        {
            if (overlayRoot != null) overlayRoot.SetActive(visible);
        }
    }
}
