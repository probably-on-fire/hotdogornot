using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Wraps AR Foundation's ARAnchorManager with a debounce: if a new detection
    /// lands within <see cref="_mergeRadiusM"/> of an existing anchor, we reuse
    /// that anchor rather than spawning another.
    ///
    /// AR Foundation 6 exposes anchor creation only via
    /// TryAddAnchorAsync(Pose) → Awaitable&lt;Result&lt;ARAnchor&gt;&gt;, so
    /// GetOrCreateAnchor is best-effort synchronous: on a novel position it
    /// kicks off an async anchor-create request and returns null for the
    /// current frame; a subsequent call at the same position (a frame or two
    /// later, once the async request resolves) will find the anchor in the
    /// cache and return it.
    /// </summary>
    [RequireComponent(typeof(ARAnchorManager))]
    public sealed class AnchorManager : MonoBehaviour
    {
        [SerializeField, Tooltip("Maximum distance in meters to consider an existing anchor a match")]
        private float _mergeRadiusM = 0.02f;

        private ARAnchorManager _anchorManager;
        private readonly List<ARAnchor> _anchors = new();
        private readonly List<Vector3> _pendingPositions = new();

        private void Awake() { _anchorManager = GetComponent<ARAnchorManager>(); }

        public ARAnchor GetOrCreateAnchor(Vector3 worldPosition, Quaternion worldRotation)
        {
            foreach (var a in _anchors)
            {
                if (a == null) continue;
                if (Vector3.Distance(a.transform.position, worldPosition) <= _mergeRadiusM)
                {
                    return a;
                }
            }

            // Don't spawn a second concurrent async request for the same position.
            foreach (var p in _pendingPositions)
            {
                if (Vector3.Distance(p, worldPosition) <= _mergeRadiusM) return null;
            }

            _pendingPositions.Add(worldPosition);
            _ = RequestAnchorAsync(new Pose(worldPosition, worldRotation));
            return null;
        }

        private async Awaitable RequestAnchorAsync(Pose pose)
        {
            try
            {
                var result = await _anchorManager.TryAddAnchorAsync(pose);
                if (result.status.IsSuccess() && result.value != null)
                {
                    _anchors.Add(result.value);
                }
            }
            finally
            {
                _pendingPositions.Remove(pose.position);
            }
        }

        public void RemoveStale()
        {
            for (int i = _anchors.Count - 1; i >= 0; i--)
            {
                if (_anchors[i] == null) _anchors.RemoveAt(i);
            }
        }
    }
}
