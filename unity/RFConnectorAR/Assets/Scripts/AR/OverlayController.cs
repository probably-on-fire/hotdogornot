using System.Collections.Generic;
using RFConnectorAR.Perception;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

namespace RFConnectorAR.AR
{
    /// <summary>
    /// Spawns and updates ring + label visuals for each verdict. Visuals are
    /// parented under ARAnchors so they stay world-locked.
    ///
    /// Plan 2 keeps this simple: one ring + one TextMesh label per verdict.
    /// Plan 2b swaps the label for a richer specs card and adds confidence glow.
    /// </summary>
    public sealed class OverlayController : MonoBehaviour
    {
        [SerializeField] private GameObject _ringPrefab;
        [SerializeField] private GameObject _labelPrefab;
        [SerializeField] private AnchorManager _anchorManager;

        private readonly Dictionary<ARAnchor, (GameObject ring, TextMesh label)> _overlays = new();

        public void UpdateVerdicts(IReadOnlyList<Verdict> verdicts)
        {
            var live = new HashSet<ARAnchor>();

            foreach (var v in verdicts)
            {
                if (v.WorldPosition is not Vector3 pos) continue;

                var anchor = _anchorManager.GetOrCreateAnchor(pos, Quaternion.identity);
                if (anchor == null) continue;

                if (!_overlays.TryGetValue(anchor, out var ov))
                {
                    var ring = Instantiate(_ringPrefab, anchor.transform);
                    var labelGo = Instantiate(_labelPrefab, anchor.transform);
                    labelGo.transform.localPosition = new Vector3(0f, 0.05f, 0f);
                    var label = labelGo.GetComponent<TextMesh>();
                    ov = (ring, label);
                    _overlays[anchor] = ov;
                }

                live.Add(anchor);
                ov.label.text = FormatLabel(v);
                ApplyConfidenceStyle(ov.ring, v.Confidence);
            }

            foreach (var kv in _overlays)
            {
                if (!live.Contains(kv.Key))
                {
                    if (kv.Value.ring != null) kv.Value.ring.SetActive(false);
                    if (kv.Value.label != null) kv.Value.label.gameObject.SetActive(false);
                }
                else
                {
                    if (kv.Value.ring != null) kv.Value.ring.SetActive(true);
                    if (kv.Value.label != null) kv.Value.label.gameObject.SetActive(true);
                }
            }
        }

        private static string FormatLabel(Verdict v)
        {
            if (v.MeasuredDiameterMm is float mm)
                return $"{v.ClassName}\n{mm:F2} mm\n[{v.Confidence}]";
            return $"{v.ClassName}\n[{v.Confidence}]";
        }

        private static void ApplyConfidenceStyle(GameObject ring, ConfidenceLevel level)
        {
            var renderer = ring.GetComponentInChildren<Renderer>();
            if (renderer == null) return;
            var mat = renderer.material;
            mat.color = level switch
            {
                ConfidenceLevel.High    => new Color(0.2f, 0.9f, 0.4f),
                ConfidenceLevel.Medium  => new Color(0.95f, 0.75f, 0.2f),
                ConfidenceLevel.Low     => new Color(0.9f, 0.3f, 0.2f),
                _                       => new Color(0.7f, 0.7f, 0.7f),
            };
        }
    }
}
