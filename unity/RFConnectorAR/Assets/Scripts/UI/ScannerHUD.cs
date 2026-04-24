using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.UI
{
    public sealed class ScannerHUD : MonoBehaviour
    {
        [SerializeField] private Text _hint;
        [SerializeField] private Text _status;

        public enum ScanState
        {
            Searching,
            Detected,
            Identified,
            LostTracking,
        }

        public void SetState(ScanState state, string detail = null)
        {
            if (_hint != null)
            {
                _hint.text = state switch
                {
                    ScanState.Searching    => "Point at a connector",
                    ScanState.Detected     => "Identifying…",
                    ScanState.Identified   => detail ?? "",
                    ScanState.LostTracking => "Move phone to re-track",
                    _                      => "",
                };
            }
        }

        public void SetStatus(string modelVersion, int confirmationCount)
        {
            if (_status != null)
            {
                _status.text = $"Model {modelVersion}  •  {confirmationCount} confirmations";
            }
        }
    }
}
