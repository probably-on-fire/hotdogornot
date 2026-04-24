using UnityEngine;
using UnityEngine.SceneManagement;

namespace RFConnectorAR.App
{
    /// <summary>
    /// Scene-switching entry points for the three top-level app modes.
    /// Wire these to UI buttons via UnityEvents in the inspector.
    /// </summary>
    public sealed class ModeRouter : MonoBehaviour
    {
        public void GoToScanner() => SceneManager.LoadScene("Scanner");
        public void GoToEnroll()  => SceneManager.LoadScene("Enroll");
        public void GoToCurate()  => SceneManager.LoadScene("Curate");
    }
}
