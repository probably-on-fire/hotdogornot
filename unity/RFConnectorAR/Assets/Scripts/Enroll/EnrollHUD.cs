using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.Enroll
{
    public sealed class EnrollHUD : MonoBehaviour
    {
        [SerializeField] private InputField _classNameInput;
        [SerializeField] private Button _startButton;
        [SerializeField] private Text _progressText;
        [SerializeField] private Slider _progressBar;

        public string ClassName => _classNameInput != null ? _classNameInput.text : "";

        public void OnStartClicked(System.Action<string> onStart)
        {
            if (_startButton != null)
            {
                _startButton.onClick.RemoveAllListeners();
                _startButton.onClick.AddListener(() => onStart?.Invoke(ClassName));
            }
        }

        public void SetIdle() => SetStatus("Type a class name and press Start.", 0f);
        public void SetCapturing(int captured, int total)
            => SetStatus($"Capturing… {captured} / {total}", (float)captured / total);
        public void SetComplete(int captured, int prototypes)
            => SetStatus($"Done. {captured} frames → {prototypes} prototypes saved.", 1f);
        public void SetError(string msg) => SetStatus(msg, 0f);

        private void SetStatus(string text, float progress)
        {
            if (_progressText != null) _progressText.text = text;
            if (_progressBar != null) _progressBar.value = progress;
        }
    }
}
