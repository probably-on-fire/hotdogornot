using RFConnectorAR.Reference;
using UnityEngine;

namespace RFConnectorAR.Curate
{
    public sealed class CurateController : MonoBehaviour
    {
        [SerializeField] private CurateHUD _hud;
        [SerializeField] private int _embeddingDim = 128;

        private OnDeviceReferenceStore _store;

        private void Awake()
        {
            string path = System.IO.Path.Combine(Application.persistentDataPath, "references.bin");
            _store = new OnDeviceReferenceStore(path, _embeddingDim);
        }

        private void Start()
        {
            if (_hud != null)
            {
                _hud.OnDeleteClicked = HandleDelete;
                _hud.Render(_store.ClassNames);
            }
        }

        private void HandleDelete(string className)
        {
            _store.Delete(className);
            _hud?.Render(_store.ClassNames);
        }
    }
}
