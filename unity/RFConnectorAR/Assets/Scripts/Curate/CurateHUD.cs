using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.Curate
{
    public sealed class CurateHUD : MonoBehaviour
    {
        [SerializeField] private RectTransform _listRoot;
        [SerializeField] private GameObject _rowPrefab;
        [SerializeField] private Text _emptyMessage;

        public Action<string> OnDeleteClicked;

        public void Render(IReadOnlyList<string> classNames)
        {
            for (int i = _listRoot.childCount - 1; i >= 0; i--)
            {
                Destroy(_listRoot.GetChild(i).gameObject);
            }

            if (classNames.Count == 0)
            {
                if (_emptyMessage != null) _emptyMessage.gameObject.SetActive(true);
                return;
            }
            if (_emptyMessage != null) _emptyMessage.gameObject.SetActive(false);

            foreach (var name in classNames)
            {
                var row = Instantiate(_rowPrefab, _listRoot);
                var label = row.GetComponentInChildren<Text>();
                if (label != null) label.text = name;
                var deleteBtn = row.GetComponentInChildren<Button>();
                if (deleteBtn != null)
                {
                    string captured = name;
                    deleteBtn.onClick.AddListener(() => OnDeleteClicked?.Invoke(captured));
                }
            }
        }
    }
}
