using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace RFConnectorAR.UI
{
    /// <summary>
    /// Trivial bottom-tab toggle: shows one panel, hides the others, swaps
    /// active tab styling. No scene loading — all panels live in the same
    /// scene as siblings of a Canvas.
    /// </summary>
    public sealed class TabRouter : MonoBehaviour
    {
        [System.Serializable]
        public class Tab
        {
            public string name;
            public Button button;
            public GameObject panel;
            public Image buttonBackground;   // optional, for active styling
        }

        [SerializeField] private List<Tab> tabs = new();
        [SerializeField] private Color activeColor = new(0.2f, 0.4f, 0.8f);
        [SerializeField] private Color inactiveColor = new(0.15f, 0.15f, 0.2f);
        [SerializeField] private int defaultTabIndex = 0;

        private void Start()
        {
            for (int i = 0; i < tabs.Count; i++)
            {
                int index = i;   // capture
                if (tabs[i].button != null)
                    tabs[i].button.onClick.AddListener(() => Show(index));
            }
            Show(defaultTabIndex);
        }

        public void Show(int index)
        {
            for (int i = 0; i < tabs.Count; i++)
            {
                if (tabs[i].panel != null) tabs[i].panel.SetActive(i == index);
                if (tabs[i].buttonBackground != null)
                    tabs[i].buttonBackground.color = (i == index) ? activeColor : inactiveColor;
            }
        }
    }
}
