# Holdout eval

- url: `http://127.0.0.1:8503/predict`
- holdout: `/opt/rfcai/repo/training/data/test_holdout`
- images: 8 (detected: 8)
- accuracy: Full 75.0% · Family 75.0% · Gender 87.5% · Detect-recall 100.0%
- latency mean/max: 9694 ms / 14071 ms

| truth | pred | family | gender | conf | crop |
|---|---|---|---|---|---|
| 2.4mm-F | 2.4mm-F | 2.4mm | F | 0.53 | hough |
| 2.4mm-F | 2.4mm-F | 2.4mm | F | 0.77 | hough |
| 2.4mm-F | 2.4mm-F | 2.4mm | F | 0.86 | hough |
| 2.4mm-M | 3.5mm-F | 3.5mm | F | 0.61 | hough |
| 2.92mm-F | 2.4mm-F | 2.4mm | F | 0.53 | hough |
| 2.92mm-M | 2.92mm-M | 2.92mm | M | 0.64 | hough |
| 3.5mm-F | 3.5mm-F | 3.5mm | F | 0.74 | hough |
| 3.5mm-M | 3.5mm-M | 3.5mm | M | 0.57 | hough |