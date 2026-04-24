# Field Test Set Capture Protocol

The random-split test accuracy is ~99% and lies. Real deployment accuracy is
what matters. This doc specifies the per-connector field-test capture set that
gets used for honest eval numbers.

## Capture requirements

For each of the 8 connector classes, capture **at least 25 images** spanning:

1. **Different lighting**
   - Direct sun / outdoor shade
   - Fluorescent (typical lab)
   - LED / warm room
   - Low light / dim

2. **Different phones**
   - An iPhone (ideally iPhone 13/14/15 non-Pro or Pro)
   - An Android (Pixel, Galaxy, OnePlus — any modern flagship)
   - Optionally, a budget Android phone to see the hard-case behavior

3. **Different operators** (at least 2)
   - Hand-held capture from a few different people; each person holds the
     connector and shoots slightly differently.

4. **Different conditions**
   - Plain white background (baseline)
   - On a lab bench with clutter
   - In an operator's hand
   - Through a plastic anti-static bag (how they arrive from supply)
   - With a scale marker (ArUco) placed next to the connector
   - Without a scale marker

5. **Deliberate hard cases** (at least 3 per class)
   - Head-on / axial view at infinity focus (worst case for size estimation)
   - Extreme oblique angle
   - Very close range (where DoF blurs parts of the connector)
   - Motion blur (simulate hand-shake during capture)

## Directory layout

```
training/data/field_test/
├── SMA-M/
│   ├── aruco_plain_iphone_001.jpg
│   ├── aruco_bench_pixel_001.jpg
│   ├── bare_hand_iphone_001.jpg
│   ├── bag_flu_pixel_001.jpg
│   └── ...
├── SMA-F/
└── ...
```

Filename convention: `<marker-status>_<background>_<phone>_<index>.jpg`.
- marker-status ∈ {aruco, bare}
- background ∈ {plain, bench, hand, bag, sun, dim}
- phone ∈ {iphone, pixel, galaxy, budget}

## Eval protocol

- `rfconnectorai.inference.eval` reports per-slice accuracy (by marker-status,
  by phone, etc.) when run on this directory.
- Report:
  - Overall top-1 accuracy
  - Accuracy with ArUco marker vs. without
  - Per-phone accuracy
  - Confusion matrix
  - Per-class recall, especially for 2.4/2.92/3.5

## Publishing accuracy numbers

Use these field-test numbers externally. Do not publish the random-split
numbers — they overstate performance by 10–20 points for fine-grained
problems and set unrealistic expectations.
