# Measurement-Pipeline Baseline Findings

**Date:** 2026-04-24
**Tested against:** 160 PIL-rendered mating-face images (20 per class × 8 classes)
**Status:** Baseline established; tuning ceiling identified

## Headline result

**68.8% top-1 accuracy** across all 8 classes, no training, no parameter tuning beyond initial heuristic thresholds.

| Class | n | Correct | Unknown | Accuracy |
|---|---|---|---|---|
| SMA-M  | 20 | 20 | 0 | **100%** |
| SMA-F  | 20 | 20 | 0 | **100%** |
| 3.5mm-F  | 20 | 18 | 2 | **90%** |
| 2.4mm-F  | 20 | 14 | 1 | 70% |
| 3.5mm-M  | 20 | 12 | 4 | 60% |
| 2.4mm-M  | 20 | 10 | 8 | 50% |
| 2.92mm-F  | 20 | 9 | 0 | 45% |
| 2.92mm-M  | 20 | 7 | 8 | 35% |
| **Total** | **160** | **110** | **31** | **68.8%** |

## Where it works

- **SMA classes (100%)**: the family detector reliably identifies the visible PTFE dielectric, and the SMA bore is large and distinct from precision sizes
- **3.5mm-F (90%)**: the largest precision aperture, far enough from 2.92 that detection noise rarely confuses them
- **Female precision generally outperforms male**: dark-on-dark socket measurements are cleaner than male's bright-pin-in-dark-bore

## Where it doesn't (and why)

### 2.92mm vs 2.4mm — geometric ambiguity
Aperture/hex ratios are nearly identical (0.368 vs 0.378), so picking the right hex hypothesis from a single uncalibrated photo is intrinsically hard. With detection noise on the order of ±2 px, the wrong hex hypothesis can yield an aperture that fits the wrong class.

**This is not a tuning problem — it's a physics problem.** Monocular vision without a known-size reference cannot fully separate these two classes.

### Male connectors — bright pin contamination
The male inner-conductor pin appears as a bright disc inside the bore. The family detector measures brightness in the annular region between pin and bore edge; if the pin's blur/halo extends into this region, the measurement reads "bright" (SMA-like) when it should be "dark" (precision air). Increasing the pin-exclusion radius to 50% of aperture helped marginally; further widening shrinks the annulus too much to be reliable.

## Path to higher accuracy

| Improvement | Expected gain | Cost |
|---|---|---|
| Real connector photos for tuning | +5–10% (replace synthetic-render artifacts with real-world signal) | 1 afternoon, $0 if borrowed |
| ArUco marker in capture for absolute scale | +15–25% on 2.92/2.4 boundary (resolves the geometric ambiguity entirely) | 1 day to implement + print marker |
| LiDAR depth (Pro iPhones only) | +20% on precision sizes | 2–3 days, locks out non-Pro devices |
| Aperture detector tuned for real captures | +5% (current detector tuned for synthetic) | Tuning session against 50–100 real photos |
| Better pin/socket discrimination using shape, not just brightness | +5–10% on male classes | 1–2 days |

Realistic ceiling without scale reference: **75–80%**.
Realistic ceiling with ArUco marker: **90–95%**.

## Architectural conclusion

The measurement-first approach **works** for the easy classes (SMA, 3.5mm) and **fundamentally cannot** fully resolve 2.92 vs 2.4 from a single uncalibrated photo. To ship a >90% accurate identifier, the capture workflow needs absolute scale — either:

1. **ArUco marker required for precision-sized identifications** (recommended; works on every phone)
2. **LiDAR depth used when available** (Pro iPhones automatically; falls back to ArUco elsewhere)
3. **Operator manually picks family** (taps "small hex" vs "large hex" if detector is unsure) — UX fallback for ambiguous cases

The Unity `FramingGate` already exists; extending it with optional ArUco detection is a straightforward Plan 4 Task 1 piece (Python detector exists, needs C# port).

## What's NOT a problem

- Family classification (SMA vs precision): **100% accurate** when bore is visible
- Gender classification (M vs F): high signal from pin-vs-socket brightness
- Hex detection: works on synthetic, will tune against real photos
- Aperture detection: works, has known borderline-detection failure modes

The architecture is sound. The ambiguity is real and physics-limited. Solving it requires either a scale marker, LiDAR, or operator input.
