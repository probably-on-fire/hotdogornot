# Architecture — How v18 Works

The deployed model is **v18**: a fine-tuned ResNet-18 with a single
linear classifier head, sandwiched between a rembg foreground filter
(in front, gates predictions) and 5× test-time augmentation (behind,
smooths the output).

This doc visualizes both pipelines. For *why* we ended up here see
`classifier_journey.md`. For deploy/retrain ops see `runbook.md`.

Roadmap note: `../../IMPLEMENTATION_PLAN.md` and `../../TASKS.md` now
define the next architecture direction. The current v18 path remains the
compatibility baseline, but the target system adds detector training,
multi-head attribute classification, structured spec lookup, abstention
states, and mobile/desktop deployment options.

Detailed Graphviz source for the full proposed I/O architecture:

```bash
dot -Tpng ../../docs/SOFTWARE_ARCHITECTURE.dot -o ../../docs/SOFTWARE_ARCHITECTURE.png
dot -Tsvg ../../docs/SOFTWARE_ARCHITECTURE.dot -o ../../docs/SOFTWARE_ARCHITECTURE.svg
```

Source: `../../docs/SOFTWARE_ARCHITECTURE.dot`

Rendered copies:

- `../../docs/SOFTWARE_ARCHITECTURE.svg`
- `../../docs/SOFTWARE_ARCHITECTURE.png`

---

## Inference flow (live in production)

```mermaid
flowchart TD
    Start([📱 User taps shutter]) --> Upload[POST /rfcai/predict<br/>~3 MB JPEG over HTTPS]
    Upload --> Decode[cv2.imdecode → BGR]

    Decode --> FGFilter{rembg fg filter<br/>foreground area<br/>≥ threshold?}
    FGFilter -->|No| EmptyResp[Return predictions: empty]
    EmptyResp --> NoConnector([📱 'No connector detected'])

    FGFilter -->|Yes| Composite[rembg full silhouette<br/>composite on white bg<br/>removes wood-bench bias]
    Composite --> Hough[cv2.HoughCircles<br/>1–3 candidate crops]
    Hough --> EachCrop[For each crop]
    EachCrop --> Resize[Resize → 224×224<br/>ImageNet normalize]
    Resize --> Model[ResNet-18 backbone<br/>+ Linear 512→6]
    Model --> TTA[5× TTA<br/>identity · h-flip<br/>+90° · -90° · center-crop]
    TTA --> Avg[Average softmax<br/>across 5 augmentations]
    Avg --> Rank[Pick highest-confidence<br/>across all crops]
    Rank --> JSON[Return JSON]

    JSON --> Gate1{confidence<br/>≥ 0.40?}
    Gate1 -->|No| LowConf([📱 'No clear connector'])
    Gate1 -->|Yes| Gate2{bbox area<br/>≥ 2% of image?}
    Gate2 -->|No| TooSmall([📱 'Move closer'])
    Gate2 -->|Yes| Result([📱 Result panel<br/>HOT DOG / NOT HOT DOG<br/>+ family + class + chips])

    classDef phone fill:#1d222c,stroke:#4f8cff,color:#fff
    classDef decision fill:#3a2f15,stroke:#ffb347,color:#fff
    classDef terminal fill:#1a3320,stroke:#4ade80,color:#fff
    classDef reject fill:#3a1a1a,stroke:#e63946,color:#fff
    class Start,Upload,JSON phone
    class FGFilter,Gate1,Gate2 decision
    class Result terminal
    class NoConnector,LowConf,TooSmall reject
```

### Stage timing (CPU on the box, ~250–500 ms total)

| Stage | ms |
|---|---|
| HTTPS upload (~3 MB JPEG) | 100–200 |
| JPEG decode | 5 |
| rembg fg filter | 80–120 |
| rembg full silhouette + composite | 60–80 |
| Hough Circle crop | 20 |
| ResNet-18 forward × 5 TTA × N crops | 30–80 |
| Response + render on phone | 30 |

GPUs on the box are now wired up (`runbook.md`) but the predict
service still runs CPU — moving classifier inference to GPU would cut
that line roughly in half.

### Production env knobs (`/etc/default/rfcai-predict`)

```
RFCAI_FG_FILTER=1                # rembg foreground gate (Stage 2)
RFCAI_CLASSIFY_ON_CLEANED=1      # composite on white before classify (Stage 3)
RFCAI_TTA=5                      # 5× test-time augmentation (Stage 5)
RFCAI_MIN_CONFIDENCE=0.40        # phone-side gate (Result panel)
RFCAI_MIN_BBOX_FRACTION=0.02     # phone-side gate (Result panel)
```

---

## Training recipe (what produced v18)

```mermaid
flowchart TD
    subgraph Sources["📥 Data sources"]
      V[3 source videos<br/>~10s each<br/>2.4mm · 2.92mm · 3.5mm]
      P[Contributed phone shots<br/>via Flutter contribute tab]
    end

    V --> Extract[extract_frames<br/>fps=4 → JPGs]
    Extract --> CropD[connector_crops<br/>Hough-detected]
    CropD --> Real[~5,545 real samples<br/>data/labeled/embedder/]
    P --> Real

    Real --> Synth[synthesize_from_clean.py<br/>━━━━━━━━━<br/>rembg silhouette →<br/>composite on:<br/>white · gray · beige<br/>wood-noise · skin · photo<br/>at 50–90% scale<br/>+ perspective · motion blur<br/>+ color jitter · rotation]
    Synth --> All[13,849 total<br/>~60% synthetic]

    All --> Balance[balance-to-smallest<br/>= 817 per class<br/>= 4,902 effective]
    Balance --> Split[dHash-grouped<br/>train/val split<br/>train 4,043 · val 859]

    Split --> Loop

    subgraph Loop["🔁 Fine-tune loop · 20 epochs"]
      direction TB
      Aug[Light augmentation<br/>h-flip · rot ±10° · color jitter]
      Aug --> FW[Forward<br/>ResNet-18 + Linear 512→6]
      FW --> LossNode[CrossEntropyLoss]
      LossNode --> Opt[Adam lr 3e-4<br/>cosine decay → 0]
      Opt --> WRS[WeightedRandomSampler<br/>capped]
      WRS --> Aug
    end

    Loop --> Weights[weights.synth_20ep.pt<br/>44 MB]
    Weights --> Deploy[systemctl restart<br/>rfcai-predict]
    Deploy --> Eval{Eval on test_holdout<br/>8 phone shots}
    Eval --> Numbers[Full 75% · Family 75%<br/>Gender 87.5%<br/>FP on backgrounds 0%]

    classDef source fill:#1d222c,stroke:#4f8cff,color:#fff
    classDef stage fill:#1a2a1a,stroke:#4ade80,color:#fff
    classDef loop fill:#2a1f15,stroke:#ffb347,color:#fff
    classDef artifact fill:#2a1a2a,stroke:#e63946,color:#fff
    class V,P source
    class Real,All,Balance,Split stage
    class Aug,FW,LossNode,Opt,WRS loop
    class Weights,Numbers artifact
```

---

## Just the model

```mermaid
flowchart LR
    Input[224×224×3 RGB<br/>ImageNet normalized] --> R[ResNet-18<br/>backbone<br/>~11M params]
    R --> Feat[512-dim feature vector]
    Feat --> FC[Linear 512 → 6]
    FC --> SM[Softmax]
    SM --> Out[3.5mm-M · 3.5mm-F<br/>2.92mm-M · 2.92mm-F<br/>2.4mm-M · 2.4mm-F]

    classDef io fill:#1d222c,stroke:#4f8cff,color:#fff
    classDef trainable fill:#2a1a2a,stroke:#e63946,color:#fff
    classDef pretrained fill:#1a2a1a,stroke:#4ade80,color:#fff
    class Input,Out io
    class R pretrained
    class FC,SM trainable
```

SMA-M / SMA-F and 1.85mm-M / 1.85mm-F are intentionally absent from
the head — we have zero training data for either family, so the
model would have nothing to learn. Both families are wired through
the data + UI layer (Flutter chips, labeler folders, auto_retrain
canonical list). The next retrain that finds ≥5 samples per class
will expand the head automatically.

---

## Hyperparameters at a glance

| Setting | Value | Why this value |
|---|---|---|
| Backbone | ResNet-18 | Bigger backbones overfit (ResNet-50 → 0% Full) |
| Head | `Linear(512→6)` | MLP head overfit (12.5% Full) |
| Input size | 224×224 | 384 caused training collapse |
| Epochs | 20 | 12 underfits, 28 overfits |
| Optimizer | Adam, lr=3e-4 | default, cosine decay to 0 |
| Loss | CrossEntropy | label smoothing + focal both regressed |
| Batch sampler | WeightedRandomSampler, capped | uncapped → minority dominates |
| Class balance | balance-to-smallest (817/class) | otherwise everything → 3.5mm-M |
| Synth ratio | ~60% (8,304 / 13,849) | the key generalization win |
| TTA at inference | 5× (id, h-flip, ±90°, ctr-crop) | recovers ~3-5 pp |
| FG filter | rembg U²-Net | enables 0% false positives |
| Classify-on-cleaned | rembg silhouette → white bg | +13 pp on held-out |

---

## Why every alternative architecture lost

We benchmarked five variants against v18 on the same 8-photo holdout.
Same data, same training infrastructure, only the model shape changed.

| Variant | Val acc | Held-out (Full / Family / Gender) |
|---|---|---|
| **ResNet-18 + linear (v18)** | 0.84 | **75% / 75% / 87.5%** |
| ResNet-18 + MLP head (deeper FC) | 0.84 | 12.5% / 37.5% / 25% |
| ResNet-18 two-head (separate fam + gender) | 0.84 | 25% / 75% / 37.5% |
| ResNet-50 backbone (~25M params) | 0.84 | 0% / 50% / 12.5% |
| Prototypical Networks (128-d, episodic, cosine) | 0.85 | 12.5% / 50% / 37.5% |

All five converge to the same val_acc but diverge wildly on held-out.
The bottleneck is data, not model capacity — bigger models just
memorize the wood-bench training distribution harder. The lever is
varied real-world phone shots (collected via the Flutter Contribute
tab), not architecture changes. See `classifier_journey.md` for the
full trial-by-trial writeup.
