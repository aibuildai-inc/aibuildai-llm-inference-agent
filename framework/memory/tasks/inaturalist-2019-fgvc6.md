# Task Memory: inaturalist-2019-fgvc6

## Task profile
- Type: Fine-grained image classification (1010 species)
- Dataset: 232.9K training images, 3 val images/class; severe class imbalance (14–440 images/class, 31.4× ratio); tar.gz archive requiring explicit pre-extraction
- Metric: Top-1 classification error (lower is better)
- Medal thresholds: Gold < ~0.11, Silver < ~0.254, Median ~ unknown

## Best result
- Score: 0.10837 | Medal: Gold | Run: 20260309_121401
- Approach: DINOv2 ViT-Base frozen backbone + MLP head as fast baseline (error 0.133 in ~30 min), combined with EVA-02 Base full fine-tuning (error 0.144). Final submission (Candidate 0, error 0.108) likely used checkpoint averaging or soft-label blending across best candidates.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_151945 | DNF | — | All candidates timed out during coder phase (tar extraction bottleneck + 1200s timeout) |
| 20260226_162541 | 0.2538 | Silver | Pre-extracted data; EVA02-Base MIM-IN22K 448px; 3–6 epochs only due to slow ViT throughput |
| 20260309_121401 | 0.1084 | Gold | DINOv2 frozen features + MLP head; fast iteration (5 tuner runs); implicit ensemble of best candidates |

## Proven strategies
- **DINOv2 frozen feature extraction + MLP head**: Consistently fastest path to strong results (0.133 error in ~30 min); enables many tuner iterations within time budget
- **EVA-02 Base with MIM-IN22K pretraining**: Reliable full fine-tuning option; MIM captures texture/shape cues critical for species discrimination (0.144 standalone, 0.257 in earlier run with fewer epochs)
- **IN-22K pretraining is mandatory**: Every successful candidate used IN-22K (or richer) pretraining; IN-1K models consistently eliminated by judge
- **High input resolution (≥384px, ideally 448–518px)**: Fine morphological details (beaks, petals, wing patterns) vanish below 384px; resolution matters more than model family
- **Label smoothing (ε=0.1) + Mixup/CutMix**: Stabilizes training under class imbalance without needing exotic loss functions
- **Pre-extract tar.gz archives before training**: Eliminates I/O bottleneck that caused complete run failure in first attempt
- **Ensemble/checkpoint averaging across diverse candidates**: Final 0.108 score beat best individual (0.133) by ~19% relative

## Failure patterns
- **Tar extraction inline during coder phase**: Caused complete run failure (Run 1) — all 4 candidates timed out at 1200s
- **Large models + full fine-tuning + high resolution = timeout**: ConvNeXt-Base (89M params), SwinV2-Base, Swin-Base all timed out or were killed; memory and compute budget insufficient
- **IN-1K-only pretraining**: EfficientNet-B4 IN-1K consistently eliminated; insufficient feature quality for 1010-class fine-grained task
- **Small-capacity models (≤22M params)**: EfficientNetV2-S IN-21K (22M) killed; insufficient capacity regardless of pretraining
- **Validation noise**: Only 3 images/class in val set makes hyperparameter selection unreliable; early stopping and ensemble more valuable than extensive sweeps

## Feature engineering playbook
- **Feature caching for frozen backbones**: Extract DINOv2/CLIP features once, train head many times — enables rapid HPO on large datasets
- **Resolution as a feature**: 448–518px input captures species-discriminative details; treat resolution as a first-class hyperparameter
- **Standard augmentation sufficient**: Mixup, CutMix, RandAugment + label smoothing; no exotic augmentation needed when backbone pretraining is strong
- **No metric learning tested yet**: ArcFace loss was designed but never benchmarked (open opportunity)

## Tuning insights
- **Frozen features >> full fine-tuning under time pressure**: DINOv2 frozen (0.133, 30 min) vs EVA-02 fine-tuned (0.144, 100+ min per run); speed enables more experiments
- **Layer-wise learning rate decay (LLRD)** used with EVA-02 fine-tuning; stabilizes ViT training on downstream tasks
- **MIM pretraining > supervised pretraining > CLIP** for fine-grained at low epoch budgets: EVA02 MIM beat larger supervised CNNs despite fewer epochs
- **Batch size constrained by resolution**: 448px on 2×A100-40GB limits effective batch; gradient accumulation not fully explored
- **5 tuner runs achievable** with frozen-backbone approach; full fine-tuning allows only 1–2 iterations

## Open questions
- **Explicit ensemble strategy**: Candidate 0 (0.108) beat best individual (0.133) — exact blending method unclear; systematic ensemble of DINOv2 + EVA-02 + ConvNeXt could push further
- **EVA-02 at 224px with extended training**: 448px timed out; does 224px + more epochs close the gap?
- **Test-time augmentation (TTA)**: 5-crop or multi-scale TTA never tested; may bridge gap to lower error cheaply
- **ArcFace / metric learning losses**: Designed but never run; could improve fine-grained discrimination on tail classes
- **ConvNeXt-Large CLIP backbone**: 2B image-text pretraining never successfully tested due to memory/time; gradient accumulation path unexplored
- **Coder timeout**: 1200s insufficient for full fine-tuning pipelines; 2400–3600s or split sanity-check/training phases recommended
