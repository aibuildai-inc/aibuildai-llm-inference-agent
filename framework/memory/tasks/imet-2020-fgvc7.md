# Task Memory: imet-2020-fgvc7

## Task profile
- Type: Multi-label classification (extreme, 3,474 classes)
- Dataset: ~120K training images, ~21K test images; museum artwork with fine-grained attributes (medium, culture, period, technique); extreme class imbalance — 0.13% avg positive rate, avg 4.4 labels/image, 1,435 labels with <10 samples, 39 zero-shot labels
- Metric: Micro-averaged F1, higher is better
- Medal thresholds: Bronze ~0.649, Silver ~0.653, Gold ~0.696

## Best result
- Score: 0.6765 | Medal: None (above median, below bronze cutoff this run) | Run: 20260422_111701
- Approach: EVA02-Base CLIP-MIM backbone + ASL (γ_neg=4, γ_pos=0, clip=0.05) + SWA over final 25% of training, 4-epoch cosine schedule with bfloat16 AMP, per-class threshold search (0.01 grid, min-positives regularization). Reviser pivoted from Swin-V2 Large (plateaued 0.593) to smaller CLIP-pretrained EVA02 with properly-sized schedule.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_152026 | ungraded | — | All candidates timed out; Swin-Base-384 reached 0.6312 intermediate |
| 20260214_125530 | 0.6382 | none | ConvNeXt-Small IN-22K + ASL + TTA; first completed submission |
| 20260215_105641 | 0.5351 | none | EfficientNet-B3 384px + focal loss; regression from wrong alpha & weaker backbone |
| 20260217_114416 | 0.6441 | none | ConvNeXt-Base CLIP + ASL + TTA; CLIP pretrain breakthrough |
| 20260221_191916 | 0.6382 | none | ConvNeXt-Base IN-21K + ASL + SWA + progressive 224→288px |
| 20260304_121320 | 0.6531 | Silver | Swin-Base 384px IN-22K + ASL + per-label thresholds + multi-scale TTA |
| 20260328_114500 | 0.6600 | Silver | ConvNeXt-Base@384 + Swin-Base@384 ensemble |
| 20260330_134034 | 0.6650 | Silver | EVA02-Base (CLIP-guided MIM) + ASL + fine threshold grid (0.01 steps) |
| 20260422_111701 | 0.6765 | None | EVA02-Base CLIP-MIM + ASL + SWA + bf16 AMP + 4-epoch cosine; after Swin-V2 Large (0.593) pivot |

## Proven strategies
- **Large backbones (86–90M) required**: ConvNeXt-Base, Swin-Base, EVA02-Base reach 0.63+; EfficientNet-B3/B5 cap ≤0.60. But **scale isn't everything**: Swin-V2 Large (197M) plateaued at 0.593 with 12-epoch plan (under-trained); EVA02-Base beat it by 0.084 F1
- **CLIP/vision-language pretraining >> IN-22K >> IN-1K**: EVA02 (CLIP-guided MIM) and ConvNeXt-Base CLIP lead by 2–4% F1
- **ASL is non-negotiable**: γ_neg=4, γ_pos=0, clip=0.05 proven across 7+ runs; addresses 99.87% negative rate
- **ASL + bfloat16 AMP + gradient clipping** is the stable recipe; fp32/fp16 + ASL triggers NaN
- **384px near-mandatory** (native ~375px); progressive resize (168→224 or 224→384) saves early-epoch compute
- **Per-class threshold search (0.01 grid, min-positives regularization)** lifts F1 by 0.05+; global threshold alone yields <0.17 F1 with 3474 classes
- **SWA over final 25% of training** provides modest gains at no inference cost
- **Layer-wise LR decay** stabilizes transformer fine-tuning
- **Two-layer MLP head** (→1024→3474) with dropout outperforms single linear

## Failure patterns
- **Timeouts dominate**: 4/6 candidates timed out at 1200s in the latest run; heavy augmentation + large model inference exceed budgets under GPU contention
- **Over-sized cosine schedules underperform**: Swin-V2 Large with 12 epochs completed only ~1.4 epochs and regressed past step 15000; shorter 4-epoch EVA02 schedule won. Size schedule to realistic budget
- **Resume-based training breaks LR schedules**: Candidate 4 (CLIP ViT-L) stuck at 0.063 from resume/stage1-overwrite; multiple historical runs lost 2–4% F1. Prefer fresh single-run training
- **Warm restart with optimizer reset**: catastrophic regression (0.758→0.659 historically); preserve Adam momentum
- **AMP + focal/pos_weight → NaN** on some backbones (EffNet-B3, EffNetV2); bf16 + ASL is the safe combo
- **OOM under cluster contention**: CLIP ViT-L stage-1 validation OOM despite 40GB card; provision isolated GPU for large models
- **Small/mid models waste budget**: EffNet-B3/B4/B5, EVA02-Small, ResNet50d consistently eliminated
- **Depth > breadth on this task**: Exploring 6 diverse candidates cost 4 timeouts; extending the one improving candidate (7 tuner extensions) added 0.107 F1

## Feature engineering playbook
- **Augmentation**: ColorJitter, ShiftScaleRotate, CoarseDropout, horizontal flip (safe on artwork)
- **Multi-scale TTA** (0.9×/1.0×/1.1×) or h-flip: typically 1–2% F1, but can interact badly with per-class thresholds calibrated sans-TTA
- **Progressive resizing**: 168/224 → 384px
- **Two-layer MLP head** (768→1024→3474) with dropout for 3474-class output

## Tuning insights
- **ASL hyperparams**: γ_neg=4, γ_pos=0, clip=0.05 (γ_neg=2 slightly weaker)
- **LR**: 1e-4 to 2e-5 for fine-tuning; 5e-5 caused stagnation in one run
- **Epochs**: 4 epochs at 384px with EVA02-Base now proven best for budget; 10–12 epochs only if backbone actually completes them (Swin-V2 Large didn't)
- **Batch**: 16–24 at 384px on single GPU; bf16 helps fit
- **Threshold search**: 0.01-step grid over 0.01–0.95; only labels with ≥5 val positives
- **Cosine schedule**: single continuous run, sized to realistic wall-clock budget
- **Depth-over-breadth**: extend the monotonically-improving candidate; don't re-spawn diverse candidates

## Open questions
- **Gold gap (0.6765→0.696)**: ~2% F1 remains; full ensemble of EVA02-Base + Swin-Base + ConvNeXt-Base CLIP still unimplemented
- **EVA02-Large / CLIP ViT-L/14**: never successfully trained (OOM under contention); likely gold path with isolated 40GB+ GPU and 336px progressive resize
- **Self-distillation** from best checkpoint as soft-label regularizer; BEiTv2 candidate planned, not run
- **Rare-class oversampling** (repeat-factor / LVIS-style sampling) for 39 zero-shot + 1,435 <10-sample classes untested — potential 0.5–1% gains
- **Label co-occurrence post-processing** (boosting correlated culture+period labels) untested
- **Robust ASL variants** and label smoothing 0.02–0.05 in ASL untested on tail classes
- **TTA × per-class threshold interaction**: re-calibrate thresholds on TTA predictions to resolve historical inconsistency
