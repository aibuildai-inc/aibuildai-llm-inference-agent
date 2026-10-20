# Task Memory: plant-pathology-2021-fgvc8

## Task profile
- Type: Multi-label image classification (6 disease classes)
- Dataset: ~14.9K training images, 4000×2672 original resolution, plant leaf photos; class imbalance (scab 30.9%, healthy 24.6%, ~4.6× ratio between most/least frequent); multi-label co-occurrence rare (~7.3% of samples)
- Metric: micro-F1 (higher is better)
- Medal thresholds: unknown

## Best result
- Score: 0.9709 | Medal: ungraded | Run: 20260212_222113
- Approach: ResNet50 with two-phase training (2 frozen epochs then full fine-tune), per-class threshold optimization (0.01 step over 0.1–0.9 range), BCEWithLogitsLoss, TTA with horizontal flip, batch size 32, 15 epochs at 384px resolution.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_175312 | incomplete (best 1-epoch val: 0.8878 Swin-Base) | ungraded | Baseline exploration; Swin/ConvNeXt/EfficientNet candidates; run terminated mid-training |
| 20260212_222113 | 0.9709 | ungraded | ResNet50 + tuner (epochs 12→15, batch 64→32, per-class threshold search) |

## Proven strategies
- **Per-class threshold optimization** is the single highest-leverage technique; fine-grained search (0.01 step, range 0.1–0.9) on validation set consistently outperforms a global threshold. ResNet50 + tuning beat larger untuned architectures.
- **Two-phase training** (freeze backbone 2 epochs at LR=1e-3, then unfreeze full model at backbone LR=1e-4): stabilizes convergence and prevents early-layer corruption on domain-specific agricultural images.
- **TTA with horizontal flip**: free accuracy boost, implemented in both runs.
- **Label-combination stratified splitting**: preserving multi-label co-occurrence distribution in train/val folds is essential given the 7.3% multi-label rate.
- **Mixed precision (AMP)**: required for memory efficiency at 384px resolution.
- **Batch size 32** (not 64): better gradient quality and avoids OOM at 384px.

## Failure patterns
- **EfficientNet-B4 consistently underperforms**: 0.7849 (1-epoch, run 1) and 0.8645 (8-epoch, run 2)—slower convergence, likely underfitting at this dataset scale.
- **ConvNeXt never fully evaluated**: despite strong 1-epoch signal (0.8808), the heavier pipeline (LLRD, EMA, Mixup/CutMix) was never tuned or compared end-to-end in run 2.
- **IMAGE_SIZE 448 causes OOM** with batch_size ≥ 32 on A100; 384 is the practical ceiling without further memory optimization.
- **Run 1 terminated mid-aggregation**: no final scores produced; judge step never executed.
- **Multi-label prediction rate remains low** (~7% output vs 7.3% ground truth)—threshold calibration may be hitting a floor imposed by label imbalance.

## Feature engineering playbook
- **Input resolution**: 384px is the sweet spot (matches Swin pretrained resolution, fits in GPU memory). Original 4000×2672 → 384 loses fine-grained disease texture; higher resolution (512+) worth exploring with gradient accumulation.
- **Augmentation**: Mixup, CutMix, RandAugment used in ConvNeXt candidate (run 1); standard augmentation + horizontal flip sufficient for ResNet50 to reach 0.9709. Heavy augmentation compensates for overfitting on 14.9K images.
- **Loss function**: BCEWithLogitsLoss without pos_weight works well when combined with augmentation + threshold search. Focal loss and label smoothing both showed promise in run 1 but were not carried forward. Soft F1 loss remains untested.

## Tuning insights
- **Epochs**: 15 > 12 for ResNet50; earlier stopping likely leaves performance on the table.
- **Batch size**: 32 is optimal; 64 causes OOM and worse gradient quality at 384px.
- **Learning rate**: differential LR (head 1e-3, backbone 1e-4) with two-phase schedule outperforms flat LR.
- **Architecture ROI**: ResNet50 (25M params) + proper tuning beat EfficientNet-B4 and matched/exceeded untuned Swin/ConvNeXt. Tuning budget matters more than model size on this task.
- **Threshold search range**: expanding to 0.1–0.9 with 0.01 step captures per-class optima that narrower ranges miss.

## Open questions
- **Swin-Base-384 vs ResNet50 after full tuning**: Swin showed 0.8878 at 1 epoch (vs ResNet50's eventual 0.9709 after 15 epochs + tuning). Would Swin + same tuning pipeline exceed 0.97?
- **ConvNeXt potential**: strong 1-epoch signal (0.8808) but never tuned end-to-end. Worth a dedicated run.
- **Higher resolution** (512px) with gradient accumulation or smaller batch: could recover disease texture detail lost in 384 resize.
- **Multi-crop / multi-scale inference**: untested; may help with fine-grained lesion patterns.
- **Co-occurrence modeling**: explicit multi-task auxiliary losses for correlated labels (e.g., complex + frog_eye_leaf_spot) never explored.
- **Soft F1 loss**: designed in run 2's Candidate 3 but never trained; could eliminate need for post-hoc threshold search.
- **Multi-label prediction gap**: output multi-label rate (~7%) matches ground truth (7.3%) but may indicate threshold calibration floor rather than true model confidence—needs investigation.
