# Pattern: agriculture-environmental

## When this applies
- Plant/leaf disease image classification (single-label or multi-label)
- Small-to-medium datasets (1.6K–18.7K images), often class-imbalanced
- High-resolution source images requiring downsampling to 224–384px
- Fine-grained visual distinctions (disease lesions, textures)

## Proven patterns
- **ImageNet-22k/CLIP pretraining beats IN-1k by 1–2%**: ConvNeXt-Small IN-22k scored 0.99969 on plant-pathology-2020; ConvNeXt-Base CLIP was top backbone on cassava (85.15%). IN-22k Swin-Tiny hit 0.99914.
- **MixUp (α=0.2–0.4) + CutMix + label smoothing (0.1)** appear in nearly every successful run across all 3 tasks.
- **EMA weight averaging (decay 0.9995–0.9998)**: critical in plant-pathology-2020 top runs; helps stability on cassava.
- **TTA (hflip baseline, multi-scale bonus)**: 0.5–1% AUC lift on plant-pathology-2020; used in both plant-pathology-2021 runs.
- **Stratified K-fold CV (5-fold preferred)**: mandatory for <2K samples; gave 0.99969 on plant-pathology-2020.
- **Per-class threshold optimization for multi-label** (0.01 step, 0.1–0.9 range): single highest-leverage technique on plant-pathology-2021 (0.9709 with ResNet50).
- **Two-phase training** (frozen backbone warmup → full unfreeze with LLRD 0.7–0.75×): used in top runs of all 3 tasks.

## Common pitfalls
- **Frozen-feature + classical ML (LightGBM/XGBoost)**: rejected 3× on plant-pathology-2020; ceiling 0.94–0.97.
- **Pure ViT/DeiT on small datasets**: overfits; Swin/ConvNeXt with hierarchical inductive bias work instead.
- **EfficientNet-B4 underperforms**: 0.7849→0.8645 on plant-pathology-2021; slow convergence.
- **Bi-tempered logistic loss**: failed on cassava and stuck at 0.9618 on plant-pathology-2020.
- **Multi-candidate parallel training on 2-GPU setups**: causes OOM (cassava run 2 burned 11 calls).
- **Tight coder timeouts (1200s)**: insufficient for ConvNeXt/ViT (need 1800–2400s) on cassava.
- **Resolution >384px without memory tricks**: OOM at 448px on plant-pathology-2021.
- **Designing 5+ candidates without executing any**: plant-pathology-2020 run 2 produced 0 metrics.

## Model selection guide
- Multi-label, <2K samples → ConvNeXt-Small IN-22k or Swin-Tiny IN-22k
- Multi-label, 10K+ samples → ResNet50 with aggressive tuning beats untuned larger models
- Single-label, small dataset, severe imbalance → ConvNeXt-Base CLIP/IN-22k + Mixup/CutMix
- Avoid: EfficientNet-B4, pure ViT, bi-tempered loss, frozen+classical ML

## Feature engineering patterns
- Resolution sweet spot: 384px (matches Swin pretraining, fits A100 memory at batch 32)
- Augmentation tiers: hflip + shift/scale/rotate + color jitter (always); CLAHE/elastic/CoarseDropout (helpful); MixUp+CutMix (essential for small data)
- GeM pooling > GAP for fine-grained disease features (plant-pathology-2020)
- Label-combination stratified splits preserve multi-label co-occurrence

## Tuning patterns
- AdamW LR=1e-4 backbone, 1e-3 head; cosine annealing + warmup
- Layer-wise LR decay 0.7–0.75× per stage in top runs
- 15 epochs sufficient with OneCycleLR or cosine; early stopping patience=5
- Class imbalance recipe: inverse-frequency weights + focal loss γ=2 + Mixup + label smoothing 0.1
- Batch size 32 at 384px (not 64—gradient quality + OOM avoidance)
- Tuner diminishing returns after 2nd run; spend remaining budget on ensemble diversity
