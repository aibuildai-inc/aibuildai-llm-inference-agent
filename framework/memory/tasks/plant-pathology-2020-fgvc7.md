# Task Memory: plant-pathology-2020-fgvc7

## Task profile
- Type: Multi-label classification (4 classes: healthy, rust, scab, multiple_diseases)
- Dataset: 1,638 training images, high-res leaf photos (2048×1365 native, typically resized to 224–512px)
- Metric: Mean column-wise ROC AUC (higher is better)
- Medal thresholds: bronze ≥ 0.97361; gold achieved by all successful runs

## Best result
- Score: **0.99969** | Medal: gold | Run: 20260320_195112
- Approach: ConvNeXt-Small with ImageNet-22k pretraining, focal loss (γ=2) with per-class weights, layer-wise LR decay (0.7×/stage) + cosine annealing, mixup/cutmix + label smoothing 0.1, EMA (decay=0.9998), 5-fold CV + TTA (hflip + multi-scale). Raw fold AUC 0.9893 → 0.99969 after TTA ensemble.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260208_034103 | N/A | ungraded | Setup failure — mle-bench pip install unavailable; 0 candidates designed |
| 20260208_035526 | N/A | ungraded | 5 candidates designed (EffNetV2-S, ViT-B/16, ResNet50+SSL, Ensemble, ConvNeXt-Tiny); judge skipped training execution — metrics empty |
| 20260210_042635 | 0.99570 | gold | EfficientNet-B2/B3 ensemble, weighted BCE, TTA (hflip) |
| 20260319_214628 | 0.99914 | gold | Swin-Tiny IN-22k + heavy augmentation + Mixup α=0.3 + LLRD + 5-variant TTA |
| 20260320_195112 | **0.99969** | gold | ConvNeXt-Small IN-22k, focal loss + per-class weights, EMA, multi-scale TTA |
| 20260320_201445 | 0.99544 | gold | ResNeXt-50 + MixUp/CutMix + EMA + OneCycleLR, 2-model ensemble |

## Proven strategies
- **ImageNet-22k pretraining is mandatory**: IN-22k consistently outperformed IN-1k. Top-2 scores (0.99969, 0.99914) both used IN-22k backbones. Critical for 1,638-sample fine-grained classification.
- **MixUp/CutMix + label smoothing (0.1)**: Used in all successful runs; prevents overfitting. Mixup α in 0.2–0.4 range works well.
- **EMA weight averaging (decay 0.9995–0.9998)**: Proven across 3 of 4 successful runs. Must verify EMA active at model instantiation.
- **Test-time augmentation**: Consistent 0.5–1%+ AUC lift. Horizontal flip baseline; multi-scale (224/256/288) and 5-variant TTA yield further gains.
- **5-fold stratified CV**: Mandatory for <2K samples. Candidate variance ±0.005 AUC.
- **Layer-wise LR decay (0.7–0.75× per stage)**: Used in top-2 runs. Stabilizes deep pretrained fine-tuning.
- **Aggressive augmentation pipeline**: RandomResizedCrop, elastic transforms, CLAHE, CoarseDropout, color jitter, shift/scale/rotate.
- **Ensemble diversity via different augmentation policies or architectures**: Even 2-model ensembles boost robustness.
- **Class weighting for multiple_diseases (5.2%, ~85 samples)**: Inverse-frequency weighted BCE or focal loss required.

## Failure patterns
- **Environment/setup failures block everything**: Run 20260208_034103 scored N/A because mle-bench pip install failed — 0 candidates reached design. Verify dependencies before Phase 2.
- **Judge/coder handoff to training step can silently skip**: Run 20260208_035526 generated 5 valid candidates but judge validated code without executing training; 0 metrics collected, 0 tuner runs. Workflow must enforce training execution before manager decisions.
- **Frozen feature extraction + classical ML (LightGBM/XGBoost)**: Rejected in 3 of 4 design phases. Hard ceiling 0.94–0.97, below bronze.
- **Pure ViT/DeiT on <2K samples**: Killed for overfitting risk. Exception: Swin (hierarchical inductive bias) succeeded.
- **Exotic loss functions (bi-tempered)**: Stuck at 0.9618 AUC due to hyperparameter sensitivity. Standard CE/focal is safer.
- **EMA implementation bugs**: Disabled-at-construction EMA cost run 2 three tuner rounds stuck at 0.9877.
- **Training budget overruns**: High-res (380–512px), large ensembles, and ViTs hit the ~20-min wall-clock limit. EfficientNet-B4 progressive resizing killed for 35–50 min estimate.
- **Focal loss alone is inadequate for class imbalance**: Must pair with mixup/cutmix augmentation.
- **Over-diverse candidate slate under tight budget**: Designing 5 architectures without executing any wastes agent calls; prefer 2–3 battle-tested designs when budget is limited.

## Feature engineering playbook
- **Resolution**: 224–384px is the sweet spot. No observed benefit from larger resolutions. Multi-scale TTA (224/256/288) captures scale variation cheaply.
- **Augmentation tiers**: Tier 1 (always): hflip, shift/scale/rotate, color jitter, normalize. Tier 2: elastic, CLAHE, CoarseDropout. Tier 3: MixUp α=0.2–0.4, CutMix.
- **Per-branch augmentation diversity**: Different augmentation policies per model (geometric vs. color) reduces correlation in ensembles.
- **GeM pooling > global average pooling**: ResNeXt + GeM captured fine-grained disease features better than GAP.
- **Two-phase fine-tuning**: Head-only warmup then full unfreeze fits the ~20-min budget better than end-to-end from step 1.

## Tuning insights
- **Architecture ranking**: ConvNeXt-Small IN-22k (0.99969) ≥ Swin-Tiny IN-22k (0.99914) > EfficientNet-B2/B3 ensemble (0.99570) > ResNeXt-50 (0.99544) > SE-ResNeXt-50 (~0.968) > EfficientNet-B3 single (~0.987)
- **Learning rate**: Cosine annealing from ~3e-4 with linear warmup. Layer-wise decay 0.7–0.75× per stage.
- **Epochs**: 15–50 with early stopping (patience=5). OneCycleLR converges in 15 epochs.
- **Class imbalance recipe**: Inverse-frequency weights (multiple_diseases ≈4–6×) + focal loss γ=2 + mixup + label smoothing 0.1.
- **Dropout**: 0.3 works combined with mixup and label smoothing for layered regularization.
- **Tuner diminishing returns**: Second tuner run typically <1% gain. Third+ rarely help. Spend budget on ensemble diversity.
- **EMA decay**: 0.9995–0.9998 range. Higher (0.9998) in best run.
- **Self-supervised pretraining (SSL) on ResNet50**: Designed but never executed; ImageNet pretraining remains default.

## Open questions
- **Workflow robustness**: Why did the 20260208_035526 judge step validate code but skip training execution? Need a guard that blocks manager's "run_tuner" decision until ≥1 candidate has a measured AUC.
- **Setup robustness**: Can setup step include a fallback data-loading path (direct Kaggle API or cached local paths) when mle-bench install fails?
- **Progressive resizing**: Never properly tested. Could 224→384→512 phased training help if budget permits?
- **Full-resolution training**: Native 2048×1365 crops never tested; potentially valuable for fine-grained texture (rust spots, scab edges).
- **3+ model ensemble ceiling**: Best (0.99969) used single ConvNeXt + TTA. Stacking Swin + ConvNeXt + EfficientNet might push past 0.9999.
- **Weighted ensemble by per-fold AUC**: Simple averaging used in all runs. AUC-weighted voting could recover 0.1–0.2%.
- **ConvNeXt-Tiny vs EfficientNetV2-S**: Designed in run 20260208_035526 but never trained; head-to-head comparison open.
