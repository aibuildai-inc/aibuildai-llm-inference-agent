# Task Memory: histopathologic-cancer-detection

## Task profile
- Type: Binary classification (tumor vs. normal)
- Dataset: ~174K 96×96 RGB histopathology image patches; 60/40 class split (40.5% positive/tumor); H&E stained tissue
- Metric: AUC-ROC (higher is better)
- Medal thresholds: Bronze ≥ 0.9738; Gold achieved at 0.995+

## Best result
- Score: 0.9981 | Medal: Gold | Run: 20260308_193041
- Approach: EfficientNet-B3 with progressive resizing (72×72 → 96×96), EMA weight averaging, and 8-transform TTA. Ensemble of 4 diverse candidates (EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny, SE-ResNeXt50) produced the final score.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260210_073903 | 0.99509 | Gold | DenseNet121 + aggressive stain-aware augmentation |
| 20260226_184223 | 0.99644 | Gold | EfficientNet-B3 + layer-wise LR decay + TTA |
| 20260308_193041 | 0.99810 | Gold | Progressive resizing + EMA + 4-model ensemble |
| 20260309_194141 | 0.99739 | Gold | ConvNeXt-Nano best individual (0.99743); 4-model ensemble |

## Proven strategies
- **EfficientNet-B2/B3 is the workhorse**: Consistently top-performing single model across all runs (0.995–0.998 AUC). B3 (10.7M params) is the sweet spot; B0 (4M) underfits, B5+ overfits/too slow for 90-min budget
- **Test-time augmentation (8 transforms)**: Flips + 90/180/270° rotations; reliable +0.005–0.01 AUC lift at negligible cost. Non-negotiable for this task
- **EMA weight averaging**: Decay 0.999–0.9995 consistently improved generalization across all architectures
- **Strong color/geometric augmentation**: HueSaturationValue, RandomBrightnessContrast, color jitter, flips, rotations, ElasticTransform, GridDistortion via albumentations. Color augmentation more impactful than geometric for H&E stained images
- **Ensemble of 3–4 diverse architectures**: Combining EfficientNet + DenseNet + ResNeXt/ConvNeXt families consistently reaches higher scores than any single model; simple logit averaging works
- **ImageNet pretrained weights transfer well**: Despite natural→histopathology domain gap, pretrained weights consistently outperform random init; no domain-specific pretraining needed for 0.99+ AUC
- **pos_weight / inverse-frequency class weighting**: Adequate for the moderate 60/40 imbalance; focal loss not needed
- **Cosine annealing LR + linear warmup**: Stable across all candidates and runs

## Failure patterns
- **EfficientNet-B0 always killed/underfitting**: 4M params insufficient; expected AUC 0.96–0.98 with thin margin above bronze. Killed in runs 2 and 4
- **Vision Transformers fail at 96×96**: DeiT-Small produces only 36 spatial tokens with patch-16; positional embedding interpolation from 224→96 degrades quality; overfits on ~174K samples. Killed in runs 3 and flagged in run 1
- **EfficientNet catastrophic failure (run 1)**: 0.5721 AUC in one instance—likely LR sensitivity or augmentation incompatibility. Never reproduced when setup was correct (runs 2–4 all >0.996)
- **Complex multi-stage input designs**: ResNet34 two-stage input approach (run 4) killed for under-specification and high implementation risk
- **ConvNeXt-Tiny patchify stem mismatch**: Downsampling mismatch at 96×96 input flagged in run 2; ConvNeXt-Nano (run 4) worked better but still not clearly superior to EfficientNet
- **Mixup/CutMix adds complexity without clear benefit**: Killed in run 3 as redundant; EMA + label smoothing + pos_weight outperformed

## Feature engineering playbook
- **Augmentation pipeline (proven)**: albumentations with HorizontalFlip, VerticalFlip, RandomRotate90, ElasticTransform, GridDistortion, HueSaturationValue, RandomBrightnessContrast, CoarseDropout
- **Center-aware cropping**: Showed marginal benefit in run 4 (EfficientNet-B2: 0.99666) but not conclusively better than random crop
- **Progressive resizing**: 72×72 → 96×96 two-stage training improved convergence in run 3 (best overall score); adapts pretrained 224×224 features gradually
- **Stain-specific augmentation (HED color perturbation)**: Partially validated in run 4 (SE-ResNeXt50: 0.99577) but did NOT outperform simpler color jitter approaches; adds implementation risk

## Tuning insights
- **15–25 epochs sufficient**: EfficientNet-B3 fits ~20 epochs in 90-min budget at ~2–3 min/epoch; diminishing returns past AUC >0.99
- **Layer-wise LR decay**: Beneficial for pretrained backbones (lower LR for early layers)
- **EMA decay**: 0.999–0.9995 range works across architectures
- **Batch size 256 on A100**: DenseNet-121 dense concatenation did NOT cause OOM despite warnings
- **Model capacity sweet spot**: 7–15M params (EfficientNet-B2/B3, DenseNet-121, ConvNeXt-Nano) optimal; >25M params (SE-ResNeXt50) risks overfitting without clear gain
- **TTA > hyperparameter micro-tuning**: Once val AUC >0.99, 8-fold TTA adds more than further epoch/LR tweaking

## Open questions
- **Stain normalization (Macenko/Reinhard/SPCN)** as preprocessing: Literature suggests 0.5–1% AUC gain; never tested across 4 runs. Could it replace heavy color augmentation?
- **Learned ensemble weights**: All runs used equal-weight logit averaging; inverse-AUC or stacking might push past 0.998
- **Knowledge distillation**: Ensemble → single model to maintain 0.998 AUC with faster inference
- **EfficientNet failure mode**: Run 1 catastrophic 0.5721 AUC never fully explained; LR sensitivity suspected but not debugged
- **HED perturbation numerical stability**: Flagged as risk in run 4; needs controlled comparison vs. ColorJitter-only
- **Cross-validation error bars**: All runs report single AUC; stratified k-fold would quantify candidate stability
- **Optimal progressive resizing schedule**: Only one configuration tested (72→96 in 2 stages); more stages or different resolution paths unexplored
