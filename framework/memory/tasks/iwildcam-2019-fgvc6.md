# Task Memory: iwildcam-2019-fgvc6

## Task profile
- Type: Multi-class image classification (14 animal species + background)
- Dataset: 179K training images from camera traps, 16.8K test images; RGB + night IR; extreme class imbalance (category 0 "background": 128K/179K images, 71.6%; category 22: 33 samples; ratio 3,893:1); zero overlap between train/test camera locations (124 train, 14 test); ~33% multi-frame sequences
- Metric: Macro F1-score (higher is better)
- Medal thresholds: Gold consistently achieved at 0.34–0.47+; median is below gold

## Best result
- Score: 0.47408 | Medal: Gold | Run: 20260211_150355
- Approach: ConvNeXt-Tiny with ImageNet pretraining, class-aware oversampling via WeightedRandomSampler, mixup regularization, and unweighted CE loss. Early stopping at ~6 epochs prevented rare-class overfitting. Achieved 0.593 val macro-F1.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_150355 | 0.47408 | Gold | ConvNeXt-Tiny + WeightedRandomSampler + mixup, early stop ~6 epochs |
| 20260318_165110 | 0.42022 | Gold | ConvNeXt-Tiny + focal loss + sqrt-frequency sampling + heavy augmentation (RandAugment/CutMix/MixUp) |
| 20260319_105744 | 0.47030 | Gold | EfficientNetV2-S progressive resizing 224→384, sqrt-weighted CE, location-based val split |
| 20260320_141552 | 0.45584 | Gold | ConvNeXt-Small (IN-22k) + focal loss + per-class threshold optimization + TTA; best single candidate hit 0.608 test |
| 20260320_141648 | 0.34449 | Gold | ConvNeXt-Tiny (IN-22k) + class-balanced focal loss (β=0.9999, γ=2.0) + layer-wise LR decay + progressive 224→288 |

## Proven strategies
- **ConvNeXt-Tiny/Small backbone**: Dominated across all 5 runs. Modern depthwise-conv + LayerNorm + larger kernels consistently outperformed ResNet-50, EfficientNet-B0/B2 for fine-grained species discrimination in camera trap images
- **ImageNet-22k pretraining > ImageNet-1k**: Run 4 showed 10–15% test F1 improvement with IN-22k ConvNeXt-Small vs IN-1k variants
- **Mixup/CutMix regularization**: Prevented overfitting on rare classes across multiple runs; soft labels integrate cleanly with CE or focal loss
- **Sqrt or effective-number-of-samples class weighting**: Raw inverse-frequency weighting (alpha ~3893 for class 22) destabilizes gradients. Sqrt-frequency or Cui et al. effective-number weighting (β≈0.999) consistently more stable
- **Early stopping / short training**: Best results at 6–8 epochs. Rare classes overfit sharply beyond that; 15-epoch runs degraded
- **Aggressive augmentation for domain shift**: RandomResizedCrop scale 0.5–1.0, ColorJitter, GaussianBlur, RandAugment essential because test locations are completely disjoint from training
- **TTA (horizontal flip)**: Cheap, consistent 1–2% macro-F1 gain across runs; negligible cost on 16.8K test images
- **Per-class threshold optimization**: Post-training greedy threshold search recovered 2–3% F1 in runs that implemented it; macro-F1 is misaligned with softmax argmax boundaries

## Failure patterns
- **WeightedRandomSampler with extreme oversampling**: Category 22 (33 images) repeated thousands of times causes severe instance overfitting. Run 4 showed regression from 0.922→0.905 val F1 when adding WeightedRandomSampler to ConvNeXt-Small. Loss-based weighting preferred over data-level rebalancing for classes <100 samples
- **Swin Transformers**: Killed in runs 4 and 5 — too slow (~15–30 min/epoch vs 7–12 min for ConvNeXt/EfficientNet). Within 300-min budget and 1200s tuner timeouts, only 1–2 full passes feasible
- **Two-stage training (frozen→unfrozen)**: EfficientNet-B2 two-stage in run 2 caused catastrophic forgetting of common-class features; stage-2 class-balanced sampling hurt calibration (0.248 vs 0.420)
- **DINOv2/frozen self-supervised features**: Killed in run 3; domain shift from web images to night IR camera trap images too large for frozen features
- **Sequence aggregation models**: Killed across runs 3, 4, 5 — multi-frame I/O overhead too expensive for 300-min budget; 67% of data is single-frame; only 0.4% label inconsistency across frames
- **MobileNetV3 distillation**: Run 5 showed F1 ceiling of 0.17–0.19; insufficient capacity
- **ResNet-50 baselines**: Consistently underperformed ConvNeXt by large margins; slower SGD convergence, weaker features

## Feature engineering playbook
- **No location embeddings**: Test locations are completely unseen; location-aware features require fallback mechanisms and showed no clear benefit
- **Sequence metadata (frame_num, seq_num_frames) unused**: Theoretically could help distinguish static backgrounds from animals, but multi-frame aggregation too slow in practice. Lightweight alternative: max-confidence frame selection at inference
- **Progressive resolution**: 224→288px gave +0.07 F1 without instability (run 5); 224→384px worked in run 3 but slower. 384px may be too slow for multiple tuner iterations
- **Object-detection cropping**: Mentioned in top-3% solutions but never tested. Tight crops could remove background domain shift and focus model on animals

## Tuning insights
- **Focal loss (γ=2.0) + per-class alpha**: More stable than plain weighted CE for extreme imbalance; focuses on hard examples
- **Layer-wise LR decay**: Backbone at 0.1x, head at 1.0x stabilized fine-tuning and prevented catastrophic forgetting (confirmed in runs 4, 5)
- **Batch size 16–32**: Memory-conscious sizing enabled full 179K training set on A100-40GB; larger batches hit OOM
- **AdamW > SGD**: Faster convergence within tight time budgets; all winning candidates used AdamW
- **EfficientNetV2-S speed advantage**: Fused MBConv blocks allowed more epochs within budget; won run 3 over slower ConvNeXt-Small. Within tight budgets, more iterations > bigger models
- **Standard DataLoader > WeightedRandomSampler** for ConvNeXt (run 4 ablation): Let loss weighting handle imbalance; sampler oversampling of tail classes caused overfitting
- **3 tuner iterations typical plateau**: Metrics stabilized after 3 tuner runs; further iterations showed diminishing returns

## Open questions
- **Per-class threshold optimization**: Implemented in some runs with 2–3% gains, but test distribution may differ from validation. Systematic exploration with cross-validation could unlock more
- **Ensemble of ConvNeXt seeds/augmentation variants**: Never systematically tested; run 3 hinted that candidate aggregation may have boosted final score (0.4703 final vs 0.5228 best single)
- **Object-detection-based cropping**: Top leaderboard solutions reportedly use bounding box pre-processing — tight animal crops could eliminate background domain shift. Never tested
- **Temperature scaling / calibration**: Could recover F1 on rare classes beyond threshold tuning; test distribution shift makes this uncertain
- **Larger ConvNeXt (Small→Base) or Swin with more compute**: ConvNeXt-Small showed promise in run 4 (0.608 test) but was bottlenecked by time. With extended budget, larger models may push further
- **Two-stage empty-vs-species pipeline**: Cascading errors not quantified. If Stage 1 false-negative rate is low, could simplify the species classifier's job significantly
- **Val-to-test gap**: Best val macro-F1 ranged 0.59–0.96 but test scores only 0.34–0.47, indicating significant domain shift. Closing this gap is the primary unsolved challenge
