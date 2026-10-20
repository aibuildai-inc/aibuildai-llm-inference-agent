# Task Memory: iwildcam-2020-fgvc7

## Task profile
- Type: Multi-class image classification (267 wildlife species + empty class)
- Dataset: Camera trap images from iWildCam 2020 FGVC7; 225 train locations, 100 test locations with **zero geographic overlap**; ~33% "empty" class; 55 classes with <10 images; 82 categories with zero training samples; temporal sequence metadata (seq_id) available; images are variable-resolution JPEGs
- Metric: Top-1 accuracy (higher is better)
- Medal thresholds: Gold ≈ 0.778+, Silver ≈ 0.70–0.78, Bronze < 0.70 (approximate from runs)

## Best result
- Score: 0.84217 | Medal: Gold | Run: 20260211_125817
- Approach: Swin-Base transformer trained on MegaDetector crops at high resolution with sqrt inverse-frequency class weighting. Three tuner rounds refined the model from 0.710→0.717 val accuracy; strong generalization to disjoint test locations yielded 0.84217 test accuracy.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_125817 | 0.84217 | Gold | Swin-Base + MegaDetector crops + sqrt inverse-freq weighting |
| 20260226_161333 | 0.77976 | Gold | EfficientNet-B5 two-stage (empty/species) + label smoothing + mixup |
| 20260323_103754 | 0.79378 | Gold | EVA-02 ViT (CLIP pretrained) + sequence aggregation + ensemble |
| 20260318_165217 | 0.735 | Silver | ConvNeXt-Base (ImageNet-21K) + focal loss + checkpoint resume |
| 20260319_105754 | 0.70729 | Silver | ConvNeXt-Tiny + TTA + sequence aggregation |
| 20260321_100654 | 0.681 | Bronze | EVA-02-Small ViT + sequence aggregation + class-freq weighting |

## Proven strategies
- **MegaDetector crops are non-negotiable**: Every successful run used MegaDetector bounding-box crops (10% padding, fallback to full image if low confidence). Full-image models consistently failed (~0.33–0.60 accuracy) due to location-specific background confounders. Skipping crops costs 30+ accuracy points.
- **Transformers ≥ CNNs for fine-grained species**: Swin-Base (0.84217), EVA-02 (0.821 val / 0.79378 test), and EVA-02-Small (0.841 val) consistently outperformed EfficientNet variants (B3: 0.676–0.775, B5: 0.762). Global attention captures discriminative morphological features better than convolutions.
- **Sequence-level aggregation is free accuracy**: Averaging logits/softmax within seq_id groups exploits camera trap burst structure for +0.4–2% at inference with zero training cost. Confirmed across runs 20260319, 20260321, 20260323.
- **Weighted random sampling (1/√count) > focal loss for accuracy metric**: Simple sqrt inverse-frequency weighting was more stable and effective than focal loss (which caused NaN issues in run 20260226). Label smoothing (0.1) and mixup (0.2) provide complementary regularization.
- **ImageNet-21K or CLIP/masked-image-modeling pretraining**: 21K weights consistently outperform 1K-only; EVA-02 CLIP and masked-image-modeling pretraining transferred well to camera traps (run 20260321, 20260323).
- **Layer-wise LR decay for transformer fine-tuning**: Exponential decay (~0.75×) across stages stabilized convergence for Swin and ConvNeXt architectures.

## Failure patterns
- **Full-image models (no crops) always fail**: Candidate 2 in runs 20260319, 20260323 both killed/failed; background habitat varies by location and dominates predictions without cropping.
- **Focal loss instability**: NaN issues in early epochs under extreme imbalance (run 20260226 candidate 2: 0.2676 accuracy). Hyperparameter brittleness makes it risky vs. weighted sampling.
- **ConvNeXt results inconsistent**: Won in run 20260318 (0.735) but plateaued at 0.357 in run 20260321 and was outperformed by transformers in most runs. Not reliably superior despite some benchmark claims.
- **1200s timeout kills complex candidates**: ConvNeXt-Base, Swin + complex preprocessing, EfficientNet-B5 two-phase, and domain-adversarial approaches repeatedly exceeded wall-clock limits. Simpler pipelines with smaller models (ConvNeXt-Tiny, EVA-02-Small) iterate faster.
- **Ensemble of undertrained models < single strong model**: Dual-model ensembles (runs 20260318, 20260319, 20260323) consistently underperformed because time budget was split, leaving each model undertrained. Build strong singles first.
- **Domain-adversarial training adds complexity without payoff**: MegaDetector crops already mitigate location shift; gradient reversal layers added instability risk for marginal gains (rejected in runs 20260319, 20260321).
- **Two-stage cascade fragility**: Empty/non-empty threshold tuning is sensitive; 21% of animals have low MegaDetector confidence, making cascade errors unrecoverable (run 20260319 candidate 5).

## Feature engineering playbook
- **MegaDetector crop pipeline**: Crop to bounding box with 10% padding; fall back to full image when detection confidence is low. This is the single most impactful preprocessing step.
- **Crop resolution**: 288×288 confirmed effective for ConvNeXt-Base; higher resolution benefits transformers (Swin-Base). Optimal resolution not fully ablated—320×320 may help.
- **Empty class heuristic**: Use MegaDetector confidence threshold to assign "empty" label directly. Full validation set required for threshold tuning (subsampling introduces noise, per run 20260226).
- **Sequence metadata**: Group images by seq_id at inference; average logits across frames. No training modification needed.
- **Test-time augmentation**: Multiple cropped views + horizontal flips at inference; confirmed benefit in runs 20260319, 20260323. Diminishing returns beyond ~4–8 views not yet measured.

## Tuning insights
- **Tuner shows diminishing returns after 3 rounds**: Run 20260211 saw 0.710→0.717 over 3 rounds (+0.7%). Budget 3–4 tuner runs max per candidate.
- **Extended training helps**: EVA-02-Small still improving at epoch 33 (run 20260321); ConvNeXt-Base jumped 0.770→0.810 after checkpoint resume (run 20260318). Allocate max training time to the best candidate.
- **EMA decay = 0.9998**: Stabilizes eval but diverges if loaded incorrectly. Always validate EMA ≠ main model before inference.
- **Batch size / BN instability**: EfficientNet at high resolution reduces batch size and destabilizes batch norm; prefer smaller-resolution EfficientNet or use transformers with LayerNorm.
- **Class weighting**: sqrt inverse-frequency is the reliable default. No ablation on α ∈ [0.3, 0.7] was run.
- **Parallel GPU training**: Running top 2 candidates on separate GPUs in parallel is essential under 1200s timeout pressure.

## Open questions
- Would Swin-Large, ViT-Large, or EVA-02-Large push past 0.84217? Larger transformers were never fully trained due to timeout constraints.
- Optimal MegaDetector confidence threshold—no systematic sweep was ever run across any experiment.
- TTA view count optimization: 8–16 views vs. 4, and interaction with sequence aggregation, remains untested.
- Could supervised contrastive loss (SupCon) improve fine-grained separation for the 55 classes with <10 samples?
- Temporal sequence models (LSTM on burst embeddings) or learned sequence gates vs. simple averaging—likely overkill but untested.
- Crop resolution sweep (256/288/320/384) across architectures to find optimal size-accuracy tradeoff.
- Explicit empty-class routing (bypass classifier when MegaDetector conf < threshold) was planned but never tuned.
- Stacking predictions from checkpoints at different epochs (temporal ensembling) as an alternative to model ensembles.
