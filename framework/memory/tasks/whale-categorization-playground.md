# Task Memory: whale-categorization-playground

## Task profile
- Type: Fine-grained image re-identification / metric learning (MAP@5 ranking)
- Dataset: ~3000+ whale ID classes, extreme long-tail (76.6% singleton classes, median=1 image/class), variable-resolution fluke images, `new_whale` unseen-class handling required
- Metric: MAP@5 (higher is better)
- Medal thresholds: bronze ≈ 0.40515; silver observed at 0.52349; gold observed at 0.5792

## Best result
- Score: 0.5792 | Medal: gold | Run: 20260221_051452
- Approach: Triplet loss on ResNet50 + heavily-weighted auxiliary classification (weight 2.0–2.5), AdamW, 448px input, BS=32, progressive fine-tuning across 6 attempts, horizontal-flip TTA adding ~3%. Peak candidate internal MAP@5 = 0.624.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260218_025446 | 0.11342 | none | EfficientNet-B4 + LMCLoss; internal 0.24 but aggregator lost the best candidate |
| 20260220_172147 | 0.58165* | n/a | EfficientNet-B3 + ArcFace metric learning; aggregator picked wrong candidate |
| 20260221_051452 | 0.5792 | gold | ResNet50 + triplet + aux-CE (w=2.0–2.5), 448px, flip TTA, progressive fine-tune |
| 20260301_081436 | 0.52349 | silver | EfficientNetV2S + CosFace with progressive margin 0.25→0.55 |

*Run 20260220 reported candidate score; final submission likely lower due to aggregator fault.

## Proven strategies
- **Metric learning >> plain classification** on 3000+ class long-tail; triplet+aux-CE and CosFace both reach silver/gold
- **Triplet + auxiliary classification (weight 2.0–2.5)** is the single best recipe observed (0.624 internal, 0.5792 public)
- **Progressive margin schedule for CosFace (0.25 → 0.55)** prevents singleton collapse while gaining discriminativity
- **Progressive / iterative fine-tuning** with LR resets unlocks 20–30% over single-pass training (Candidate 2: 0.325 → 0.624)
- **Horizontal-flip TTA**: consistent ~+3% on ResNet50 metric-learning models
- **Higher resolution over larger batch**: 448px @ BS=32 beats 256px @ BS=64 on fluke fine detail
- **Singleton-aware CV split** (group by whale ID) mandatory to avoid leakage

## Failure patterns
- **Aggregator / submission pipeline is the #1 score killer**: 20260218 lost 0.13 MAP@5 (0.24→0.11); 20260220 submitted a weaker candidate than the best one; multiple torch-checkpoint / pandas / embedding-cache errors across runs
- **ArcFace is unstable on this distribution**: collapses to 0.30–0.38 even after scale/margin tuning (s=30 m=0.5 and s=64 m=0.3 both failed); CosFace is the stable replacement
- **MixUp/CutMix fundamentally incompatible with metric-learning losses** (CosFace, triplet) — must be disabled
- **ViT backbones (ViT-Base, ViT-Small, ArcFace-ViT) underperform CNNs** on this dataset, likely due to limited per-class data; TTA can even hurt ViT-Small
- **Learned `new_whale` bias is brittle**: candidates drift to always-predict or never-predict; no run successfully learned an explicit similarity threshold
- **WeightedRandomSampler destabilizes metric-learning training** (observed in 20260301 candidate 1)
- **CLIP + hybrid CE hybrids**: repeated failure (0.26), plus data-split bugs

## Feature engineering playbook
- Input resolution: 448px for ResNet50 metric learning; 256px acceptable for CosFace when margin warmup needed
- Augmentation: horizontal flip + flip-TTA only; disable MixUp/CutMix under metric losses
- Sampler: plain shuffled loader beats WeightedRandomSampler for metric learning stability
- CV: group-aware split honoring singleton whale IDs; never naive k-fold
- TTA: horizontal-flip averaging on CNN embeddings; skip for ViT

## Tuning insights
- Loss dominates backbone choice by an order of magnitude; pick the loss first
- Working loss ranking on this task: **Triplet+aux-CE ≈ CosFace (progressive margin) > LMCLoss > ArcFace > CE+center-loss**
- Working backbone ranking: **ResNet50 ≥ EfficientNetV2S ≥ EfficientNet-B3/B4 > ConvNeXt > ViT**
- CosFace: start margin 0.25, ramp to 0.55; avoid abrupt high margins (misclassifies singletons)
- ArcFace: do not bother — unstable across s∈{30,64}, m∈{0.3,0.5}
- Auxiliary CE weight 2.0–2.5 alongside triplet is the sweet spot
- Optimizer: AdamW; batch size 32 at 448px

## Open questions
- Why does the aggregator repeatedly fail to select / emit the best candidate's checkpoint? This alone has cost gold→none in one run and gold→silver in others — highest-leverage fix
- Can an explicit validation-tuned cosine-similarity threshold solve the `new_whale` decision instead of relying on learned bias?
- Would rank-averaging two candidates with opposite `new_whale` biases (e.g., 20260220 Candidates 1 + 2) recover the 13.7% gap?
- Can center loss replace auxiliary CE without hurting final MAP@5 while speeding convergence?
- Why does flip-TTA help ResNet50 but slightly hurt ViT-Small? Architecture-specific or hyperparam mismatch
