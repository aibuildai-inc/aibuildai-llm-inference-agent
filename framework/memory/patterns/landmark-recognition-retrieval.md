# Pattern: landmark-recognition-retrieval

## When this applies
- Fine-grained image retrieval/classification with many classes (1000s+)
- Long-tailed distributions with few images per class (median <10)
- MAP@K-style metrics where embedding quality matters more than classification accuracy
- High-resolution images with discriminative fine details

## Proven patterns
- **ArcFace > CE for retrieval**: hotel-id-2021 gold run (0.7392) used ArcFace; CE+bottleneck scored only 0.515 despite val MAP@5=0.967 — massive train/test gap proves metric learning generalizes far better
- **GeM pooling + L2-normalized embeddings**: standard retrieval stack used in every competitive hotel-id-2021 submission
- **Progressive resolution training (224→384px)**: gold run scaled up; saves memory early, captures detail later
- **TTA with horizontal flip**: cheap ~1-2% MAP@5 boost, included in gold run
- **Layer-wise LR decay for pretrained backbones**: 0.1× backbone / 10× head prevents catastrophic forgetting (CLIP runs)
- **Checkpoint resumption across tuner rounds**: critical for maximizing wall-clock budget; set `weights_only=False` for PyTorch 2.6+

## Common pitfalls
- **CE + bottleneck overfits catastrophically**: hotel-id-2021 run 20260226 val=0.967 vs test=0.515 (largest gap observed)
- **SubCenter ArcFace fails to train**: 3 attempts in hotel-id-2021, all OOM/killed — K sub-centers × 7770 classes explodes parameters
- **Triplet loss underperforms**: ~10% lower ceiling than ArcFace, slow convergence, PK-sampling fragile on long tails
- **Large ViTs OOM**: EVA-02-Large (304M) + ArcFace head OOM on 40GB even with gradient checkpointing
- **Implementation complexity kills runs**: multi-scale GeM, O(n²) re-ranking, auxiliary heads timed out repeatedly in coder phase
- **Aggressive ArcFace margin (0.5)** plateaus on fine-grained classes; reducing to 0.3 broke plateau in gold run

## Model selection guide
- Tight budget / proven baseline → EfficientNetV2-S @ 384px + ArcFace (hotel-id-2021: 0.7392 gold)
- Strong pretrained init available → CLIP ViT-L/14 + ArcFace with partial unfreezing (0.7156 silver)
- Avoid: ConvNeXt-Small + CE (0.515), CLIP ViT-B/16 small scale (0.546), EVA-02-Large (OOM)
- Default ranking: EfficientNetV2-S@384 > CLIP-L/14 > ConvNeXt-S+ArcFace >> any CE variant

## Feature engineering patterns
- Resolution sweet spot 256–384px; gold runs used 384px
- 512-dim L2-normalized embedding bottleneck is standard
- Augmentation: RandomResizedCrop + ColorJitter + HFlip + Mixup/CutMix (regularizes without blurring fine details)
- EMA weights at inference for stability
- Auxiliary labels (e.g., chain in hotel-id-2021) are weak signals alone; better as re-ranking priors than multi-task heads

## Tuning patterns
- ArcFace margin: 0.25–0.35 sweet spot for 1000s of classes; scale 30–64
- Training plateaus at ~12–16 epochs; gains slow to ~0.006 MAP@5 per 500 steps near convergence
- Allocate 60–80% of budget to extended tuner runs on proven low-complexity designs, not design/code churn
- Prune weak candidates after 1–2 epoch sanity checks to free GPU budget for winner
- For CLIP backbones: gradual block unfreezing (10–20 blocks) — full freeze plateaus, full unfreeze destabilizes
