# Task Memory: hotel-id-2021-fgvc8

## Task profile
- Type: Fine-grained image retrieval / classification (7,770 hotel classes)
- Dataset: ~87.8K training images, median 8 images/hotel, 723 hotels with ≤5 images, 2 hotels with only 1 image; chain labels available (88 chains) but 18K images unlabeled; image resolutions span 768×1024 to 5312×2988
- Metric: MAP@5, higher is better
- Medal thresholds: Silver ≈ 0.50+, Gold ≈ 0.72+ (estimated from runs)

## Best result
- Score: 0.7392 | Medal: Gold | Run: 20260309_121002
- Approach: EfficientNetV2-S + ArcFace (margin 0.3, scale 64) at 384px with GeM pooling, L2-normalized embeddings, TTA (horizontal flip), and extended training (~7.4 epochs). Backbone upgrade from EfficientNet-B3 + resolution scaling + margin reduction broke a plateau.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_092648 | 0.69857 | Silver | ConvNeXt-Small + ArcFace, progressive 224→256px |
| 20260226_163300 | 0.51502 | Silver | ConvNeXt-Small + CE + bottleneck (val MAP@5=0.967 but test gap) |
| 20260309_121002 | 0.73920 | Gold | EfficientNetV2-S + ArcFace margin 0.3, 384px, TTA |
| 20260318_165035 | 0.71559 | Silver | CLIP ViT-L/14 + ArcFace, partial unfreezing |
| 20260320_141753 | 0.54596 | Silver | CLIP ViT-B/16 + ArcFace, 336px, layer-wise LR decay |

## Proven strategies
- **ArcFace metric learning consistently outperforms CE on test set**: Despite run 20260226 showing high val MAP@5 (0.967) with CE, it scored only 0.515 on test — a massive train/test gap. ArcFace runs (0.699, 0.739, 0.716, 0.546) generalized far better.
- **GeM pooling + L2 normalization**: Standard retrieval stack; used in every competitive submission. Learnable pooling exponent adapts to data automatically.
- **Progressive resolution training**: 224→256px (run 1), 288→384px (run 3). Starting lower saves memory; scaling up captures fine-grained hotel details. Direct jump to high res risks timeouts.
- **Checkpoint resumption across tuner rounds**: Critical for maximizing training within wall-clock budgets. Must handle PyTorch 2.6 `weights_only=True` issue (use `weights_only=False`).
- **Test-time augmentation (horizontal flip)**: Cheap ~1-2% MAP@5 boost; always include for retrieval tasks.
- **Aggressive early pruning of weak candidates**: 1-2 epoch sanity checks reliably identify non-viable candidates; free GPU budget for winner.
- **Layer-wise LR decay for pretrained backbones**: 0.1× backbone / 10× head prevents catastrophic forgetting, especially critical for CLIP and large models.

## Failure patterns
- **Timeouts during coder phase**: Candidates with multi-scale GeM, re-ranking O(n²), or complex auxiliary heads repeatedly timed out before training even started. Implementation complexity is the #1 killer.
- **SubCenter ArcFace**: Attempted in 3 runs, never successfully trained. K sub-centers × 7,770 classes creates parameter explosion; with median 8 images/hotel, K>2 is unsupported by data. Killed by judge or OOM every time.
- **Triplet loss**: Consistently killed or underperformed; ~10% lower ceiling than classification/ArcFace per Hotel-ID 2021 paper, slow convergence, PK sampling edge cases with long tail.
- **Large ViTs (EVA-02-Large)**: OOM at 304M params + ArcFace head even with gradient checkpointing on 40GB GPU.
- **CE + bottleneck overfits**: Run 20260226 val=0.967 vs test=0.515 is the largest gap observed. CE learns class-specific features that don't transfer to retrieval on unseen test distribution.
- **PyTorch 2.6 `weights_only=True` default**: Caused multiple candidate timeouts on checkpoint resume across runs 4-5. Always set `weights_only=False`.

## Feature engineering playbook
- **Image resolution sweet spot**: 256-384px balances detail capture vs memory/time. 384px was used in the gold run. Beyond 448px untested but promising if budget allows.
- **Augmentation stack**: RandomResizedCrop, ColorJitter, horizontal flip, Mixup/CutMix at batch level. Mixup/CutMix regularizes without blurring fine-grained details.
- **EMA model tracking**: Provides stability; use EMA weights at inference for marginal gains.
- **Embedding dimension**: 512-dim bottleneck standard across all runs. Ablation of 256 vs 1024 still untested.
- **Chain labels (88 chains)**: Weak auxiliary signal; not sufficient alone. Multi-task heads add complexity without clear gains. Potentially useful as post-hoc re-ranking prior.

## Tuning insights
- **ArcFace margin**: 0.5 is too aggressive for 7,770 fine-grained classes; reducing to 0.3 broke a plateau in the gold run. Typical sweet spot: 0.25–0.35. Scale: 30–64.
- **Training epochs**: Plateau at ~12-16 epochs for most architectures. Monitor validation curves; MAP@5 gains slow to ~0.006 per 500 steps near convergence.
- **Backbone ranking** (by test MAP@5): EfficientNetV2-S@384px (0.739) > CLIP ViT-L/14 (0.716) > ConvNeXt-Small+ArcFace (0.699) > CLIP ViT-B/16@336px (0.546) > ConvNeXt-Small+CE (0.515).
- **CLIP pretraining**: Strong initialization (0.66 MAP@5 in <2 epochs with CLIP ViT-L/14), but didn't beat EfficientNetV2-S with full ArcFace fine-tuning. CLIP ViT-B/16 underperformed, suggesting scale matters for CLIP on this task.
- **Partial unfreezing for CLIP**: Gradual block unfreezing (10-20 blocks) prevents mode collapse. Full freeze plateaus, full unfreeze destabilizes.
- **Budget allocation**: Spend 60-80% of agent calls on extended tuner runs, not design/code iteration. Stick to proven low-complexity designs when budget is tight.

## Open questions
- **SubCenter ArcFace on CNNs**: Never successfully trained despite 3 attempts. Could K=2-3 sub-centers with EfficientNet-B4 address multi-modal hotel images (bathrooms vs bedrooms)? Estimated +0.02-0.05 if implementable.
- **CLIP ViT-L/14 + extended fine-tuning**: Reached 0.716 with minimal epochs; could full ArcFace fine-tuning at 384px match or beat the 0.739 gold?
- **Ensemble of ArcFace + CosFace or CLIP + EfficientNet**: Score fusion at inference untested. Two independently trained metric learning models could reduce variance.
- **Progressive multi-scale training** (224→448→672): Could recover fine details lost in downsampling for high-res hotel images.
- **BN-neck ablation**: Alive but never trained in run 3. Compare BN-neck vs standard L2-norm projection.
- **Higher resolution (448-672px)**: Only tested up to 384px. Hotel images are high-res; more input resolution may help if memory permits.
- **Test-time augmentation beyond horizontal flip**: Multi-scale crops, rotation; untested but potentially significant for retrieval.
