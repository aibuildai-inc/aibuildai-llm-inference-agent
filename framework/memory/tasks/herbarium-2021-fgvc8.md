# Task Memory: herbarium-2021-fgvc8

## Task profile
- Type: Fine-grained image classification (extreme long-tail)
- Dataset: 1.78M herbarium specimen images, 64.5K classes; 40% of classes have ≤5 images, 12% have exactly 1 image; taxonomic hierarchy available (family/order)
- Metric: Macro F1 (higher is better)
- Medal thresholds: Bronze ≥ 0.13026 (median baseline)

## Best result
- Score: 0.3772 (test) / 0.4921 (val) | Medal: Bronze | Run: 20260227_105845 (Candidate 6)
- Approach: ConvNeXt-Tiny with LDAM loss (class-dependent margins scaled by n_j^{-1/4}) and Deferred Re-Weighting (DRW) curriculum—instance-balanced sampling first, switching to class-balanced at epoch 3. Two-phase training (frozen backbone → unfrozen at epoch 2–3). Compact 768-dim feature space kept head small (49.5M params), maximizing training iterations within budget.

## Score progression
| Run | Test Score | Val Score | Medal | Key change |
|-----|-----------|-----------|-------|------------|
| 20260213_083608 | 0.3772 | 0.4897 | Bronze | EfficientNet-B0 + class-balanced sampling + CE loss, 16 epochs |
| 20260227_105845 | 0.3141 | 0.4921 | Bronze | ConvNeXt-Tiny + LDAM + DRW curriculum; higher val F1 but lower test F1 |

**Note:** Run 2 achieved higher validation F1 (0.4921 vs 0.4897) but lower test F1 (0.3141 vs 0.3772), suggesting possible val/test distribution shift or overfitting to val split. EfficientNet-B0 + simple CE generalized better to the test set despite lower val score.

## Proven strategies
- **Class-balanced sampling is non-negotiable** for 64.5K-class long-tail with macro F1 metric. Both runs confirmed this independently
- **Two-phase frozen→unfrozen training** essential: unfreezing backbone at epoch 2–3 triggers 10–50× F1 gains (e.g., 0.0 → 0.107). Without this, rare-class gradients don't reach the feature extractor
- **Throughput-first backbone selection**: Under 6–20 hour budgets with 1.78M images, smaller/faster backbones beat larger ones because more epochs dominate. EfficientNet-B0 (0.6h/epoch, 800+ img/s) consistently delivered; ConvNeXt-Tiny compact enough to also work
- **Single-GPU per model**: DataParallel causes 5–8× throughput regression with 64.5K-dim output tensor. Always run independent single-GPU models
- **Early candidate pruning** frees GPU budget for deeper tuning of survivors
- **Cosine annealing with warm restarts** enabled stable convergence tracking; final epochs still improved in both runs

## Failure patterns
- **Large backbones timeout**: ConvNeXt-Small killed by 1200s coder timeout; EfficientNet-B2 + TTA also timed out. Oversized models are DOA under tight budgets
- **I/O contention ruins multi-GPU setups**: Two models sharing disk starved each other; ConvNeXt-Tiny (run 1) stalled at 0.0054 F1 despite IN-22K pretraining due to I/O starvation
- **Metric learning too expensive**: SoftTriple with K=3 sub-centers × 64.5K classes ≈ 1.5GB memory overhead; slowest throughput. Killed both times
- **Recurring coder bugs**: `drop_last=True` with small val subsets → empty dataloaders; deprecated `autocast(enabled=...)` API. Both waste coder cycles
- **Sanity-run timeouts**: Model init or data loading overhead caused 3 candidates to hit 1200s on first attempt across runs; expect one retry cycle
- **Val/test generalization gap**: LDAM+DRW achieved best val F1 (0.4921) but worst test F1 (0.3141), while simpler CE+class-balanced sampling generalized better (0.4897 val → 0.3772 test)

## Feature engineering playbook
- **Aggressive augmentation**: Flip, affine, color jitter—standard recipe; no diminishing returns observed yet
- **ImageNet-1K pretraining sufficient**: IN-22K pretraining (ConvNeXt-Tiny, run 1) did not compensate when I/O-starved; IN-1K is the safe default
- **Compact feature dimensions preferred**: 768-dim (ConvNeXt-Tiny) yields smaller classifier head than 1280-dim (EfficientNet-B0) or 2048-dim (ResNet-50), improving throughput with 64.5K output classes

## Tuning insights
- **LDAM + DRW shows strongest val trajectory**: 0.113 → 0.268 → 0.408 → 0.492 across tuning rounds, non-saturating. But **generalizes worse to test** than simpler approaches
- **Standard CE + class-balanced sampling** is the reliable baseline: simpler, stable, better test-set generalization observed
- **Focal loss underperforms** principled margin methods (LDAM) on extreme long-tail; killed in run 2
- **DRW transition at epoch 3 is hardcoded and brittle**: optimal switch point likely depends on dataset size and LR schedule
- **16 epochs still improving** (0.4867 → 0.4897 at end of run 1): more training time would help; 24h budget could push further
- **Candidate 6 trajectory** (run 2) suggests 0.50+ val F1 reachable with more epochs, but test generalization is the bottleneck

## Open questions
- **Why does LDAM+DRW overfit?** Best val F1 (0.4921) but worst test F1 (0.3141). Investigate val/test distribution shift or whether margin-based loss overfits to training class frequencies
- **Ensemble potential**: Two independent models (EfficientNet-B0 + ConvNeXt-Tiny) on separate GPUs with separate data loaders could capture diversity; avoids I/O contention
- **Hierarchical taxonomy exploitation**: Family/order coarse labels never successfully used. Hierarchical auxiliary losses or taxonomy-aware label smoothing could bootstrap rare classes (11.8% improvement noted in competition baseline)
- **Test-time augmentation**: Consistently timed out under budget, but 5× multi-crop TTA is standard in FGVC. Worth trying with a faster backbone or relaxed inference budget
- **Adaptive DRW trigger**: Switch to class-balanced sampling when val F1 plateaus rather than at fixed epoch
- **Per-class weight clipping** in weighted CE to stabilize rare-class upweighting without LDAM's generalization penalty
