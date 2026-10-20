# Task Memory: kuzushiji-recognition

## Task profile
- Type: Object detection + classification (character recognition on historical Japanese manuscript pages)
- Dataset: Kaggle "Kuzushiji Recognition" — high-res page images (~2100×3250), 4,113 character classes with extreme long-tail (775 singletons, 2,240 classes with ≤10 examples). ~3,600 training images, ~500k+ character instances.
- Metric: F1-score (center point within ground-truth bbox + correct Unicode label), higher is better
- Medal thresholds: Silver ≈ 0.87–0.89, Gold ≈ 0.91+

## Best result
- Score: 0.94019 | Medal: Gold | Run: 20260308_193236
- Approach: Class-agnostic CenterNet detector (ResNet-50 FPN, heatmap regression, 5 epochs) locates character centers; separate EfficientNet-B0 classifier on cropped patches handles 4,113 classes with RepeatFactorSampler (10–50× upsampling), focal loss, mixup (α=0.8), label smoothing (0.1), and cosine annealing LR with warmup.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_041328 | N/A | — | Setup failure (Kaggle auth/interactive prompt blocked) |
| 20260211_064332 | 0.8862 | Silver | Two-stage Faster R-CNN detector + EfficientNet-B0 classifier; threshold tuning |
| 20260225_224939 | 0.8762 | Silver | Two-stage Faster R-CNN (ResNet-FPN) + ResNet classifier; k-means anchor tuning, class-balanced sampling |
| 20260308_193236 | 0.9402 | Gold | CenterNet detector + EfficientNet-B0 classifier; RepeatFactorSampler, focal loss, mixup, label smoothing |
| 20260309_194057 | 0.9168 | Gold | CenterNet + EfficientNet-B0; pre-extract zip to disk for 4× data loading speedup; 12-epoch classifier |

## Proven strategies
- **Two-stage decoupling is mandatory**: Class-agnostic detection (binary: char vs. background) + separate N-class classifier. Every successful run uses this. End-to-end 4,113-class detection heads fail consistently (F1=0.096 in run 2, rejected in all later runs).
- **CenterNet > Faster R-CNN for this metric**: Heatmap center-point regression naturally aligns with eval metric (center within bbox). CenterNet achieved 0.94 vs. Faster R-CNN's 0.88. Direct center prediction avoids box→center conversion imprecision.
- **EfficientNet-B0 classifier on 96×96 crops**: Best accuracy/compute tradeoff. Saturates by epoch 3; ResNet-18 undersized, ResNet-50 overkill for cropped patches.
- **RepeatFactorSampler for long-tail**: Upsampling rare classes 10–50× in one large epoch (~523k crops) beats many small epochs. Superior to WeightedRandomSampler for this distribution.
- **Focal loss + label smoothing (0.1) + mixup (α=0.8)**: This trio consistently stabilizes rare-class learning in the classifier stage.
- **Cosine annealing LR with warmup (500 steps)**: Prevents overshooting on small rare-class batches.
- **Detector saturates fast (~5 epochs)**: ResNet-50 FPN detector reaches ~0.856 recall in 5 epochs; extra epochs waste budget. Shift budget to classifier.
- **Threshold tuning on detector confidence**: Directly trades recall↔precision; was the main lever for initial gains (0.827→0.887 in run 2).

## Failure patterns
- **End-to-end 4,113-class detection**: Fails every time. Faster R-CNN end-to-end: F1=0.096 after 3 epochs (run 2). FCOS, RetinaNet, DETR all rejected at design review for same reason.
- **Per-class heatmap (4,113 channels)**: ~400MB per sample; completely infeasible.
- **Single-stage dense detectors (FCOS, RetinaNet)**: Memory-prohibitive with 4k+ classes; rejected in runs 3–5.
- **Sliding window / tiling approaches**: Boundary artifacts, slow inference (30–90 min), no precedent.
- **Semantic segmentation + instance embedding**: Overlapping characters unsolvable; lowest expected F1.
- **Zip-based on-the-fly data loading**: Hidden bottleneck forcing NUM_WORKERS=1. Pre-extracting to disk (~30s) enables 4× speedup. Always profile data loading first.
- **Kaggle interactive auth in headless environments**: `input()` + browser-based rule acceptance causes EOFError. Must pre-accept rules and use env-var credentials.
- **Coder timeouts**: Recurring issue; fix with lighter sanity checks, tuned batch sizes (64→256), and pre-extraction.

## Feature engineering playbook
- **K-means anchor tuning**: Cluster ground-truth box dimensions (6px–520px range) to set custom anchor sizes. Generic anchors miss small characters.
- **Tiling + overlap + NMS merging**: Handles high-res pages without memory explosion. Overlapping crops prevent edge artifacts and false negatives at tile boundaries.
- **Crop size 96×96 for classifier**: Upgrade from 64×64 meaningfully improves accuracy with minimal compute cost.
- **Pre-extract images from zip to disk**: ~30s upfront cost unlocks multi-worker data loading (NUM_WORKERS=4).

## Tuning insights
- **Iterative tuning on the leader pays off**: Candidate 5 in run 3 gained +0.11 F1 across 4 tuning rounds (0.761→0.873). Early-stop weaker candidates to free budget.
- **Classifier is the bottleneck, not detector**: Detector saturates at 5 epochs; spend remaining budget on classifier epochs, augmentation, and sampling strategy.
- **Batch size 256 for classifier**: Larger batches avoid timeout issues and stabilize training with class-balanced sampling.
- **Detector batch_size=2 on full-res images (max 1024px)**: Memory-safe for ResNet-50 FPN.
- **RepeatFactorSampler > WeightedRandomSampler**: Former handles extreme long-tail better; latter caused confusion in implementation.
- **Precision is typically the bottleneck**: Raising detector confidence threshold reduces false positives; inspect precision first.

## Open questions
- **Larger classifier backbone**: EfficientNet-B2/B3 with higher-res crops (96→128px) and longer training may push beyond 0.94.
- **Ensemble CenterNet + Faster R-CNN detectors**: Different architectures; soft voting on classifier confidence could improve robustness (untested across all runs).
- **Metric learning for singletons**: ArcFace/CosFace on Stage 2 classifier may outperform cross-entropy + oversampling for 775 singleton classes.
- **CenterNet-specific augmentation**: Adaptive Gaussian sigma per instance size could reduce detector miss rate on tiny characters (<30px).
- **Pre-cropping detected patches for classifier training**: Tuner-suggested but never tested; could speed up Stage 2 significantly.
- **Simpler classifier head**: Could MLP suffice on FPN-quality crops instead of full ResNet/EfficientNet? May save training time.
