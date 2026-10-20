# Task Memory: cdiscount-image-classification-challenge

## Task profile
- Type: Extreme multi-class image classification (5,270 categories)
- Dataset: 53–56 GB BSON; ~7.07M products, ~12–15M images at 180×180 JPEG; 1–4 images per product; 706K–1.27M test products
- Metric: Top-1 accuracy (higher is better); per-product prediction via softmax-mean aggregation over images
- Medal thresholds: Bronze ≈ 0.70898; silver unknown; Kaggle winner ≈ 0.796

## Best result
- Score: 0.71322 | Medal: Bronze | Run: 20260418_200431
- ConvNeXt-Tiny with ImageNet-22k pretrain, MixUp/CutMix + RandAugment + random erasing + label smoothing 0.10, lazy BSON streaming, per-product softmax-mean aggregation, iterative checkpoint-resume tuning over 6 passes in 24 h. 20260420_193337 reached 0.71313 with the same recipe after fixing an optimizer param-group resume bug (internal val 0.7422).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260305_074056 | 0.6371 (val) | ungraded | EfficientNet-B0 on BSON, I/O-bottlenecked at 200–900 img/s |
| 20260305_181654 | — | ungraded | Killed at setup: Kaggle rules-acceptance 403 |
| 20260305_182848 | 0.7118 | Bronze | ResNet-50d + full LMDB conversion; checkpoint resume with LR decay |
| 20260418_200431 | 0.71322 | Bronze | ConvNeXt-Tiny IN-22k + MixUp/CutMix + streaming BSON |
| 20260419_162328 | 0.66071 | none | ResNet50 frozen→unfrozen; hierarchical/multi-image variants failed |
| 20260420_193337 | 0.71313 | Bronze | ConvNeXt-Tiny rerun after fixing optimizer resume bug |

## Proven strategies
- ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`) + GeM pooling + single 5,270-way linear head — two independent bronze runs
- Lazy BSON offset-index streaming DataLoader with persistent workers; avoids loading 53 GB into memory
- Per-product softmax-mean aggregation of per-image logits at inference — no learned fusion needed
- Checkpoint-resume multi-stage training under 24 h budget; incremental LR decay (e.g., 1e-3 → 3e-5) beats restart
- Label smoothing 0.10 CE; AdamW + cosine warmup + AMP; channels_last on A100
- Strong aug stack (RandAugment N=2 M=7 + MixUp + CutMix + RandomErasing) when paired with modern backbone
- ~25M-param sweet spot (ResNet-50d or ConvNeXt-Tiny) for 5,270 classes at 3–4 epoch budgets
- Aggressive early triage: cut 6 designs → 2–3 survivors to concentrate tuner passes

## Failure patterns
- BSON random-access I/O without conversion/streaming: <5% GPU util, 200–900 img/s (runs 1, early run 3)
- Kaggle competition rules acceptance blocks API-only setup (run 2) — must be pre-accepted via web
- Scaling up architecture without fixing I/O: ConvNeXt-Base 0.535 vs Tiny 0.742; ConvNeXt-Small pruned for convergence risk
- DINOv2 ViT-B frozen: 0.39 accuracy — SSL/ViT mismatch for 180px product photos; upsampling 180→224 hurts
- Hierarchical multi-task (3-level taxonomy) degraded leaf accuracy despite lower total loss (run 5)
- Multi-image attention / MIL aggregation plateaued at 0.631, beaten by simple softmax-mean
- Warm-restarting a BSON-trained checkpoint on LMDB loader degraded convergence (format mismatch)
- Aggressive LR on resume (3e-3) can destroy a tuned checkpoint (0.6701 → degraded)
- Optimizer param-group mismatch on checkpoint reload silently blocks multi-stage training (run 6 fix)
- MobileNetV3 (5.4M) underfits 5,270 classes; EfficientNet-B3 progressive resize too risky in 2–3 epochs
- Single-GPU full-test inference: 3.6 h on 1.27M products — batch/distributed eval required
- Coder sanity runs sometimes fail to emit `results.json`/checkpoints (cands 1–2 in run 5), killing candidates before training

## Feature engineering playbook
- Build BSON offset index once (cold-start, minutes) and stream via lazy `__getitem__` with on-the-fly JPEG decode
- Alternative: full pre-extraction to LMDB (30–40 min one-time) delivers 5–10× throughput; only worthwhile if training end-to-end on LMDB — do not warm-restart across formats
- Stratified 80/20 train/val split; fixed 300–500-sample holdout for periodic validation every ~5k steps
- Inference: average softmax over 1–4 product images; no learned fusion
- Input 180→192 px acceptable; 180→224 (ViT) or 180→384 (large CNN) I/O-bounded the GPU
- Consider libjpeg-turbo / PIL-SIMD to relieve decode bottleneck (untried)

## Tuning insights
- Optimizer: AdamW + cosine annealing + warmup; AMP; channels_last
- LR on resume must be conservative (1e-3 → 3e-5); aggressive resets destroy prior gains
- 6 short checkpoint-resume passes > single long epoch under 24 h budget
- Time-budgeted trainer with graceful shutdown + fallback-to-latest-checkpoint guarantees a valid submission
- Drop high-overlap candidates at design (B3 vs B2; ConvNeXt-Small vs Tiny) to free tuner budget

## Open questions
- Ensemble of surviving candidates (Tiny + B0/ResNet-50d) or TTA to push 0.713 → silver — never attempted
- ConvNeXt-Small/Base diagnosis: genuine I/O saturation vs resume bug vs initialization — isolate with decode-throughput profiling
- Can multi-GPU or larger-batch inference cut 3+ h eval to <15 min?
- Why did hierarchical multi-task lower loss but hurt leaf accuracy — loss-weight balance (1.0/0.3/0.3) or lossy mapping?
- Can MixUp/RandAugment design from failed cand (run 5) be salvaged on a working trainer?
- Is 0.676 a hard ceiling for ResNet50-frozen path, or would pseudo-labeling / multi-crop TTA break it?
