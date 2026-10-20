# Pattern: ecommerce-booking-recommendations

## When this applies
- Extreme multi-class classification on e-commerce product catalogs (thousands of categories)
- Large BSON/binary product datasets (50+ GB) with multiple images per product
- 24h compute budgets where I/O throughput dominates training wall time
- Top-1 accuracy metrics with per-product aggregation over multi-image inputs

## Proven patterns
- ConvNeXt-Tiny IN-22k pretrain delivers bronze (0.71322, 0.71313 in cdiscount runs 20260418_200431 and 20260420_193337) — the ~25M param sweet spot for 5,270 classes
- Per-product softmax-mean aggregation beats learned fusion / MIL attention (cdiscount: simple aggregation 0.713 vs multi-image attention plateau at 0.631)
- Lazy BSON offset-index streaming with persistent DataLoader workers — avoids loading 53 GB into RAM (cdiscount best runs)
- Iterative checkpoint-resume with conservative LR decay (1e-3 → 3e-5) over 6 short passes beats single long training under 24h budget
- Strong augmentation stack (RandAugment N=2 M=7 + MixUp + CutMix + RandomErasing + label smoothing 0.10) pairs well with modern backbones
- AdamW + cosine warmup + AMP + channels_last on A100 is the reliable optimizer recipe
- Early triage: cut 6 designs → 2–3 survivors to concentrate tuner budget (cdiscount run 5 lessons)

## Common pitfalls
- BSON random-access without streaming/LMDB: <5% GPU util, 200–900 img/s (cdiscount runs 1 and early run 3 stalled at 0.6371 val)
- Scaling backbone without fixing I/O: ConvNeXt-Base 0.535 vs ConvNeXt-Tiny 0.742 (cdiscount)
- DINOv2 ViT-B frozen on 180px product photos: 0.39 accuracy — SSL/ViT mismatch for low-res e-commerce imagery
- Hierarchical multi-task heads (3-level taxonomy) lowered total loss but degraded leaf accuracy (cdiscount run 5: 0.66071)
- Warm-restarting BSON-trained checkpoint on LMDB loader degraded convergence — format mismatch
- Aggressive LR on resume (3e-3) destroyed a tuned checkpoint (0.6701 → degraded)
- Optimizer param-group mismatch on checkpoint reload silently blocks multi-stage training (cdiscount run 6 fix recovered to 0.71313)
- Kaggle rules-acceptance must be pre-accepted via web UI — API-only setup hits 403 (cdiscount run 2 killed)
- Single-GPU full-test inference takes 3.6h on 1.27M products — must batch/distribute eval

## Model selection guide
- Modern e-commerce image catalog at 180–224px → ConvNeXt-Tiny IN-22k (first choice)
- Backup / second candidate → ResNet-50d (cdiscount run 3 hit 0.7118 bronze)
- Avoid: MobileNetV3 (underfits 5K+ classes), ConvNeXt-Base/Small (I/O-bound, convergence risk in 2–3 epochs), frozen ViT/DINOv2 on small images, EfficientNet-B3 progressive resize (too risky under tight budgets)
- Decision rule: if I/O is unsolved, do NOT scale up the model — fix data pipeline first

## Feature engineering patterns
- Build BSON offset index once (minutes) and stream JPEG decode lazily in `__getitem__`
- Alternative: pre-extract to LMDB once (30–40 min) for 5–10× throughput — but train end-to-end on one format; never warm-restart across formats
- Stratified 80/20 split + fixed 300–500-sample holdout validated every ~5k steps
- Per-product inference: average softmax across 1–4 images, no learned fusion needed
- Keep native resolution (180→192 OK); upsampling to 224/384 saturates I/O without accuracy gain
- Consider libjpeg-turbo / PIL-SIMD when decode is the bottleneck

## Tuning patterns
- Conservative LR on every resume (1e-3 → 3e-5); aggressive resets destroy prior gains
- 6 short checkpoint-resume passes > one long epoch under 24h wall clock
- Time-budgeted trainer with graceful shutdown + fallback-to-latest-checkpoint guarantees a valid submission
- Verify optimizer param-groups reload correctly — silent mismatches block multi-stage gains (cdiscount: run 6 recovered to 0.71313 only after fix)
- Label smoothing 0.10 CE consistently helps at 5K+ class scale
- Drop high-overlap candidates at design time (B3 vs B2, ConvNeXt-Small vs Tiny) to free tuner passes for survivors
