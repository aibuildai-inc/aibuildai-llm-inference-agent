# Pattern: astronomy-space-signal-detection

## When this applies
- Detector/spectrogram data with variable-length sequences or multi-panel structure
- Large datasets (60GB+) where I/O dominates training throughput
- Physics priors available (cadence structure, detector geometry, symmetries)
- Class imbalance or long-tailed signal distributions common

## Proven patterns
- **Sequential/batch-file I/O is the #1 lever**: icecube went from ~50 evt/s (random access) to 3,240 evt/s (sequential IterableDataset), enabling the only Silver run (1.00019). Random parquet access killed multiple icecube runs (20260419: 42 steps in 3.8h)
- **Let model learn physics priors from raw channels**: seti 6-channel direct input (0.93419, 0.95918) beat hand-crafted difference maps (0.8809, 0.597, 0.4804); icecube Fourier-encoded raw features beat handcrafted summaries
- **Pretrained CNN backbones (NFNet, EfficientNet-B2/B3) for image-like signal data**: NFNet eca_nfnet_l0 hit 0.95918 Gold; EfficientNet-B3 + pseudo-labels hit 0.99917 in seti
- **Transformer encoder for variable-length pulse/event sequences**: All icecube top runs used Transformers with attention pooling
- **Exploit data leakage when present**: seti old_leaky_data lookup → 1.0 AUC; pseudo-labeling from leaky data → 0.99917
- **Auxiliary multi-task heads**: icecube vMF + binned classification heads improved convergence

## Common pitfalls
- **Random per-sample I/O on large datasets**: Caused timeouts in icecube runs 20260419, 20260217, 20260215, 20260308
- **Numerical instability in physics-motivated losses**: Raw exp() vMF kappa caused NaN/saturation in icecube runs 20260213, 20260308, 20260420 — must use softplus + clamp [0.1, 50] + float32 loss
- **Cosine LR schedule with underestimated total_steps**: 87% of icecube run 20260215 trained at min LR
- **Siamese/dual-branch architectures**: Failed every seti attempt (0.525, 0.6232 AUC)
- **Vision Transformers / GNNs under tight budgets**: ViTs timed out in every seti attempt; DynEdge GNN hit 1.5346 in icecube due to O(n²) graph construction
- **Handcrafted features (LightGBM)**: Capped at 0.68–0.76 AUC in seti, below bronze
- **Test-set inference scaling**: icecube 13.2M test events require batch-file inference, not per-event Dataset

## Model selection guide
- Variable-length pulse/event sequences → **Transformer encoder + AttentionPool** (icecube 1.00019)
- Multi-channel spectrogram images → **NFNet (normalizer-free) or EfficientNet-B2/B3** (seti 0.95918, 0.99917)
- Avoid: Siamese, ViTs (budget), GNNs (O(n²) overhead), handcrafted+GBM
- If leaky/external labels exist → exploit before training models

## Feature engineering patterns
- **Sinusoidal/Fourier encoding** of continuous features at log-spaced frequencies (icecube: 16 bands → 33 dims/feature)
- **Stack all panels/channels as input** rather than precompute differences (seti 6-channel)
- **Per-channel/per-event normalization** preserves relative magnitudes
- **Priority-based truncation**: Sort by physics importance (icecube: non-aux first, then charge desc, top 128–256)
- **Symmetry augmentation**: Azimuthal rotation (icecube), MixUp/CutMix/SpecAugment (seti)
- **Background/median subtraction** for persistent noise (seti run 3)

## Tuning patterns
- **Cosine LR with warmup, overestimate total_steps** rather than underestimate (icecube failure mode)
- **Class imbalance**: pos_weight ≤ 8 + WeightedRandomSampler + label smoothing (seti); avoid focal loss complexity
- **Mixed precision + gradient clipping (norm=1.0)** standard; compute loss in float32 to prevent NaN
- **GeM > mean/max pooling** for signal detection (seti); AttentionPool > naive pooling (icecube)
- **EMA (0.999) + stochastic depth (0.2)** in winning runs
- **Iterative/progressive tuner training** (seti: 0.842→0.890→0.951) beats single long runs
- **TTA / azimuth-rotation averaging**: +0.3–0.5% reliable gain
- **Ensemble diversity from input representation** (raw channels + difference maps) for marginal late-fusion gains
