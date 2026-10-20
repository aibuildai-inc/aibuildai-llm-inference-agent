# Pattern: rating-ranking-prediction

## When this applies
- Predicting ordinal/bounded targets: essay scores (1-6), preference labels (3-class), pet popularity (1-100)
- Subjective/noisy human-labeled data with class imbalance or right-skewed distributions
- Metrics that reward calibration: QWK, log loss, RMSE on bounded targets

## Proven patterns
- **Match loss to metric, not target dtype**: BCE on [0,1]-normalized targets beats MSE for bounded regression (petfinder: 0.91 RMSE jump on switch); Huber/SmoothL1 > MSE for ordinal essays (essay-scoring 0.8425 Gold). MSE on raw scales fails consistently.
- **Post-hoc calibration is free wins**: Nelder-Mead threshold optimization recovers 0.01-0.03 QWK (essay-scoring); temperature scaling cut log loss 15% (lmsys 1.32→1.11).
- **Test-time augmentation via label symmetry**: Swap+average for pairwise tasks (lmsys silver 0.98784); HFlip+rotation TTA gains 0.2-0.3 RMSE (petfinder).
- **Cross-paradigm ensemble > same-family ensemble**: CLIP+MIM pretraining diversity broke 16.9 barrier on petfinder (Gold 16.877); regression+ordinal blend hit 0.84111 on essay-scoring. Same-family stacking degrades.
- **Multi-sample dropout (5 masks)**: Improved calibration in lmsys (0.98784) and essay-scoring Gold runs.
- **EMA (~0.999)**: Consistent 0.002-0.005 metric gains, low overhead, used in all Gold runs.

## Common pitfalls
- **Oversized backbones timeout the 1200s sanity budget**: DeBERTa-v2-xlarge (lmsys), ViT-Large-CLIP, EVA02-448 (petfinder), DeBERTa-v3-large fp32 (essay-scoring) all killed.
- **Ordinal classification heads collapse**: CORN→QWK=0.0, CORAL fragile standalone (essay-scoring); use only as ensemble diversity.
- **Mixup/CutMix on noisy labels**: Diverged on SwinV2-L (petfinder RMSE 28.18); skip for crowd-sourced targets.
- **Weak ensemble < single strong**: DeBERTa-base ensemble (~0.79 QWK) lost to single DeBERTa-large (essay-scoring); 3+ model auto-ensembles fail on GPU/fold alignment (petfinder).
- **Prediction-level stacking degrades** (petfinder 17.94→18.11); embedding-level (SVR on pooled) works.
- **Extended context/resolution hits compute wall**: 1536 tokens (lmsys 18% epoch), 448px ViT (petfinder OOM), seq_len 1024 fp32 (essay-scoring).

## Model selection guide
- Text ordinal/preference → DeBERTa-v3-large at 512 tokens (RoBERTa-large fallback if NaN issues)
- Bounded image regression → ConvNeXt-Large or Swin-Large at 384px (200M param sweet spot)
- Need Gold? → 2-model cross-paradigm ensemble (CLIP+MIM, regression+ordinal, CNN+Transformer)
- <50M params underfits; >300M times out on small datasets (<20K samples)

## Feature engineering patterns
- **Resolution/sequence-length is the dominant feature**: 384>224px (+0.1-0.5 RMSE); 768-1024 tokens > 512 for long essays (+0.02 QWK).
- **Token budget allocation for pairwise**: 25% prompt / 75% responses (lmsys); head-tail truncation for essays (captures intro+conclusion).
- **Pooled embeddings → bottleneck → SVR/Ridge**: 128-dim bottleneck on ConvNeXt-CLIP gave petfinder 17.008 solo; effective on small data.
- **Attention pooling > mean > CLS** for transformer regression heads (essay-scoring Gold 0.8389).
- **Handcrafted features (length, readability, verbosity)**: Orthogonal signal for ensemble diversity only; never competitive standalone.
- **Stratified K-fold on binned targets**: Mandatory for imbalanced/skewed distributions.

## Tuning patterns
- **LLRD/differential LR mandatory** for Large backbones: backbone 1e-5 to 2e-5, head 1e-4 to 1e-3.
- **Mixed precision (bf16 preferred over fp16)**: 2x speedup; DeBERTa-v3 needs selective fp32 attention to avoid NaN.
- **Folds**: 10-fold for Large models when budget allows (petfinder Gold), 5-fold stratified floor; 3-fold only for iteration.
- **Epochs**: 1 for DeBERTa-v3-large on lmsys (overfits after); 3-5 for essays; 6/fold for image Large.
- **Cosine warmup 10%**; switch to linear decay if resuming from checkpoint.
- **AWP/SWA**: ~0.005 metric gain but ~50% step overhead—budget-permitting only.
