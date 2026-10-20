# Pattern: text-classification

## When this applies
- Text-only or text-dominant classification (binary, multi-label, multi-class)
- Dataset sizes from ~3K (random-acts-of-pizza) to ~1.8M (jigsaw-unintended-bias)
- Class imbalance is common (75:25 to 99.7:0.3)

## Proven patterns
- **Multi-architecture transformer ensembles win**: 3-model (BERT+RoBERTa+DeBERTa) → Gold 0.98821 on jigsaw-toxic; 10-model stack → Silver 0.22188 on spooky-author. Tokenizer diversity (WordPiece/BPE/SentencePiece) adds 0.3-0.5% over best solo.
- **k-fold stratified CV (k=5) with OOF stacking** beats simple averaging on jigsaw-toxic (Gold) and spooky-author (Silver 0.22188 vs 0.22503).
- **Mean pooling > [CLS] token** consistently (jigsaw-toxic Gold, jigsaw-unintended best run).
- **HuggingFace Trainer over custom loops**: spooky-author run 2 required this pivot; custom loops repeatedly failed to converge.
- **TF-IDF + LR as ensemble member**: adds non-transformer diversity (spooky-author 0.22188 stack); also viable baseline on small data.
- **Tree boosting (XGBoost/LightGBM) dominates small tabular+text** (<5K samples): random-acts-of-pizza Silver 0.78067; transformers consistently failed there.

## Common pitfalls
- **DistilBERT/small transformers straddle medal thresholds**: 0.98652 bronze on jigsaw-toxic, 0.402 on spooky-author — under-capacity for nuanced tasks.
- **DeBERTa-v3 numerical instability**: collapsed to 0.5 on jigsaw-unintended (fp16 weight loading); crashed 15+ times on spooky-author. Use explicit `model.float()` or fp16+GradScaler (not bf16+autocast).
- **Data leakage from post-hoc features**: `_at_retrieval` columns inflated random-acts-of-pizza to AUC 0.9977 (fake).
- **Stacking overfits on <3K samples**: random-acts-of-pizza two-level stack collapsed to AUC=1.0.
- **Coder timeouts at 1200s**: killed all 6 candidates on jigsaw-unintended run 1; common on DeBERTa + 1M+ samples.
- **LR scheduler computed on wrong epoch count** caused divergence on jigsaw-unintended.
- **Frozen transformer embeddings** (random-acts-of-pizza): always lose to tree models on small domain-specific data.

## Model selection guide
- <5K samples, mixed text+tabular → XGBoost/LightGBM + hand-crafted features + TF-IDF→SVD(50)
- 5K–500K samples, pure text → RoBERTa-large or DeBERTa-base ensemble with 5-fold CV
- >500K samples, time-budgeted → DeBERTa-v3-base or RoBERTa-base (throughput sweet spot, 2-3 epochs)
- Multi-class stylistic → stack RoBERTa-large + BERT-large-cased + TF-IDF+LR (spooky-author Silver)
- Avoid: DistilBERT for medal targets, LightGBM for stylistic text (spooky 0.47-1.08), DeBERTa-large unless k-fold CV available

## Feature engineering patterns
- **Pretrained transformers**: no hand engineering needed; cased tokenizers preserve case signals (jigsaw-toxic).
- **Small data (<5K)**: hand-crafted features dominate — char/word counts, punctuation ratios, custom domain lexicons (random-acts-of-pizza Silver).
- **Imbalanced classes**: stratified split on most-frequent positive label; identity-balanced batch sampling for bias-aware metrics (jigsaw-unintended).
- **TF-IDF → TruncatedSVD(50)** mandatory on small data; raw TF-IDF overfits.
- **Sequence length 256** covers ~95% content for comment-style text; 192 OK if budget-tight.

## Tuning patterns
- **Layer-wise LR decay 0.95/layer**: reliable across BERT/RoBERTa/DeBERTa (jigsaw-toxic Gold).
- **pos_weight capped at 10-50** (or sqrt-dampened): raw inverse-frequency causes NaN explosions.
- **Focal loss (γ=2.0, α=0.25)** for severe imbalance; matched/beat weighted BCE on jigsaw-unintended.
- **scale_pos_weight ≈ class ratio** for tree models (3.0 on random-acts-of-pizza).
- **2-4 epochs** sufficient; LR=2e-5 standard; fp16+GradScaler essential for large models.
- **Tuner plateaus after 1-2 rounds**: gains come from features/architecture, not param sweeps (random-acts-of-pizza).
- **GPU pinning (1 candidate per GPU)**: eliminated 2.5-7× slowdowns on spooky-author.
