# Pattern: sentiment-opinion-mining

## When this applies
- Binary or multi-class classification of subjective text (insults, sentiment, opinions)
- Small-to-medium datasets (<10k samples) with class imbalance
- Social media / user-generated content with artifacts (caps, slang, misspellings, punctuation)
- AUC-ROC or similar ranking metrics

## Proven patterns
- **Lightweight transformer ensemble beats single large model**: DistilBERT + DistilRoBERTa + ALBERT trio achieved 0.93306 AUC (Gold) on detecting-insults-in-social-commentary; variance reduction outweighs capacity loss on small datasets
- **Weighted averaging optimized on validation AUC** beats unweighted voting (evidence: insults task gold run)
- **Mixed precision (FP16) + gradient accumulation** to reach effective batch 32 within tight training budgets (insults task: 5-min budget)
- **Stratified train/val split** preserves class balance without explicit reweighting (insults: 26.6%/73.4% imbalance handled successfully)
- **Byte-level BPE tokenization (RoBERTa-family)** preserves social media artifacts better than WordPiece

## Common pitfalls
- **Single large transformer on small data**: RoBERTa-base alone scored only 0.54262 AUC on insults task—severe overfitting
- **Training too many epochs**: Validation AUC plateaus by epoch 2–3; >3 epochs causes divergence
- **Aggressive lowercasing/normalization**: Destroys signal-bearing caps and punctuation in social media
- **Large models risking timeout**: RoBERTa-large risks exceeding tight training budgets
- **Missing submission files**: Multiple candidates returned null scores in insults Run 2 due to incomplete output validation

## Model selection guide
- Dataset <5k samples → lightweight transformer ensemble (DistilBERT + DistilRoBERTa + ALBERT)
- Dataset 5k–50k → consider RoBERTa-base with strong regularization + ensemble
- Tight time budget (<5 min) → distilled models + FP16 mandatory
- Need fast baseline → TF-IDF + char n-grams (untested ceiling, worth benchmarking)

## Feature engineering patterns
- **Preserve casing and punctuation**: Capitalization is signal in insults/sentiment
- **Character n-grams** for classical pipelines: handles misspellings and slang
- **Byte-level BPE tokenizer**: Better OOV handling for social media tokens than WordPiece
- **No aggressive cleaning**: Keep emojis, repeated punctuation, ALL CAPS

## Tuning patterns
- **Regularization stack**: Dropout 0.1–0.3 + label smoothing 0.1 + early stopping (patience ≤ 2)
- **Epochs**: Cap at 3; early stopping by epoch 2–3 (insults gold run)
- **Effective batch 32**: Via gradient accumulation (8 steps × micro-batch 4) with FP16
- **Stratified sampling > class reweighting**: Worked on insults task without focal loss
- **Skip tuner/reviser if initial ensemble hits target**: Insights gold reached without iteration
