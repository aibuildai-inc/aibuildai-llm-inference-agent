# Task Memory: detecting-insults-in-social-commentary

## Task profile
- Type: Binary classification (insult vs. non-insult)
- Dataset: ~4,000 samples from diverse social commentary sources (forums, blogs, news); class imbalance ~26.6% insult / 73.4% non-insult; text contains social media artifacts (caps, misspellings, slang, punctuation)
- Metric: AUC-ROC (higher is better)
- Medal thresholds: Gold achieved at 0.93306; median baseline ~0.79111

## Best result
- Score: 0.93306 AUC-ROC | Medal: Gold | Run: 20260209_011923
- Approach: Ensemble of three lightweight transformers (DistilBERT, DistilRoBERTa, ALBERT) trained 3 epochs each with mixed precision, gradient accumulation (effective batch 32), early stopping (patience 2), and dropout 0.1–0.3. Predictions aggregated via weighted averaging optimized on validation AUC.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260209_003502 | ungraded | — | Pipeline state corruption; 5 designs created but none trained |
| 20260209_011923 | 0.93306 | Gold | Lightweight transformer ensemble (DistilBERT + DistilRoBERTa + ALBERT) with weighted averaging |

## Proven strategies
- **Lightweight transformer ensemble** over single large model on <5k samples: variance reduction outweighs capacity loss; DistilBERT + DistilRoBERTa + ALBERT is the proven trio
- **Early stopping with patience ≤ 2**: Validation AUC plateaus by epoch 2–3; overfitting is the primary failure mode on this dataset
- **Mixed precision (FP16) + gradient accumulation (8 steps)**: Mandatory to fit effective batch size 32 within the 5-minute training budget
- **Weighted prediction averaging** optimized on validation AUC beats unweighted voting
- **Stratified 80/20 split** preserves class balance without needing explicit class reweighting
- **Byte-level BPE tokenization** (RoBERTa-family) preserves social media artifacts (caps, punctuation); avoid aggressive lowercasing

## Failure patterns
- **Single large transformer (RoBERTa)**: Candidate 2 scored only 0.54262 AUC—severe overfitting or poor augmentation on tiny dataset
- **Pipeline state corruption** (Run 1): Designer succeeded but judge never invoked; `alive_nums` emptied without execution evidence—root cause undiagnosed
- **Timeout risk**: Larger single models (RoBERTa-large) risk exceeding the 5-minute budget
- **Incomplete submissions**: Candidates 1 and 5 in Run 2 returned null scores due to missing submission files

## Feature engineering playbook
- **Character n-grams** matter for classical pipelines: misspellings and slang in social media demand `char_ngram` features
- **Preserve casing and punctuation**: Social media insults heavily use capitalization and special characters as signals; do not aggressively lowercase
- **Tokenizer choice matters**: Byte-level BPE (RoBERTa-family) handles out-of-vocabulary social media tokens better than WordPiece

## Tuning insights
- **Regularization stack**: Dropout 0.1–0.3 + label smoothing 0.1 + early stopping (patience 2) is the working recipe
- **Epochs**: 3 epochs max; training diverges after epoch 2–3 on this dataset size
- **Batch size**: Effective batch 32 via gradient accumulation (8 steps × micro-batch 4) with FP16
- **No tuner/reviser runs were attempted** in winning run; initial design choices were sufficient for gold
- **Focal loss / explicit class reweighting**: Not tested in winning run; stratified sampling + early stopping sufficed

## Open questions
- **Focal loss / asymmetric loss**: Never tested; could improve recall on missed insults if business context penalizes false negatives
- **Data augmentation diversity across ensemble members**: Back-translation, synonym replacement, contextual masking—could reduce correlated errors if all models overfit the same patterns
- **Ensemble weight robustness**: Current weights optimized on single validation fold; cross-validation or held-out test fold needed to confirm transferability
- **Distribution shift resilience**: Diverse source mix (forums, blogs, news) may cause verification set shift; out-of-fold stacking or domain-aware sampling untested
- **Classical baseline ceiling**: TF-IDF + char n-gram ensemble (<1 min training) never successfully benchmarked due to Run 1 failure; worth establishing as a fast baseline
- **Pipeline state corruption**: Root cause from Run 1 still undiagnosed; needs explicit permission/path verification before execution
