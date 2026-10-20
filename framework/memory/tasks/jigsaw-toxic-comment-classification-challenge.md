# Task Memory: jigsaw-toxic-comment-classification-challenge

## Task profile
- Type: Multi-label text classification (6 binary toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate)
- Dataset: ~159K Wikipedia comments; extreme class imbalance — 89.8% all-negative, threat 0.30%, severe_toxic 1.0%, identity_hate 0.55%; text is informal, median ~205 chars, max ~5000
- Metric: Mean column-wise ROC AUC (higher is better)
- Medal thresholds: Bronze ≥ 0.98639, Silver & Gold above that (Gold achieved at 0.98821)

## Best result
- Score: 0.98821 | Medal: Gold | Run: 20260308_193205
- Approach: 3-model ensemble (BERT-base + RoBERTa-base + DeBERTa-large) with 5-fold stratified CV, BCEWithLogitsLoss with pos_weight capped at 50.0, layer-wise LR decay (0.95/layer). Final submission averaged sigmoid probabilities across all models and folds, exploiting tokenizer diversity (WordPiece, BPE, SentencePiece).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260209_173938 | N/A | — | Kaggle auth failure; no training |
| 20260209_174843 | N/A | — | Incomplete; 3 transformers implemented, val AUC 0.989–0.991 at epoch 1 |
| 20260209_214256 | N/A | — | Incomplete; coder stuck on tokenizer/sklearn bugs |
| 20260209_220139 | N/A | — | Zero candidates; initialization failure |
| 20260209_221526 | 0.9915 (val) | ungraded | BERT-base single model, weighted BCE, 4 epochs; no test submission |
| 20260210_182451 | 0.98652 | Bronze | DistilBERT + focal loss (γ=2.0, α=0.25), 3 epochs, batch 48 |
| 20260226_182158 | 0.98723 | Silver | 3-model ensemble (DeBERTa-v3-large + DeBERTa-v3-base + RoBERTa-base), LLRD, pos_weight capping |
| 20260227_023008 | 0.98551 | None | DeBERTa-v3-base + RoBERTa ensemble; large model overfit, base outperformed large |
| 20260227_184903 | 0.98711 | Silver | BERT-base + RoBERTa + DeBERTa-v3-base ensemble, mean pooling, LLRD |
| 20260308_193205 | 0.98821 | Gold | 3-model ensemble with 5-fold CV, pos_weight capped at 50.0, LLRD |

## Proven strategies
- **3-model ensemble with tokenizer diversity is the winning formula**: Combining WordPiece (BERT), BPE (RoBERTa), and SentencePiece (DeBERTa) consistently adds 0.3–0.5% AUC over the single best model. Simple sigmoid-probability averaging is sufficient.
- **Layer-wise LR decay (LLRD) at 0.95/layer**: Consistently improves convergence and final AUC across all transformer architectures; prevents catastrophic forgetting. Backbone LR ~1/3 of head LR for large models.
- **BCEWithLogitsLoss with capped pos_weight**: Standard approach for multi-label. Raw inverse-frequency pos_weight causes NaN/explosion (threat → 330×); cap at 10–50 range. sqrt-dampening also works.
- **k-fold stratified CV (k=5)**: Essential for variance reduction on imbalanced multi-label data with only ~160K samples; fold ensembling more impactful than hyperparameter tuning.
- **Mean pooling > [CLS] token**: Aggregating all non-padding token representations outperforms [CLS]-only for this task.
- **Sequence length 256**: Covers ~95% of semantic content in Wikipedia comments; avoids memory bloat vs 512.
- **Cased tokenizers preserve toxicity signals**: ALL-CAPS, aLT cApS patterns correlate with toxicity; uncased models lose this entirely. RoBERTa (cased BPE) and DeBERTa consistently outperform uncased BERT by 0.2–0.3% AUC.

## Failure patterns
- **DistilBERT consistently below threshold**: Expected AUC 0.982–0.986 straddles bronze (0.98639); killed in judge phase across 4+ runs. 6 layers insufficient for subtle toxicity distinctions.
- **LSTM/CNN architectures fundamentally under-capacity**: Plateau at 0.97–0.98 AUC; waste budget. Non-transformers lack contextual semantics for nuanced toxicity.
- **DeBERTa-v3-large overfit risk under budget constraints**: In run 20260227_023008, DeBERTa-large (0.9889) underperformed base (0.9917) — rare-class overfitting with 304M params on 159K samples. But when 5-fold CV + proper regularization used (20260308), large model shone (0.9926 best fold). Large models need k-fold to work reliably.
- **TF-IDF + LightGBM**: Max AUC ~0.985; cannot capture sentence-level semantics. Rejected.
- **BERT-large**: 3× compute of base, overlapping AUC range with RoBERTa-base, same tokenizer family — no ensemble diversity gain.
- **DeBERTa-v3 tokenizer dependency issues**: Requires sentencepiece/tiktoken; fails in constrained offline environments. Always verify package availability.
- **Coder timeouts**: DeBERTa models hit 1200s timeout repeatedly; inefficient data loading and missing fp16→float conversion are common culprits.
- **Sklearn incompatibility with pandas ArrowDtype**: `train_test_split` fails on `.values` with Arrow-backed DataFrames; convert to numpy explicitly.

## Feature engineering playbook
- **No feature engineering needed beyond tokenization**: Pretrained transformers handle raw text directly; no manual feature extraction required.
- **Stratify train/val split on 'toxic' column** (most frequent positive label) to ensure minority classes appear in both splits.
- **Class distribution to memorize**: toxic 9.58%, severe_toxic 1.00%, obscene 5.29%, threat 0.30%, insult 4.94%, identity_hate 0.88%. 89.8% of samples are all-negative.
- **Label correlations exist** (severe_toxic → toxic, insult → obscene) but explicit multi-task modeling of dependencies added complexity without proven benefit over per-label focal/BCE loss.

## Tuning insights
- **pos_weight capping is the #1 knob**: Cap at 10–50; raw frequency inversion causes gradient explosions. sqrt-dampening is a good alternative.
- **Focal loss (γ=2.0, α=0.25) works but isn't strictly necessary**: Achieved bronze (0.98652) with DistilBERT; not proven superior to weighted BCE with proper pos_weight capping on larger models.
- **Learning rate**: 2e-5 is standard; LLRD factor 0.95/layer reliable across BERT/RoBERTa/DeBERTa.
- **Epochs**: 2–4 epochs sufficient for base models; early stopping recommended. Large models (DeBERTa-large) converge in 2–3 epochs.
- **Batch size**: 32–48 depending on model size and GPU memory; mixed precision (fp16) essential for fitting large models.
- **Multi-sample dropout (5 masks, averaged)**: Improved generalization without extra inference cost.
- **R-Drop regularization**: Mentioned in designs but never properly ablated; potential unexplored gain.
- **Weighted layer pooling (layers 4–12)**: Outperforms fixed [CLS]; learnable softmax weights capture toxicity signals at multiple abstraction levels.
- **Validation every 500 steps**: Better than epoch-level validation for early overfitting detection on 159K samples.

## Open questions
- **Weighted vs. simple ensemble averaging**: Per-label or per-model learned weights (e.g., DeBERTa strong on 'threat', RoBERTa strong on 'obscene') never tested; estimated +0.0005–0.001 AUC.
- **Stacking (logistic regression on OOF predictions)** vs. averaging: Never implemented; standard technique that could squeeze marginal gains.
- **Per-label focal loss α/γ tuning**: Global γ=2.0 used everywhere; per-label tuning on threat/identity_hate could improve rare-class calibration.
- **Pseudo-labeling / self-training**: threat and severe_toxic have <500 positives; semi-supervised methods on test data untested.
- **Test-time augmentation (TTA)**: Averaging predictions from original + lowercased input for cased models — free ensemble diversity at inference, never tried.
- **Sequence length 384**: Trades compute for coverage; might help on long comments that truncate at 256, especially for rare labels embedded in longer text.
