# Task Memory: tweet-sentiment-extraction

## Task profile
- Type: Span extraction (QA-style) on short text
- Dataset: ~24K tweets, very short sequences (~23 tokens mean), 3-class sentiment (neutral ~40-90% returns full text); 55-59% of gold spans are 1-2 words
- Metric: Word-level Jaccard similarity (higher is better)
- Medal thresholds: Silver/Bronze target ≈ 0.71705

## Best result
- Score: 0.71918 | Medal: Silver | Run: 20260302_235514
- Approach: Ensemble of three QA-pretrained transformers (deepset/roberta-base-squad2, roberta-large-squad2, deberta-v3-base-squad2). SQuAD2 pretraining + architectural diversity across base/large families yielded complementary errors; OOF 0.7204, LB 0.71918.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260221_224429 | 0.5362 | none | RoBERTa-base + SQuAD2; no tuning phase reached; stale candidate context bugs |
| 20260219_005659 | 0.7172 | ungraded | RoBERTa-large, linear head, label smoothing 0.1, standard span decoding |
| 20260301_162944 | 0.71795 | Silver | DeBERTa-V3-Large w/ standard HF QA head, 4-fold ensemble, 2 epochs/fold |
| 20260224_225105 | 0.71875 | Silver | Majority-vote ensemble of RoBERTa-base (MSD+LS) + RoBERTa-large multi-task |
| 20260302_235514 | 0.71918 | Silver | 3-model SQuAD2-pretrained ensemble (roberta-base/large + deberta-v3-base) |

## Proven strategies
- **SQuAD2 pretraining (deepset/*-squad2)**: +0.01 Jaccard lift across families; strongest single transfer signal
- **Standard `AutoModelForQuestionAnswering` head**: beats custom sentiment-aware heads by 60-130bp (custom variants fall to 0.58-0.66)
- **Cross-family ensemble (3+ models)**: consistently breaks the ~0.712 single-model plateau by +0.005-0.007
- **Label smoothing 0.1 + multi-sample dropout (5 passes)**: stable 0.712 on RoBERTa-base
- **Linear scheduler > cosine** for word-level Jaccard
- **4-fold CV preferred over 5-fold** for completion reliability without losing ensemble benefit
- **Neutral heuristic (return full text)**: required baseline for neutral class, though leaves ~0.0094 on the table

## Failure patterns
- **DeBERTa-v3-large tokenizer bug in transformers 5.2.0**: `__call__()` drops [CLS]/[SEP]; collapses scores to 0.60-0.70. Must manually construct special tokens or fall back to deberta-base (v1) / deberta-v3-base.
- **Custom sentiment-aware attention heads**: catastrophic (0.577-0.66); always revert to standard QA head
- **Word-boundary / char-level post-processing**: degrades vs. raw token-level predictions
- **Regularization is model-specific**: EMA + multi-sample dropout helps DeBERTa but regresses RoBERTa-large
- **Per-fold variance** near threshold (e.g., 0.716→0.706→0.713→0.706→0.714) signals fragile generalization
- **Stale candidate context in long pipelines** (run 20260221): state machine fails to advance; candidates 5/6 never trained
- **BIO tagging on DistilBERT**: underperforms QA-style span extraction
- **Base models (RoBERTa-base, DeBERTa-v3-base)**: plateau ~0.715; need large variants or ensembling to clear silver

## Feature engineering playbook
- Use tokenizer offset mapping for span recovery; do **not** add char-aware refinement
- Treat neutral as special class: full-text heuristic handles ~40-90% of samples at ~0.976 Jaccard
- All improvement headroom is in positive/negative span boundaries (each ~0.53-0.54 individually)
- Measure per-sentiment Jaccard before tuning — neutral is solved, others are the bottleneck
- Keep architecture simple: single linear QA head > multi-layer/multi-head designs
- Short-span problem (55-59% are 1-2 words) is unsolved; candidate lever for future gains

## Tuning insights
- Start with deepset SQuAD2 checkpoints, not raw pretrained
- RoBERTa-large > RoBERTa-base by ~0.003-0.005 but fragile without careful regularization
- DeBERTa-V3-Large with standard QA head + 4-fold/2-epoch ensemble reached 0.718
- Hard majority-vote worked; soft/weighted voting untested
- Top-K start/end logit-sum decoding is sufficient; beam/Viterbi/constrained decoding not needed
- Multi-sample dropout + top-K variants give <0.1% — drop as complexity overhead

## Open questions
- Root cause of DeBERTa-v3-large tokenizer collapse in transformers 5.2.0 — library bug vs. cache corruption?
- Does training a dedicated neutral-span predictor recover the 0.0094 heuristic gap?
- Would soft/confidence-weighted or learned-gating ensembling beat hard majority-vote?
- Can boundary-specific loss weighting or span-length-aware decoding improve 1-2 word span accuracy?
- Why does SQuAD2 transfer help only ~4pts in small-scale runs (0.5362) but ~10pts once full tuning pipeline completes?
