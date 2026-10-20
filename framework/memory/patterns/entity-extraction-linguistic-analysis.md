# Pattern: entity-extraction-linguistic-analysis

## When this applies
- Span extraction or multi-label tagging from short/medium NL text
- Large or imbalanced label/output space (1-2 word spans; 10K+ tags)
- Word/token-level metrics (Jaccard, micro-F1) where boundary precision matters
- Strong class-skew baselines exist (neutral=full-text; duplicate hash lookup)

## Proven patterns
- **Task-aligned pretraining beats generic**: SQuAD2 checkpoints (deepset/*-squad2) gave +0.01 Jaccard in tweet-sentiment (0.71918 vs ~0.712 plateau); DistilBERT/BERT-base generic pretrain beat code-aware CodeBERT/SOBERT on Stack Overflow tags (Run 9).
- **Standard heads beat custom architectures**: `AutoModelForQuestionAnswering` linear head won tweet-sentiment (custom sentiment-aware heads collapsed to 0.577-0.66); flat LightXML sampled-softmax beat CascadeXML hierarchical (0.1566 collapse) in keyword extraction.
- **Cross-family ensembles break single-model plateaus**: 3-model ensemble (roberta-base/large + deberta-v3-base) pushed tweet-sentiment 0.712→0.71918 silver.
- **Class-conditional heuristics dominate**: Neutral=full-text covers ~40-90% of tweet-sentiment samples at 0.976 Jaccard; clean-body-hash duplicate override covered ~53% of keyword-extraction test (val 0.6154→test 0.8026 gold).
- **Literal/lexical OR-overrides add free F1**: tag-in-text override gave +2-3 F1 in keyword extraction; analogous to neutral heuristic — orthogonal to learned model.

## Common pitfalls
- **Custom attention/multi-head architectures regress**: tweet-sentiment custom heads dropped to 0.58-0.66; keyword-extraction 4-head fusion OOM'd / crashed (Runs 7-8 C5/C6).
- **Library/tokenizer regressions silently collapse scores**: DeBERTa-v3-large transformers 5.2.0 bug dropped [CLS]/[SEP], collapsing to 0.60-0.70.
- **Threshold/decoding grid truncation**: keyword-extraction threshold grid [0.0, 0.60] saturated at boundary, likely cost +0.05-0.08 (Run 8 0.69 vs 0.71345 bronze).
- **Validation-to-test gap is bidirectional**: Run 6 val 0.6154→test 0.8026; Runs 4-5 val 0.68→test 0.23. Always estimate heuristic-coverage pre-submission.
- **Char-level / word-boundary post-processing regresses** vs raw token offset mapping (tweet-sentiment).
- **Pipeline-state bugs kill candidates**: stale candidate context (tweet-sentiment Run 20260221), missing artifact persistence (keyword-extraction Run 9 C1 gold-repro killed).

## Model selection guide
- Short span extraction (<50 tokens) → QA-pretrained RoBERTa/DeBERTa + standard QA head; ensemble 3 cross-family for plateau-break.
- Extreme multi-label (>1K tags) → DistilBERT/BERT-base + LightXML sampled head + Asymmetric Loss (γ_neg=4.0); avoid hierarchical/cascade heads.
- Always include lexical baseline (TF-IDF+OvR or hash-lookup heuristic) for ensembling and coverage diagnostics.
- Avoid domain-specific backbones (CodeBERT, SOBERT) unless validated; generic pretrain wins.

## Feature engineering patterns
- Use tokenizer offset mapping for span recovery; skip char-aware refinement.
- Identify class-conditional shortcuts early (neutral=full-text; duplicate-hash) — measure per-class metric before tuning.
- TF-IDF: word 1-2 + char_wb 3-5, sublinear_tf, min_df=3 as non-transformer fallback.
- Title/header weighting (3:1 in keyword-extraction); preserve code/identifiers case-sensitive.

## Tuning patterns
- Linear scheduler > cosine for word-level metrics.
- Label smoothing 0.1 + multi-sample dropout stable on RoBERTa; regularization is model-specific (EMA helps DeBERTa, hurts RoBERTa-large).
- 4-fold CV preferred over 5-fold for completion reliability.
- Per-tag/per-quantile threshold calibration on large holdout (200K stable, 30K unstable); **grid range [0.0, 1.0] mandatory**.
- LLRD 0.95 + warmup prevents transformer cold-start divergence at large output spaces.
- bf16 + gradient checkpointing for batch 256+ at 10K+ labels.
- Hard majority-vote ensembling sufficient; soft/weighted untested but plausible upside.
