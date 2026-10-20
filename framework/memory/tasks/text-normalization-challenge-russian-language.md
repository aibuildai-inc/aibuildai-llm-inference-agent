# Task Memory: text-normalization-challenge-russian-language

## Task profile
- Type: Sequence labeling / text normalization (token class + surface form)
- Dataset: Russian text tokens; ~88% identity-mapped, ~12% non-trivial (CARDINAL, DATE, ORDINAL, MEASURE, etc.); morphologically rich (case, gender, number)
- Metric: Token-level accuracy, higher is better
- Medal thresholds: Bronze ≈ 0.9768 (rule-only baseline landed just above); Silver achieved at 0.98649

## Best result
- Score: 0.98649 | Medal: Silver | Run: 20260301_081011
- Approach: Hybrid lookup + ML. Lookup table resolves 88% of tokens at ~0.999 accuracy; LightGBM disambiguator handles ambiguous tokens (94.14% on-occurrence); rule-based fallback for unseen tokens.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260301_081011 (Cand 1) | 0.9768 | ~Bronze | Pure rule-based baseline |
| 20260301_081011 (Cand 6 mT5) | 0.98649 test (val 0.98983) | Silver | Seq2seq — validation overfit |
| 20260301_081011 (Cand 5, winner) | 0.98649 | Silver | Lookup + LightGBM + rule fallback |

## Proven strategies
- Lookup table over training corpus resolves ~88% of tokens at ~0.999 — dominates any sophisticated alternative
- LightGBM disambiguator on ambiguous-token subset reaches 94.14%; beats seq2seq on non-identity tokens (68.1%)
- Hybrid (rules + ML + lookup) scales better than end-to-end seq2seq for this metric
- Rule-only pipeline alone clears bronze (0.9768), giving cheap safety floor

## Failure patterns
- mT5 seq2seq: high val (0.98983) but no lift on test — suspected validation leakage from lookup built on full dataset, and overfitting with limited epochs
- Unseen multi-word tokens return identity instead of normalizing: dates ("2 марта 1941 года"), measures ("3 см") — recurred across 3+ tuning rounds, unresolved
- Pure rules ceiling ~0.9768 due to Russian morphology (case inflection, gender agreement) not capturable by hand-written rules alone
- Seq2seq only 68.1% on non-identity tokens despite pretraining — insufficient signal at >2 epochs on available corpus

## Feature engineering playbook
- Build class-conditional lookup table (token → normalized form) from training data; treat as first-stage router
- Split ambiguous vs unambiguous tokens; route ambiguous to ML disambiguator with context features
- Keep a rule-based fallback specifically for CARDINAL / DATE / ORDINAL / MEASURE with explicit inflection logic
- Hold out lookup construction from validation split to avoid leakage inflating val scores

## Tuning insights
- Candidate pruning: 6 designed → 3 survived; hybrid candidate won over seq2seq despite lower val
- 5 tuner runs + 1 reviser run sufficient; diminishing returns on rule polishing after edge-case CARDINAL/DATE/ORDINAL pass
- Trust test > validation when validation pipeline touches lookup source

## Open questions
- Can pymorphy2 morphological analysis replace hand-rolled inflection rules for unseen multi-word tokens?
- Is Cand 6's val/test gap due to lookup leakage into val, or genuine seq2seq overfit? Clean split experiment needed
- Would stacking Cand 5's lookup+LGBM output with mT5 as a residual corrector push past 0.987?
