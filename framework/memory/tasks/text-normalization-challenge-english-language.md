# Task Memory: text-normalization-challenge-english-language

## Task profile
- Type: Token-level sequence classification + normalization (semiotic class assignment → verbalization)
- Dataset: English sentences segmented into tokens; 16 semiotic classes; PLAIN/PUNCT ≈ 93% trivial baseline; heavy class imbalance
- Metric: Accuracy (higher is better), evaluated per token
- Medal thresholds: bronze ≈ 0.99038; gold achieved at 0.99762

## Best result
- Score: 0.99762 | Medal: gold | Run: 20260228_225251
- Approach: DistilBERT token classifier + class-conditional lookup table. Classifier predicts semiotic class, then lookup returns the normalization conditioned on the predicted class (not majority vote). Disambiguates tokens like "1" ∈ {DIGIT, CARDINAL, DATE, ORDINAL}.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260228_032923 | 0.98441 | none | Pure rule-based (num2words + inflect), 22 refinement attempts; validation 0.9999 but test dropped ~1.4% |
| 20260228_225251 | 0.99762 | gold | DistilBERT + class-conditional lookup replaces majority-vote lookup; fixes DIGIT (68→~90%) and ORDINAL (89→~95%) |

## Proven strategies
- Class-conditional lookup gated on predicted class: the single biggest lever (0.9967 → 0.9991 jump in internal val); resolved 311 DIGIT errors in run 2
- DistilBERT (not full BERT) sufficient for token-level semiotic classification
- Early baseline diversity across rule-based (C1), GBDT/LightGBM (C5), and BERT (C3) enabled the reviser to isolate which architecture actually helps
- Error-driven iteration via per-class accuracy breakdown to target rare classes (DIGIT, ELECTRONIC, TELEPHONE) where the metric is actually decided
- num2words + inflect reliably handle cardinal/ordinal/date verbalization once class is known

## Failure patterns
- Pure rule-based overfits validation: 0.9999 val → 0.9844 test in run 1 (1.4% inflation) — test semiotic class distribution differs from validation
- Majority-vote lookup is a hidden footgun: naïve aggregation locks wrong class for ambiguous tokens (the "1" problem)
- Pure rules and pure LightGBM both plateau ~0.9933–0.9949; reviser tweaks to handlers show diminishing returns without architectural change
- T5 seq2seq (run 1, C4) scored only 0.9958 despite 4×80GB GPUs — wrong inductive bias vs. rule/classification decomposition
- Pipeline throughput issue in run 1: candidates 3/5/6 designed but never trained (coder/aggregator incomplete); designer had pip/zip env failures

## Feature engineering playbook
- Decompose task: (1) token semiotic classification, (2) class-conditional verbalization — do NOT fuse into seq2seq
- Build per-class lookup tables from training data, but key by (token, predicted_class), not token alone
- Keep rule-based handlers as fallback for classes with deterministic structure (DATE, TELEPHONE formatting, Roman numeral ORDINAL)
- Precision on the <7% non-trivial tokens dominates the score — spend optimization budget there, not on PLAIN/PUNCT

## Tuning insights
- Token classification accuracy is the bottleneck: perfect classification + weak rules > imperfect classification + perfect rules
- DistilBERT > LightGBM > pure rules for the classifier stage on this data
- Rule iteration hits ceiling ~0.9949 on a pure-rule pipeline; switch architecture rather than keep patching corner cases
- Validation accuracy ≠ test accuracy on semiotic tasks — check class distribution parity before trusting val gains

## Open questions
- Root cause of run-1 val-to-test gap (0.9999 → 0.9844): is it class-distribution skew or rule overfit to specific training corner cases?
- Can a smaller model (BiLSTM, tiny BERT) match DistilBERT at lower train cost?
- LightGBM at 0.9943 in run 1 — which specific rules help vs. hurt when hybridized? Ablation never performed
- Candidates 3/5/6 in run 1 never executed: deprioritized or blocked by coder errors — pipeline reliability issue to diagnose
- Per-class ceiling for ELECTRONIC and TELEPHONE not yet measured in run 2
