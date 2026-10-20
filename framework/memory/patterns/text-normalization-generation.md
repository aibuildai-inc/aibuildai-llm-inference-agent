# Pattern: text-normalization-generation

## When this applies
- Token-level normalization with skewed class distribution (~88-93% identity/trivial tokens)
- Multiple semiotic classes requiring class-specific verbalization (CARDINAL, DATE, ORDINAL, MEASURE, etc.)
- Token-level accuracy metric where the <12% non-trivial tail decides medal placement
- Evidence: English (16 classes, 93% PLAIN/PUNCT trivial) and Russian (88% identity-mapped)

## Proven patterns
- **Decompose into (classify → verbalize) — never end-to-end seq2seq**: English DistilBERT+lookup hit 0.99762 (gold) vs T5 at 0.9958; Russian hybrid hit 0.98649 (silver) vs mT5 with no test lift
- **Class-conditional lookup keyed by (token, predicted_class)**: biggest single lever in English (0.9967→0.9991 internal val); naïve majority-vote lookup is a footgun on ambiguous tokens like "1"
- **Lookup table from training corpus dominates**: resolves ~88% Russian tokens at ~0.999 accuracy; resolves trivial English classes similarly
- **Rule-based fallback for deterministic classes** (DATE, TELEPHONE, Roman ORDINAL, MEASURE inflection) — cheap floor that alone clears bronze in Russian (0.9768)
- **Diverse early candidates** (rules + GBDT + BERT/seq2seq) so the reviser can identify which architecture actually moves the needle

## Common pitfalls
- **Validation overfit**: English pure-rule run inflated 0.9999 val → 0.9844 test (1.4% gap); Russian mT5 hit 0.98983 val but no test lift — likely lookup leakage into val split
- **End-to-end seq2seq underperforms**: T5 (English 0.9958) and mT5 (Russian 68.1% on non-identity) both lose to hybrid decomposition despite heavy pretraining/compute
- **Pure rule-based plateaus**: ~0.9933-0.9949 English, ~0.9768 Russian — patching corner cases yields diminishing returns; switch architecture instead
- **Unseen multi-word tokens** silently return identity (Russian dates, measures) — recurring failure across tuning rounds
- **Majority-vote lookup** locks the wrong class for polysemous tokens

## Model selection guide
- Need bronze floor fast → pure rule-based with num2words/inflect (English) or pymorphy + hand rules (Russian)
- Need silver/gold → hybrid: lookup (token, class) → ML disambiguator (LightGBM or DistilBERT) → rule fallback
- Token classifier: DistilBERT > LightGBM > rules for accuracy; LightGBM if compute-constrained or only ambiguous-subset routing needed
- **Avoid**: T5/mT5 end-to-end — wrong inductive bias for this metric

## Feature engineering patterns
- Build (token, predicted_class) → normalized_form lookup; never key by token alone
- Hold out lookup construction from validation split to prevent leakage-inflated val scores
- Split tokens into ambiguous vs unambiguous; route only ambiguous through ML
- Per-class accuracy breakdown drives iteration — target DIGIT/ELECTRONIC/TELEPHONE (English) and CARDINAL/DATE/MEASURE (Russian) where score is decided
- Spend optimization budget on the <7-12% non-trivial tail, not PLAIN/PUNCT

## Tuning patterns
- Trust test > validation when val pipeline touches lookup source (Russian Cand 6 lesson)
- Rule iteration ceiling reached fast (~5 tuner runs); after that, architectural change > more rule tweaks
- Classifier accuracy is the bottleneck: perfect classification + weak rules beats imperfect classification + perfect rules
- Prune candidates aggressively: Russian 6→3 survivors, hybrid won despite lower val than mT5
