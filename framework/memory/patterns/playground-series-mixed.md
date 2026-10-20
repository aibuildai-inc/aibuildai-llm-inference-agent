# Pattern: playground-series-mixed

## When this applies
- Synthetic Kaggle TPS-style tabular data, 0.8M–3.6M rows, mixed continuous + low-cardinality integer + occasional high-cardinality string features
- Classification (binary or multi-class), metric is AUC or accuracy
- Often contains synthetic-generation artifacts (CTGAN negatives, broken one-hots) or unusual feature types (10-char strings)

## Proven patterns
- **Heterogeneous GBDT + neural stacking** lifts beyond single-model ceilings: dec-2021 blend of XGB candidates 3+6 → 0.96325 (gold) vs best OOF 0.96251; may-2022 5-base stack → 0.99718 vs single-model ceiling ~0.996
- **5-fold stratified CV with averaging** beats single-fold across both tasks (dec-2021 lifted from 0.96246 → 0.96325; may-2022 nested CV stable at 0.997)
- **Domain-specific decomposition of unusual features** is mandatory: f_27 char-tokenization in may-2022 (essential across all bases); soil-type groupings + cyclic encodings in dec-2021 (~+0.007 lift)
- **Logit-transform base probabilities + L2 LR meta-learner** outperforms shallow GBDT meta (may-2022 stacker pattern)
- **CPU GBDT training** outperforms GPU at scale (dec-2021: 3.6M rows OOM on GPU across all GBDTs; may-2022: XGBoost GPU CUDA OOM, CPU hist 41 min)

## Common pitfalls
- **Stacking with default/untuned meta-learner regresses below bases** (dec-2021 run 1 stack 0.95954 < base 0.96246)
- **Aggressive FE saturates and regresses**: may-2022 polynomial+MI dropped Cand 3 from 0.9963→0.9951; dec-2021 also showed diminishing returns past 141 features
- **Non-fold-stratified MI/feature selection leaks** into outer folds
- **CatBoost at scale times out**: dec-2021 ~74 min/fold killed 5-fold runs
- **Setup/install hangs** on torch/xgboost without explicit pinning + `--no-cache-dir` (dec-2021)
- **Multi-pass pseudo-labeling** causes confirmation bias (may-2022 stuck at 0.9867); single-round only
- **Smoke AUC <0.55 never recovers** in full training (may-2022 Cand 2/5)

## Model selection guide
- Rows >2M + pure numeric/integer → GBDT trio (LGBM/XGB), CPU, 5-fold; skip CatBoost if per-fold budget <12h
- Rows <1M with string/high-card features → 5-base stack: LGBM + XGB + CatBoost + MLP + FT-Transformer
- Always include neural diversity (FT-T or MLP); homogeneous GBDT ensembles plateau (may-2022: 0.985–0.991)
- Meta-learner: L2 logistic regression with nested CV C-tuning, on logit-transformed base probs

## Feature engineering patterns
- **Cyclic encoding** (sin/cos) for directional features (dec-2021)
- **Character-position decomposition** for string features: positional tokens + bigrams + unique-char count (may-2022 f_27)
- **Pairwise ratios, polynomial squares, log transforms** on continuous (dec-2021: 63→141 gained ~0.007)
- **Domain groupings beat generic one-hot** (dec-2021 soil types)
- **Ternary interaction features from EDA** (may-2022: f_02×f_21 etc.)
- **Don't clean synthetic artifacts** — add indicator flags, let trees handle (dec-2021 CTGAN negatives)
- Conservative FE wins above metric saturation (>0.995 AUC)

## Tuning patterns
- **GBDT defaults**: lr=0.05, max_depth=10, ~300 boosting iterations, 5-fold stratified (dec-2021 best)
- Deeper trees + lower LR + more folds > more iterations
- Diminishing returns past ~370 boosting iterations (dec-2021)
- FT-Transformer scaling fragile: deeper/wider without meta retuning regresses (may-2022 6L/192d dropped 0.0006)
- Auxiliary masked-char reconstruction loss at 5% weight stabilizes char transformers (may-2022)
- Joint Bayesian co-tuning of full stack fails cold; warm-start from known-good config required
