# Task Memory: tabular-playground-series-dec-2021

## Task profile
- Type: Multi-class classification (7-class Forest Cover Type, synthetic CTGAN-generated)
- Dataset: ~3.6M rows × 54 features; extreme class imbalance (Class 2: 56.6%, Class 5: 1 sample); CTGAN artifacts (negative feature values, broken one-hot encoding)
- Metric: Accuracy (higher is better)
- Medal thresholds: bronze ≥ 0.95658; both runs landed gold

## Best result
- Score: 0.96325 | Medal: gold | Run: 20260228_193737
- Approach: XGBoost with 141 engineered features (pairwise ratios, polynomial squares, cyclic encoding, log transforms, soil groupings, GAN artifact indicators) + 5-fold stratified CV averaging. lr=0.05, max_depth=10, CPU training, 280–370 boosting iterations. Final blend of candidates 3+6 exceeded all individual OOF scores (0.96251).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260228_020614 | 0.96246 | gold | Baseline GBDT trio; CatBoost ordered boosting picked (CV 0.96167) |
| 20260228_193737 | 0.96325 | gold | +141 engineered features, XGBoost 5-fold CV, deeper trees (depth 10), blend of cands 3+6 |

## Proven strategies
- GBDT family (LightGBM, XGBoost, CatBoost) all clear bronze (0.956) reliably on this dataset
- 5-fold (or 7-fold) stratified CV + averaging consistently beats single-fold predictions
- CPU training outperforms GPU on 3.6M rows (GPU OOM on all GBDTs)
- Aggressive feature engineering (63→141 features) gained ~0.007; domain features (soil groupings, cyclic encodings) > generic polynomial/ratio
- Deeper trees (depth 10) + lower LR (0.05) + more folds > more boosting iterations
- Blending diverse reliable base learners (cands 3+6) lifted LB above any single OOF

## Failure patterns
- CatBoost impractical at scale: ~74 min/fold, 2000+ iterations needed; times out on 5-fold CV — skip when per-fold budget <12h
- Stacking ensembles underperform base learners (0.95954 in run 1) when meta-learner is default/untuned or when a base learner times out (run 2)
- GPU training OOM across LightGBM/XGBoost/CatBoost at 3.6M rows — forces CPU fallback
- Setup/install hangs (torch, xgboost) on network — required explicit pinning (`torch==2.5.*`) and `--no-cache-dir`
- MLP and Random Forest candidates abandoned at design phase; never worth implementing given GBDT dominance on this size/shape

## Feature engineering playbook
- Pairwise ratios between continuous features
- Polynomial squares of key numerics
- Cyclic encoding (sin/cos) for directional/aspect features
- Log transforms on skewed numerics
- Domain-specific soil-type groupings (beats generic one-hot)
- GAN artifact indicator flags (negative values, broken one-hot patterns) — don't clean, let trees handle
- 141 engineered features is a proven sweet spot; baseline 63 is too sparse

## Tuning insights
- lr=0.05, max_depth=10, 5-fold stratified CV, ~300 boosting iterations is a strong default
- Diminishing returns past ~370 iterations
- LightGBM best single-fold: 511 leaves, lr=0.03 → 0.96193
- XGBoost edged LightGBM by ~0.001 at matched feature count (0.96251 vs 0.96173)
- CatBoost ordered boosting gives <1 point edge over XGB/LGB — not worth the compute cost
- Further tuning likely needed to break 0.963 ceiling

## Open questions
- Can stacking be made to work here if CatBoost base learner is replaced with a faster alternative, or if meta-learner is properly tuned (not default logistic)?
- Why did final LB (0.96325) exceed best OOF (0.96251)? Blending mechanics vs post-hoc calibration not fully audited
- Class 5 (single sample) handling under stratified k-fold — edge case not explicitly verified
- Is there headroom above 0.963 via pseudo-labeling, NN blends, or target encoding that wasn't attempted?
