# Task Memory: spaceship-titanic

## Task profile
- Type: Binary classification (predict `Transported` boolean)
- Dataset: ~8,700 training samples, 13 raw columns, mixed tabular (categorical + numerical). Structured string fields (Cabin, PassengerId) encode latent features. Near-perfect 50/50 class balance.
- Metric: Accuracy (higher is better)
- Medal thresholds: Bronze ≥ 0.80967, Silver ≥ ~0.815 (inferred)

## Best result
- Score: 0.8536 | Medal: ungraded (likely gold-range) | Run: 20260311_230243
- Approach: 5-fold CV ensemble of KNN-based model averaging all fold predictions. KNN imputation (k=5), RobustScaler normalization, 59 engineered features from 13 raw columns. Balanced predictions (52/48 split).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260208_214657 | — | — | Failed: 0 candidates, setup error |
| 20260208_214757 | 0.77241 | none | Shallow MLP + embeddings, cabin/group features |
| 20260208_220559 | 0.81494 | silver | Switched to XGBoost/LightGBM, same feature set |
| 20260311_174635 | — | — | Failed: incomplete/corrupted run |
| 20260311_174754 | — | — | Failed: 6 designs, 0 agent calls executed |
| 20260311_205801 | — | — | Failed: blocked at env init / data download |
| 20260311_230243 | 0.8536 | ungraded | KNN model, 5-fold ensemble, 59 features, KNN imputation |

## Proven strategies
- **Tree-based models dominate neural nets** on this dataset. XGBoost/LightGBM (0.8149) crushed MLPs (0.772) and Transformers (0.50). Shallow trees (depth ~6) hit the sweet spot for ~8.7k samples.
- **Cabin decomposition**: Parsing Cabin into deck/room_number/side consistently provides strong signal across all successful runs.
- **PassengerId group features**: Extracting group ID, group size, and per-group spending aggregates (mean/max/sum) correlates with transportation outcome.
- **5-fold CV ensemble averaging all folds** (not cherry-picking best) drove the top score of 0.8536.
- **KNN imputation (k=5)** outperformed mean/median imputation in the best run.
- **RobustScaler** for numerical normalization retained across train/test splits.
- **Feature engineering >> model complexity**: Expanding 13→59 features via domain-aware preprocessing dominated; deeper/fancier architectures couldn't compensate for weak features.

## Failure patterns
- **Neural nets consistently underperform**: Deep Tabular NN (0.8069), shallow MLP (0.772), Transformer (0.50). Insufficient data for NN complexity.
- **Transformer catastrophic failure**: Candidate 3 in run 20260208_214757 scored 0.50 (random). Likely architecture collapse or hyperparameter mismatch on small tabular data.
- **Pipeline initialization failures**: 3 of 7 runs (43%) failed before producing any candidates—env setup, conda deps, Kaggle API, or manager init issues.
- **Fold model persistence bugs**: Training saved only fold 0 but inference averaged all 5 folds → semantic mismatch. Must ensure training saves all fold models when using ensemble inference.
- **Missing dependency chain**: pyyaml, optuna required mid-run conda fixes.
- **Hybrid/complex candidates killed**: Neural-boosting hybrids and Transformers consistently killed by judge for budget/feasibility risk.

## Feature engineering playbook
- **Cabin parsing**: Split `Cabin` → `deck` (categorical), `room_number` (numerical), `side` (binary port/starboard). Deck encodes spatial patterns.
- **PassengerId structure**: Extract `group_id` (first 4 digits), compute `group_size`, aggregate spending stats per group.
- **CryoSleep × spending interactions**: Domain rule: CryoSleep=True → zero spending. Engineer anomaly flags for violations. Soft learning underperforms explicit encoding.
- **Spending aggregates**: Total spending, per-category ratios, zero-spending flags across RoomService/FoodCourt/ShoppingMall/Spa/VRDeck.
- **Group-level aggregates**: Per-group mean/max/min spending, destination heterogeneity, age statistics.
- **Target expansion**: 13 raw → 59 features achieved best result. Domain-aware expansion matters more than model choice.

## Tuning insights
- **XGBoost/LightGBM sweet spot**: max_depth=6, learning_rate=0.05, early stopping ~50 rounds on 10-20% validation holdout.
- **AdamW + cosine annealing warmup** worked for MLP but still lost to trees.
- **Label smoothing (ε=0.1)** improved MLP calibration but didn't close gap with trees.
- **Stratified splits essential** given balanced classes to preserve distribution.
- **5-min training budget was never the bottleneck**—algorithmic/feature gaps dominated.
- **No tuner or reviser runs were ever successfully executed** across all runs; potential untapped gains.

## Open questions
- Can stacking ensemble of trees + KNN improve beyond 0.8536? No multi-family ensemble tested successfully.
- Would explicit polynomial/crossed features (Age × CryoSleep, Destination × HomePlanet) push past 0.86?
- Is group-wise imputation (impute missing values using same-group statistics) better than KNN imputation?
- Why do 43% of runs fail at initialization? Pre-stage data and pin conda environments.
- Is the 0.8536 CV estimate optimistic or pessimistic vs. held-out leaderboard?
- Tuner/reviser modules never ran—could automated hyperparameter search unlock further gains on the best candidate?
