# Task Memory: new-york-city-taxi-fare-prediction

## Task profile
- Type: Regression
- Dataset: ~55M rows, tabular (pickup/dropoff coordinates, datetime, passenger count, fare amount); NYC taxi trip records
- Metric: RMSE (lower is better)
- Medal thresholds: Bronze ≈ 2.92 RMSE

## Best result
- Score: 3.21 RMSE (submitted) | Medal: none | Run: 20260305_194201
- Approach: LightGBM leaf-wise with tight outlier filter, airport distance features, early stopping. Note: run 20260210_161554 scored 2.99 internal but 5.23 submitted (leakage suspected).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260227_230332 | 3.26 | none | XGBoost gpu_hist on 10M rows, 45 rich features |
| 20260302_234643 | 3.38 (5.38 sub) | none | Random split fix, RMSE objective fix; 5M sample |
| 20260302_234708 | 6.51 | none | Full 55M rows but minimal tuning (0 tuner runs) |
| 20260305_194201 | 3.21 | none | LightGBM leaf-wise, tight outlier filter, airport dist, early stop |
| 20260210_161554 | 5.23 (2.99 int) | none | XGBoost depth-wise, 3 tuner runs, max_depth=11, lr=0.025, 3000 iters |
| 20260210_221916 | 4.788 (3.174 int) | none | LGBM lr=0.05, num_leaves=63, depth=8, k-means(k=10), fare-per-mile stats; 55M rows |

## Proven strategies
- **LightGBM leaf-wise scales better than XGBoost level-wise on 55M rows** (run 7 confirmed): Candidate 1 LGBM beat XGBoost Candidate 2 on same pipeline/features; XGBoost used 998/1000 rounds but couldn't match. Earlier run 5's XGBoost-depth-wise win (2.99 int) did not submit-generalize
- **Rich feature engineering >> more data or HP tuning**: 10M rows with 45 features (3.26) beat 22M rows with 16 lean features (3.67); feature additions worth ~0.4 RMSE, tuning <0.1; run 7 tuner stuck 8.5% above target across 3 iters
- **Airport proximity features are essential**: Distance to JFK/LGA/EWR from pickup and dropoff captures flat-rate premium routes
- **K-means location clustering (k=10) adds spatial signal** (run 7): complements haversine/Manhattan for geo-regression
- **Aggressive outlier filtering**: Fare clip [2.5, 200], passenger [1, 6], NYC bbox; unfiltered 2.1% invalid coords degrade models by ~0.2 RMSE
- **Random train/val split (not temporal)**: Temporal split causes 0.3–0.4 RMSE degradation from fare tariff creep 2009–2015
- **RMSE objective must match metric**: Huber/fair loss misaligns; direct RMSE required
- **Chunked CSV + float32 + parquet caching**: Avoids OOM on 55M rows; saves ~60 min
- **GPU mandatory**: LightGBM device='gpu' or XGBoost gpu_hist enables 55M-row runs in <60min
- **Early stopping (50–100 rounds)**: Prevents overfitting across all runs

## Failure patterns
- **Internal-to-submission score gap persists (runs 5, 7: 2.99→5.23, 3.17→4.79)**: Systematic gap signals validation leakage, target encoding contamination, or train/test temporal distribution drift; must audit val split construction and consider temporal CV
- **Tuner plateau at ~8–8.5% above target after 2–3 runs** (run 7): HP tuning alone cannot close the gap; feature interactions or stacking required next
- **MLP/neural with embeddings consistently fails**: run 7 Candidate 3 plateaued at 3.260; 3.79 RMSE best across history; neural underperforms trees on tabular geo-data even at 55M samples
- **Hybrid Neural-Embedding + GBDT killed by judge** (run 7 Candidate 4): embeddings redundant with engineered distance features under 60-min budget
- **CatBoost unavailable**: Missing install blocks 3-way stacking
- **Under-using agent calls correlates with bad scores**: Run 3 used 4/10 calls → 6.51 RMSE; run 7 used 7/10 → 4.79
- **Regime-split models (airport vs non-airport)**: Classifier routing errors outweigh regime benefits

## Feature engineering playbook
- **Distance**: Haversine (primary, r≈0.82), Manhattan, straightness ratio, bearing (pickup→dropoff)
- **Airport**: Euclidean to JFK/LGA/EWR from pickup and dropoff; binary proximity flags
- **Temporal**: Hour (18–19 peak), DoW, month, year; rush-hour flag/decay; weekend; holiday; sin/cos cyclical
- **Spatial**: Lat/lon grid binning, NYC zone flags, k-means clustering (k=10 confirmed useful)
- **Interactions**: distance × hour, hour × DoW demand patterns; polynomial distance × hour, cluster × rush-hour untested
- **Target stats**: Fare-per-mile historical stats by zone/hour (run 7)
- **Passenger count**: Filter [1, 6]; weak but included

## Tuning insights
- **LGBM sweet spot (run 7)**: lr=0.05, num_leaves=63, max_depth=8, 1000 trees; baseline for future runs
- **Progressive depth/iter expansion** (run 5 XGB): max_depth 8→11, lr→0.025, iters→3000 gained ~0.04 RMSE internally but didn't submit-generalize
- **Learning rate**: 0.025–0.05 sweet spot; 0.01 untested
- **num_leaves (LGBM)**: 63 used in run 7; [128, 512] search recommended
- **reg_lambda**: [0.5, 2.0] range untested
- **Single 5–10% holdout risks submission gap** (runs 5, 7); temporal CV (train 2009–2014, val 2015) candidate diagnostic
- **Tuner diminishing returns after 2–3 runs**: Gains <0.5% per pass; stops closing target gap

## Open questions
- **Validation-submission gap recurring (runs 5, 7)**: Audit feature pipeline determinism, test-time NaN handling, temporal drift; try temporal CV as diagnostic
- **Sub-3.0 RMSE barrier on submission unbroken**: Best real score 3.21; internal <3.2 doesn't generalize
- **Feature interactions never grid-searched**: polynomial distance × hour, cluster × rush-hour, distance × hour × DoW × passenger
- **Stacking LGBM + XGBoost with linear meta-learner** (run 7 suggestion): diversity between leaf-wise and level-wise may capture complementary signal
- **Quantile/fair loss for outlier robustness**: could reduce sensitivity to 95th %ile fare outliers ($40+)
- **External features**: Weather, surge, events never incorporated; haversine explains only ~67% of variance
- **Target transforms**: Yeo-Johnson/Box-Cox untested; log inconclusive
