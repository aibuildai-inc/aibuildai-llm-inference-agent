# Task Memory: nfl-player-contact-detection

## Task profile
- Type: Binary classification (contact detection per player-pair per timestep)
- Dataset: NFL tracking data, ~4.2M rows, 216 training plays, ~54K groups. ~253 player pairs per play at 10Hz sampling. Optional video (3 views × plays). Extreme class imbalance: 1.10% positive for player-player (P2P), 4.11% for player-ground (P2G).
- Metric: Matthews Correlation Coefficient (MCC), higher is better
- Medal thresholds: Bronze ≈ 0.71, Silver ≈ 0.728+, median below 0.71

## Best result
- Score: 0.72849 | Medal: Silver | Run: 20260227_230430
- Approach: Separate XGBoost classifiers (GPU `tree_method='hist'`) for P2P and P2G contacts. Monotone constraints on distance features, all-pairs nearest-neighbor distance features, formation-level aggregates (team centroid, spread). Per-contact-type MCC threshold optimization. GroupKFold by game_play.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_035725 | 0.7107 | Bronze | LightGBM baseline, separate P2P/P2G, distance+closing speed features, scale_pos_weight |
| 20260227_230430 | 0.7285 | Silver | XGBoost GPU, monotone constraints, nearest-neighbor distance features, formation aggregates |
| 20260228_091324 | 0.6708 | None | LightGBM with rolling temporal windows; regression likely from helmet feature noise + coarse thresholds |
| 20260228_194325 | 0.7145 | Bronze | LightGBM, negative downsampling (10:1), cyclical orientation encoding, Gaussian temporal smoothing |

## Proven strategies
- **Separate P2P / P2G models**: Consistently essential across all 4 runs; single model conflates different contact mechanics and imbalance rates
- **Distance-based features dominate**: Euclidean distance, closing speed, lag/lead distances are the core signal in every successful run
- **Per-contact-type MCC threshold optimization**: Gains 3–5% MCC over global 0.5 threshold; P2P threshold ~0.49, P2G ~0.54
- **GroupKFold (or StratifiedGroupKFold) on game_play**: Mandatory; 0% play overlap between train/test, naive row-level CV leaks play structure
- **GBDT models (LightGBM / XGBoost)**: Consistently outperform neural approaches on this tabular data
- **GPU acceleration**: XGBoost `tree_method='hist'` gives 2–3× speedup, enabling more tuning iterations within budget
- **Fast baseline + iteration**: LightGBM 0.70+ in 3 iterations consistently beats ambitious architectures that timeout

## Failure patterns
- **Neural nets (MLP, 1D-CNN, Bi-LSTM, GRU)**: Failed or severely underperformed in every run. MLP: MCC 0.077 (2 epochs). Bi-LSTM: 0.6755. 1D-CNN: killed by timeout (non-vectorized sequence building). Trivial-negative dominance + sequence padding complexity kills feasibility
- **Video/CNN fusion**: Killed in every run. 4–8 hour preprocessing, imperfect helmet bounding boxes, frame-to-step alignment noise (0.41–5.52× ratio), marginal gains don't justify compute
- **Stacking ensembles**: Killed; 2.5-hour training (3 base × 5 folds) infeasible in automated budget; OOF generation bug-prone
- **Helmet bounding box features**: ~37% coverage, noisy alignment; mostly a distraction vs. tracking-only features
- **Position encoding via `.cat.codes`**: Produces different codes on train vs test, silently losing signal (bug found in run 2)
- **P2G model is the persistent bottleneck**: Best P2P MCC ~0.737 but P2G only 0.508–0.560, dragging overall score down

## Feature engineering playbook
- **Core**: Euclidean distance between players, closing speed (distance derivative), relative velocity components
- **Temporal**: Lag/lead distance at ±1,2,3,5 steps; rolling window stats (mean/std/min over 1–3 steps); projected distance via velocity+acceleration extrapolation
- **Proximity**: Min distance to nearest player, count of players within 2/3/5 yards (strong for P2G)
- **Formation**: Team centroid, spread aggregates
- **Orientation**: Cyclical sin/cos encoding (period=360°) outperforms raw angles for trees
- **Monotone constraints**: On distance features in XGBoost — injects domain knowledge directly into splits (silver run)
- **Post-prediction**: Gaussian temporal smoothing exploiting ~15-step mean contact duration

## Tuning insights
- **Class imbalance**: Negative downsampling (10:1) + early stopping outperforms scale_pos_weight tuning; scale_pos_weight ~72:1 (P2P) / ~20:1 (P2G) as fallback
- **LightGBM vs XGBoost**: LightGBM faster on CPU (15 min vs 26 min for 5-fold); XGBoost GPU wins when CUDA available. XGBoost GPU can OOM with high max_bin
- **Early stopping**: P2G models stop early (15–32 iterations) due to weak signal; increase early_stopping_rounds
- **Threshold grid**: Coarse 20-point grid leaves ~0.01 MCC; fine-grained search worthwhile
- **5-fold > 3-fold CV**: Speed savings from LightGBM/GPU enable more folds for stable estimates
- **No scale_pos_weight is viable** when using downsampling (run 4 insight; avoids probability calibration drift)

## Open questions
- P2G model improvement: deceleration features, foot-plant-angle proxies from orientation change, proximity to sideline/endzone — all proposed, none tested
- Temporal smoothing / HMM post-processing to enforce contact persistence and reduce prediction flicker
- LightGBM + XGBoost ensemble across P2P/P2G: does model-family diversity add value beyond single-family tuning?
- Weighted focal loss vs. downsampling + threshold search for imbalanced classes
- Jerk (acceleration derivative) and time-to-collision features — hinted at signal but never fully tuned
- Fusing helmet proximity (bbox IoU) with tracking distance into a single contact likelihood feature
- Higher-quality hand-labeled helmet detections: would they justify compute cost, or is tracking data sufficient?
