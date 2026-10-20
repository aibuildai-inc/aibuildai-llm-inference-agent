# Task Memory: predict-volcanic-eruptions-ingv-oe

## Task profile
- Type: Regression (time-to-eruption prediction)
- Dataset: 3,987 training segments, each with 10 sensors × 60,001 timesteps of seismic data; target is time-to-eruption in seconds (range ~6K–49M, roughly uniform over 5 orders of magnitude); sensors 2, 3, 5, 7, 8, 9 have 11–28% missing data
- Metric: MAE (lower is better)
- Medal thresholds: Bronze ≤ 4,990,000 | Gold ≤ ~3,050,000 (empirical); naive median baseline ~11.6M

## Best result
- Score: 2,469,620 | Medal: Gold | Run: 20260212_222314
- Approach: XGBoost with `reg:absoluteerror`, ~1,297 signal-processing features (statistical, rolling-window, FFT, autocorrelation, Hilbert envelope, Welch PSD, spectrogram stats), 5-fold CV with 5000 boost rounds and early stopping at 100. Feature caching eliminated recomputation overhead.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_175343 | 3,360,000 (OOF, MLP) | Ungraded | Baseline 1162-feature MLP; tree models incomplete |
| 20260212_222314 | 2,469,620 | Gold | XGBoost + signal-processing features (Hilbert, Welch, spectrogram); ~1,297 features |
| 20260228_024413 | 2,502,544 | Gold | LightGBM + 481 dense stats/FFT features; 3 tuning rounds → 2,490,200 OOF |
| 20260228_091419 | 3,053,249 | Gold | XGBoost + tsfresh-style 934→500 features; log-target transform |

## Proven strategies
- **Time-series → tabular conversion is the core pipeline**: Raw 60K-point windows → engineered features → GBDT. Feature quality dominates model choice consistently across all runs.
- **Signal-processing features outperform raw statistics**: Hilbert envelope, Welch PSD bands, spectral centroid/bandwidth/entropy, spectrogram stats consistently lift scores over mean/std/quantile baselines (best run used these heavily).
- **GBDT models (LightGBM, XGBoost) dominate**: All gold medals came from tree-based models. XGBoost with `reg:absoluteerror` and LightGBM with `regression_l1` both work; choice is secondary to features.
- **MAE-aligned objectives**: Using `reg:absoluteerror` / `regression_l1` directly optimizing the competition metric outperforms default MSE.
- **Null handling via indicators + native tree handling**: Binary null-indicator features + NaN-fill (0 or nanmean) + letting trees handle missingness natively. No complex imputation needed; dropping null-heavy sensors hurts.
- **Parallel feature extraction** (8 workers, joblib) + `.npy` caching: Essential for staying within time budgets; ~7 min for all 3,987 samples.
- **5-fold CV with early stopping** (patience=100): Prevents overfitting on the small 3,987-sample dataset; models converge in 2K–5K iterations.
- **Feature selection for small datasets**: Reducing from 900+ to ~500 features (zero-variance removal, correlation filtering at 0.98, importance-based top-K) consistently helps or is neutral.

## Failure patterns
- **Deep learning fails on this dataset**: MLP (3.36M), 1D-CNN (killed/failed), LSTM/GRU (12.6M), pure CNN (10.7M) — all underperform GBDT by 1.5–7x. 3,987 samples is too few for learned representations.
- **Too many features without selection causes overfitting**: XGBoost with 679 features scored 9.1M (run 3); feature selection is mandatory above ~500 features.
- **Linear models inadequate**: Ridge/ElasticNet with PCA scored 5.4M; cannot capture nonlinear seismic patterns.
- **XGBoost huber objective can collapse**: One run saw huber objective produce 3.17B MAE; fallback to `reg:absoluteerror` is necessary.
- **Stacking/ensemble candidates killed for complexity**: Stacking violates standalone requirements; CNN+GBDT hybrids too slow for marginal gains.
- **Aggressive downsampling of raw waveforms loses signal**: RNN/CNN approaches that downsample from 60K destroy critical temporal patterns.

## Feature engineering playbook
- **Per-sensor statistics**: mean, std, min, max, quantiles (5/25/50/75/95), skewness, kurtosis, energy, zero-crossing rate, crest factor
- **FFT features**: spectral centroid, bandwidth, dominant frequency, spectral entropy, band powers, top-20 FFT magnitudes
- **Signal-processing (high-value)**: Hilbert envelope stats, Welch PSD band powers, spectrogram statistics
- **Temporal features**: Rolling window stats at multiple scales (500, 1000, 3000, 6000 timesteps), autocorrelation at multiple lags, peak features
- **Automated features (tsfresh-style)**: Entropy, complexity estimates, extended autocorrelation — discovered non-obvious patterns that hand-crafted stats missed
- **Cross-sensor features**: Pairwise correlations (weak signal, max ~0.13–0.19), energy ratios, PCA — low value but included for diversity
- **Null features**: Per-sensor fraction missing, all-missing binary flags — structurally informative
- **Optimal feature count**: 500–1,300 features after selection; best result used ~1,297

## Tuning insights
- **Learning rate, tree depth, regularization**: Default configs scored gold; tuning squeezed ~5% improvement (2,624K → 2,490K in one run)
- **Log-target transform** (`log1p`): Used in run 4, helped gradient stability for the wide target range; worth including
- **Early stopping patience=100** is the right ballpark; models converge in 2K–5K rounds out of 5K–15K limit
- **CPU training is faster than GPU** for tabular GBDT at this dataset size; GPU useful only for XGBoost `hist` mode if needed
- **Ensemble blending** (LightGBM + XGBoost weighted average) was never fully executed but candidates with similar gold-level MAE (~2.5–3.0M) suggest averaging could push below 2.4M

## Open questions
- **Ensemble of top GBDT candidates**: Multiple runs produced gold-level LightGBM and XGBoost models with different feature sets — blending OOF predictions is the most obvious untried improvement
- **Feature pruning for XGBoost**: Correlation-based pruning from 679→300 features may recover XGBoost's potential when it overfits
- **Sensor-specific sub-models**: Sensors 1, 4, 6 are reliable; others frequently null. Training separate models per sensor-reliability group and blending may exploit independence structure
- **Pre-processing for FFT quality**: Int16 saturation/clipping artifacts may corrupt spectral features; pre-detrending or de-clipping (tanh model) untested
- **Welch PSD vs raw FFT**: Welch (smoothed) appeared in best run's features; systematic comparison with raw FFT not done
- **Log-target transform impact**: Used in one run (run 4, score 3.05M) but not in the best run (run 2, score 2.47M); unclear if it helps or hurts — needs controlled comparison
