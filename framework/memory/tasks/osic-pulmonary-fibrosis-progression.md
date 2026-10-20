# Task Memory: osic-pulmonary-fibrosis-progression

## Task profile
- Type: Regression with uncertainty quantification (predict FVC + confidence interval)
- Dataset: 158 patients, ~6–10 FVC measurements each (~1394 rows); tabular clinical metadata (Age, Sex, SmokingStatus, baseline FVC) + CT DICOM scans (512×512); test patients entirely disjoint from train
- Metric: Modified Laplace Log Likelihood (higher/less negative = better)
- Medal thresholds: Bronze ≈ -6.8683

## Best result
- Score: -7.00675 | Medal: none | Run: 20260419_213250
- Approach: Per-patient TheilSen/OLS slopes with James-Stein shrinkage to population mean, global Huber regressor predicting slopes from clinical features, parametric horizon-scaled sigma (α + β|Δweeks|, floor=70). Last-3-visit OOF scoring. GroupKFold by Patient ID.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_055110 | -7.5245 | none | Bayesian hierarchical linear regression, empirical Bayes shrinkage |
| 20260303_040220 | -7.1727 | none | BayesianRidge + Deep Ensemble MLP (5×), sigma floor 70ml |
| 20260303_060652 | -7.0400 | none | LightGBM custom Laplace NLL objective, horizon-scaled sigma |
| 20260419_060547 | -7.3546 | none | Aggregator bug selected killed C4; real C6 test was -7.11 (stacked blend) |
| 20260419_170259 | -7.4070 | none | Quantile GBM + isotonic sigma calibration; OOF -6.659, gap 0.75 |
| 20260419_200544 | -7.0126 | none | BayesianRidge + cross-fitted residual sigma; parametric horizon σ, floor=140 |
| 20260419_213250 | -7.0068 | none | Per-patient TheilSen + James-Stein shrinkage + Huber; σ floor=70, last-3 OOF |
| 20260419_224910 | -7.0372 | none | Jackknife+ LOO ensemble (158 BayesianRidge), empirical residual σ, tau=1.15 |

## Proven strategies
- **GroupKFold by Patient ID is mandatory**: Test patients are entirely disjoint; any other split leaks and inflates OOF scores
- **Per-patient linear trend is the dominant signal**: Mean slope ~-4.36 mL/week; non-linear temporal models never improved across eight runs
- **James-Stein shrinkage is critical on sparse longitudinal data**: Shrinking per-patient OLS slopes to population mean prevents overfitting with ~9 measurements/patient
- **Bayesian / regularized methods dominate at N=158**: BayesianRidge, Huber, and regularized LightGBM consistently outperform neural nets
- **Sigma floor (70–140 mL) prevents collapse**: Without it, overconfident predictions get catastrophically penalized
- **Horizon-scaled sigma (α + β|Δweeks|) outperforms constant sigma**: Confirmed across runs 3–8
- **Parametric sigma (3–4 hyperparams) > nonparametric (isotonic) or empirical LOO**: Isotonic gap=0.75; empirical LOO gap=0.43; parametric gap=0.37–0.40. Smooth constraints generalize best at N=158
- **Single-model safety > multi-model blending**: Stacking, Platt calibration, ensemble blending, and Jackknife+ LOO (158-model ensemble) added complexity without closing the gap
- **Conservative tau/sigma inflation saturates around 1.1–1.2×**: Tau grid [1.0, 1.4] with prior 1.15 improved over default but showed diminishing returns; >1.5× risks OOF over-inflation

## Failure patterns
- **OOF-to-test gap of 0.35–0.43 is structural, not a sigma-methodology artifact**: Eight runs confirm the gap persists across parametric, empirical LOO, isotonic, and GP-native sigma approaches. This is likely distributional shift between train/test patients, not calibration error.
- **Neural networks consistently underperform**: MLP, TorchFVC, multi-quantile NN with pinball loss, direct Laplace NLL gradient descent all failed across eight runs
- **Stacking / Platt calibration / Jackknife+ adds no value**: Confirmed across runs 4–8; Jackknife+ LOO (-7.037) underperformed simpler TheilSen+shrinkage (-7.007)
- **CT image features add variance at N=158**: EfficientNet-B0 + PCA scored -7.39 (run 4); not justifiable
- **Sigma methodology churn yields diminishing returns**: After 8 runs testing 5+ sigma paradigms, no approach has closed the ~0.37 gap

## Feature engineering playbook
- **Core features**: baseline FVC, delta_weeks, Age, Sex (encoded), SmokingStatus (encoded), baseline_Percent
- **Key interaction**: baseline_FVC × delta_weeks captures per-patient decline at population level
- **Per-patient OLS/TheilSen slope+intercept**: effective when shrunk toward population prior (James-Stein or Bayesian)
- **Global slope predictor**: Huber regressor from [baseline_FVC, baseline_week, baseline_Percent, Age, Sex, SmokingStatus] → predicted slope; used as shrinkage target
- **Avoid per-patient parameters at test time**: Test patients have only 1 baseline measurement; population-level priors + interaction features are more robust

## Tuning insights
- **Sigma calibration is make-or-break AND the hardest part**: Raw FVC predictions are secondary; sigma fitting is where test performance diverges
- **Parametric sigma (α, β, floor) with grid search is current best practice**: Tractable search, forces regularization
- **Prior strength / regularization > architecture**: BayesianRidge prior, Huber epsilon, LightGBM L1/L2 reg matter more than model swaps
- **Budget allocation**: 4–6 candidates with early killing, ~11 agent calls used of 15; reserve for post-tuning revision
- **Reviser steps produced no measurable improvement** in runs 6–8; sigma multiplier revision blocked or flat

## Open questions
- **Is the ~0.37 gap due to test-patient selection bias?** Only 18 test rows from disjoint patients; bootstrap metric variance analysis could quantify whether model variance << holdout variance, making the gap irreducible without more data.
- **Per-patient heterogeneous sigma**: Current approaches use global sigma functions; patient-level metadata (smoking status, age, baseline FVC decline rate) might predict individual prediction risk.
- **Feature engineering is the unexplored frontier**: Eight runs focused on sigma methodology; new features (FVC acceleration, age×smoking interaction, lung-capacity percentile bins) are untested and may reduce FVC prediction error directly.
- **Conformalized Quantile Regression**: Dropped early in run 8 due to budget; CQR offers finite-sample coverage guarantees and might exploit the gap as distribution-shift rather than calibration.
- **Post-hoc fixed sigma inflation (1.1–1.3×)**: If gap is predictable (~0.37), a constant correction on the best C4/run-7 approach may bridge to bronze.
