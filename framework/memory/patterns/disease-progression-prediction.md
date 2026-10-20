# Pattern: disease-progression-prediction

## When this applies
- Small longitudinal cohorts (N~150 patients, ~6-10 measurements each) with disjoint train/test patient splits
- Regression with uncertainty quantification (predict value + sigma); metric penalizes overconfidence (e.g., Laplace Log Likelihood)
- Sparse per-subject time series with mixed tabular clinical metadata + optional imaging

## Proven patterns
- **GroupKFold by Patient ID is mandatory**: osic-pulmonary-fibrosis-progression confirmed across 8 runs that any other split leaks; test patients are entirely disjoint
- **Per-patient linear trend dominates**: mean slope (~-4.36 mL/week) captured by OLS/TheilSen + James-Stein shrinkage to population mean (best run 20260419_213250: -7.0068)
- **Bayesian/regularized methods win at small N**: BayesianRidge, Huber, regularized LightGBM consistently beat neural nets (BayesianRidge variant -7.0126 vs MLP/TorchFVC failures)
- **Horizon-scaled parametric sigma (α + β|Δt|, with floor 70-140)**: outperformed constant, isotonic (gap 0.75), empirical LOO (gap 0.43); parametric gap 0.37-0.40
- **Global slope predictor as shrinkage target**: Huber regressor from clinical features predicts per-patient slope, used to shrink sparse per-patient estimates

## Common pitfalls
- **Neural networks fail consistently**: MLP, deep ensembles, multi-quantile NN with pinball loss, direct NLL gradient descent all underperformed (run 20260303_040220: -7.17)
- **Stacking/blending/Jackknife+ adds no value**: 158-model Jackknife+ LOO (-7.037) lost to simpler TheilSen+shrinkage (-7.007)
- **CT image features add variance**: EfficientNet-B0 + PCA scored -7.39 at N=158
- **Sigma floor omission causes catastrophic penalties**: overconfident predictions destroy Laplace LL
- **Sigma methodology churn has diminishing returns**: 5+ paradigms tested, none closed the structural OOF-to-test gap (~0.37)
- **Per-patient parameters at test time fail**: test patients have only 1 baseline; must use population priors

## Model selection guide
- N < 200 patients with sparse longitudinal → BayesianRidge or per-patient TheilSen + James-Stein shrinkage + global Huber predictor
- Need uncertainty → parametric horizon-scaled sigma (α, β, floor) with grid search; avoid isotonic/empirical LOO
- Avoid: deep nets, image-based CNN features, complex stacking, Platt calibration at this sample size

## Feature engineering patterns
- Core: baseline value, delta_time, Age, Sex, SmokingStatus, baseline_Percent
- Key interaction: baseline_value × delta_time (captures population-level decline)
- Per-patient slope+intercept only when shrunk to population prior (James-Stein/Bayesian)
- Avoid raw per-patient parameters at inference; use population priors + interactions

## Tuning patterns
- Sigma calibration is the bottleneck, not point predictions; budget tuning effort here first
- Parametric sigma with 3-4 hyperparams (α, β, floor, optional tau) via grid search beats flexible nonparametric
- Tau/sigma inflation saturates at 1.1-1.2×; >1.5× risks OOF over-inflation
- Prior strength (BayesianRidge alpha, Huber epsilon, L1/L2 reg) > architecture swaps
- Budget 4-6 candidates with early killing; reviser/post-hoc sigma multiplier revisions yielded no measurable gains across runs 6-8
- Accept structural OOF-to-test gap (~0.35-0.43) as likely distributional shift, not calibration error; consider fixed post-hoc sigma inflation rather than further methodology churn
