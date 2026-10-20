# Task Memory: ventilator-pressure-prediction

## Task profile
- Type: Regression (per-timestep pressure prediction on breath sequences)
- Dataset: Sequential respiratory waveforms with R (resistance) and C (compliance) group labels, u_in control input
- Metric: dice-hausdorff-combo (lower is better, per run log)
- Medal thresholds: Bronze ~0.1364 (targeted, not reached)

## Best result
- Score: 0.2163 | Medal: ungraded | Run: 20260308_190724
- Approach: Transformer with d_model=256, 23 engineered features, 1-fold CV, 300 epochs. Converged fast and stable but only narrowly beat a CNN+GRU with a richer 37-feature physics-guided set (0.2292).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260308_190724 (C1) | 16.89 | ungraded | Per-(R,C) group normalization — broken |
| 20260308_190724 (C3) | 0.2292 | ungraded | CNN+GRU + 37 physics features (baseline 0.363 → 0.229) |
| 20260308_190724 (C2, best) | 0.2163 | ungraded | Transformer d_model=256, 23 features, 1-fold, 300 epochs |

## Proven strategies
- Transformer with d_model=256 converges fast and stable on this sequential regression
- Physics-guided cross-term features drive the biggest gains (≈13% over base features on CNN+GRU; 0.363 → 0.229)
- Extended feature set (~37 features) including max/mean/sum/std of u_in, future_mean_u_in, inv_C, u_in × inv_C, cum_u_in × inv_C, RC_product, second-order diff u_in_diff2

## Failure patterns
- Group normalization per-(R,C) universally breaks convergence (stalled at MAE 0.402 after 200 epochs; final 16.89)
- Revision loops cycling without closing the gap: 5 revision cycles never bridged 0.229 → 0.1364 bronze target (missed by ~37%)
- Reviser outputs truncated (steps 5 & 9), losing actionable diagnosis
- 1-fold CV on the best candidate provided no stability signal

## Feature engineering playbook
- Base: u_in stats (max, mean, sum, std), future_mean_u_in
- Physics: inv_C, RC_product, u_in × inv_C, cum_u_in × inv_C
- Dynamics: u_in first and second-order diffs (u_in_diff2)
- Stick to 37-feature physics set over 23-feature base when architecture allows

## Tuning insights
- Transformer d_model=256 > smaller variants at these feature counts
- 300 epochs sufficient for Transformer convergence
- CNN+GRU benefits disproportionately from physics features vs. Transformer
- 3-fold CV worth the cost for robustness when gap to medal is uncertain; 1-fold hides instability

## Open questions
- Does Transformer (d_model=256) + full 37-feature physics set exceed 0.2163? Untested combination.
- Per-breath R, C distribution: would per-breath auxiliary prediction replace the broken group-normalization approach?
- What architectural or loss change is needed to close the 0.2163 → 0.1364 bronze gap (37% improvement)?
- Would deeper ensembling (Transformer + CNN+GRU blend) exploit their differing feature responses?
