# Pattern: time-series-sequence

## When this applies
- Per-timestep regression/classification on sequential waveforms (e.g., respiratory, sensor streams)
- Tasks with control inputs and physical/structural metadata (R, C groups)
- Fixed-length breath/window sequences with cross-timestep dependencies

## Proven patterns
- Transformer with d_model=256 gives stable, fast convergence on sequential regression (ventilator-pressure-prediction best 0.2163 at 300 epochs)
- Physics-guided cross-term features drive largest single gain: CNN+GRU 0.363 → 0.229 (~37% reduction) on ventilator-pressure-prediction
- Combine base statistics (max/mean/sum/std of control input) with future-window aggregates (future_mean_u_in)
- Include first AND second-order diffs of control input to capture dynamics

## Common pitfalls
- Per-group normalization (e.g., per-(R,C)) breaks convergence catastrophically: ventilator-pressure-prediction stalled MAE 0.402, final score 16.89 vs 0.2163 baseline
- 1-fold CV hides instability when gap to medal threshold is uncertain (ventilator-pressure-prediction best run used 1-fold, no stability signal)
- Revision loops without architectural changes cycle indefinitely: 5 cycles failed to close 0.229 → 0.1364 gap (~37% short of bronze)
- Truncated reviser outputs lose actionable diagnosis (ventilator steps 5 & 9)

## Model selection guide
- Sequential regression, rich feature count (~23): Transformer d_model=256 (best: 0.2163)
- Sequential regression, physics-heavy features (~37): CNN+GRU exploits them better than Transformer (0.2292 with 37 features vs 0.2163 with 23)
- Untested but promising: Transformer + full 37-feature physics set; ensemble blend of Transformer + CNN+GRU for differing feature responses

## Feature engineering patterns
- Base: control-input stats (max, mean, sum, std), future-window means
- Physics: inverse compliance (inv_C), RC_product, control × inv_C, cumulative control × inv_C
- Dynamics: first-order and second-order diffs of control input
- Prefer 37-feature physics set over 23-feature base when architecture supports it (CNN+GRU especially)

## Tuning patterns
- Transformer: d_model=256, 300 epochs sufficient for convergence
- Use ≥3-fold CV when distance to medal threshold is uncertain; 1-fold only after stability is established
- Avoid group-level normalization; prefer per-feature scaling or learned embeddings of group labels
- Budget revision cycles against architectural diversity, not parameter tweaks, when gap exceeds ~20%
