# Pattern: molecular-biochemical-prediction

## When this applies
- Small-data regimes (N ≈ 2k samples) with structured molecular inputs (crystals, RNA sequences)
- Multi-output regression with metric-aligned target transforms (RMSLE, MCRMSE)
- Tasks where raw tabular/sequence features have a hard performance ceiling without domain-specific structural features

## Proven patterns
- **Domain-structural features are the largest lever**: geometry features from .xyz files moved nomad2018 from 0.069 → 0.049 (Gold); BPP matrices moved stanford-covid from 0.339 → 0.216 (~65% reduction). Sequence/tabular-only approaches plateau well below medal thresholds.
- **Data quality filtering beats noisy-data weighting**: stanford-covid SN_filter=1 hard cutoff was the single biggest gain (0.67→0.24). Noise-aware weighting/L2 schemes consistently underperform hard filtering.
- **Diverse-architecture ensembles at convergence yield 1–3% gains**: stanford-covid 4-model blend (BiGRU/CNN+GCN/Transformer/Structured) dropped 0.232 → 0.216; nomad2018 LightGBM+XGBoost 0.6/0.4 blend stabilized Gold-tier scores.
- **Stratified/k-fold CV (5–10 fold) on the dominant categorical/structural axis** is required to detect leakage and stabilize per-position or per-spacegroup estimates.
- **Metric-aligned target transforms are mandatory**: log1p for RMSLE (every nomad2018 medal run), target outlier clipping for MCRMSE (preserved 37% of data on stanford-covid).

## Common pitfalls
- **Deep/pretrained architectures on N≈2k**: GNNs (nomad2018, abandoned every run), pretrained RNA LMs (stanford-covid 0.497–0.699), Transformers without preprocessing (stanford-covid 0.669) — complexity ≠ capability.
- **Complex stacking ensembles overfit catastrophically**: nomad2018 CV 0.029 → test 0.082 (2–3× gap). Simple weighted blends beat 2-level meta-learners.
- **Dense/redundant descriptors inflate dimensionality**: Coulomb matrix eigenvalues, 300-bin RDF histograms, full Behler-Parrinello sets correlate with simpler features and overfit.
- **Pipeline stalls**: 3/11 nomad2018 runs died before evaluation; robust initialization and early sanity-check submissions are critical.
- **Val/LB distribution mismatch**: stanford-covid run 1 (0.193 val → 0.339 LB) from training on noisy data that doesn't match high-SNR test set.

## Model selection guide
- N < 3k + tabular/structural features → **LightGBM/XGBoost** (every nomad2018 medal)
- N < 3k + sequence + per-position targets → **BiGRU/CNN with structural features** (stanford-covid Bi-GRU beat Transformers)
- Single-model ceiling reached → **diverse-architecture weighted ensemble** (not stacking)
- Avoid: GNNs, pretrained domain LMs, deep Transformers — consistently fail at this scale

## Feature engineering patterns
- **Tier 1 (mandatory)**: domain-structural features (BPP matrices via ViennaRNA; coordination numbers + bond angles via PBC minimum-image)
- **Tier 2 (high impact)**: element/position n-grams along nearest-neighbor chains; per-element-pair statistics
- **Tier 3 (physics-informed)**: electronegativity, ionic radii, lattice distortion (nomad2018 0.070 → 0.063)
- **Cap curated geometry features at ~20–30**, or aggressively regularize if using 200+
- **Mask loss to scored positions only** while feeding full sequence (stanford-covid 0–67 of 107)

## Tuning patterns
- **Per-target hyperparameters**: easier targets need heavier regularization (nomad2018 formation_energy: depth≤8, reg_lambda≥3); harder targets need deeper trees with moderate reg (bandgap: depth 8–11, reg_lambda 0.8–2.0)
- **Early stopping (50–200 rounds, default 100)** with n_estimators=500–1000 — let it decide
- **Multi-seed averaging (3–5 seeds)** meaningfully reduces variance in small-data regime
- **Tuner plateaus after 3–5 rounds**: shift budget to feature engineering and ensembling
- **Preprocessing > architecture > hyperparameters**: once data is clean and structural features are in, model family converges (stanford-covid all architectures hit 0.23–0.24 ceiling)
