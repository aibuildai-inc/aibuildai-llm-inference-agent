# Task Memory: nomad2018-predict-transparent-conductors

## Task profile
- Type: Multi-output regression (two targets: formation_energy_ev_natom, bandgap_energy_ev)
- Dataset: 2,160 train / 240 test samples; 11 tabular features (lattice params, spacegroup, composition percentages for Al/Ga/In/O) + optional geometry.xyz files (10–80 atoms/structure, periodic crystals); 6 spacegroup categories
- Metric: Mean column-wise RMSLE (lower is better)
- Medal thresholds: Bronze ≤ 0.06582; Gold achieved at ~0.049–0.053

## Best result
- Score: 0.04899 | Medal: Gold | Run: 20260228_055601
- Approach: LightGBM with crystal-graph-inspired geometry features (nearest-neighbor coordination at multiple radii, element-type n-grams along atomic chains, local environment signatures, M-O-M angle statistics) combined with tabular features. Two independent regressors optimizing RMSE on log1p targets via 5-fold spacegroup-stratified CV. Minimum image convention for periodic boundary handling.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260209_232946 | ungraded | — | GB + RDF features implemented, not evaluated |
| 20260209_235044 | 0.0594 | Bronze | NN with spacegroup embedding + dual output heads |
| 20260210_002329 | ungraded | — | Pipeline stalled after design phase |
| 20260210_003016 | ungraded | — | Pipeline stalled after design phase |
| 20260210_003718 | 0.1345 | None | Chained XGBoost; bandgap overfitting or metric bug in submission |
| 20260210_011114 | 0.0691 | None | XGBoost + engineered features; 5% above bronze |
| 20260210_090625 | 0.0634 | Bronze | XGBoost with target-specific HPs + physics features + 5-seed ensemble + Ridge meta |
| 20260228_023826 | 0.0825 | None | Stacked ensemble overfit (CV 0.029 → test 0.082); but Cand 1 LightGBM hit 0.0531 |
| 20260228_055522 | ungraded | — | All 6 candidates failed before evaluation |
| 20260228_055601 | **0.04899** | **Gold** | LightGBM + crystal-graph n-grams + geometry features |
| 20260228_091222 | 0.05556 | Gold | LightGBM+XGBoost multi-seed ensemble + Coulomb/RDF features |
| 20260319_214744 | 0.04958 | Gold | LightGBM+XGBoost 0.6/0.4 blend + 225 geometry features, 5 tuner rounds |

## Proven strategies
- **Geometry features from .xyz files are essential for gold.** Tabular-only tops out ~0.059–0.069; geometry features (coordination numbers, NN distances, bond angles, element n-grams) push to 0.049–0.053. This is the single largest lever.
- **LightGBM/XGBoost dominate** on this 2,160-sample tabular+geometry task. Every medal-winning run used gradient boosting; neural nets consistently underperformed (MLP: 0.06+, NN: 0.059 at best).
- **Log1p target transformation is mandatory.** Directly aligns MSE loss with RMSLE evaluation metric. Every successful run used this.
- **Target-specific hyperparameters:** Formation energy (easier, ~0.03–0.04 RMSLE) needs heavier regularization (reg_lambda=3+, depth≤8); bandgap (harder, ~0.06–0.09 RMSLE) needs deeper trees (depth=8–11) and moderate regularization.
- **Spacegroup-stratified 5-fold CV** prevents distribution leakage and stabilizes estimates. Spacegroup is the strongest categorical feature; 6 classes with unbalanced sizes.
- **Multi-seed averaging (3–5 seeds)** reduces variance meaningfully in the small-data regime.
- **Simple ensembles (LightGBM + XGBoost weighted average)** beat complex stacking. Optimal weights ~0.5–0.6 LightGBM / 0.4–0.5 XGBoost.
- **Periodic boundary handling (minimum image convention)** is non-negotiable for correct distance/coordination features.

## Failure patterns
- **GNN / geometric deep learning always fails under time budget.** Graph construction with PBC is slow (>20s/structure in naive loops); training exceeds 5–20 min budget. Abandoned in every run that attempted it.
- **Complex stacking ensembles overfit catastrophically.** CV-test gap of 2–3× observed (e.g., CV 0.029 → test 0.082). Two-level ensembles with 5+ base learners bleed signal on 2,160 samples.
- **Dense geometry descriptors (65+ features) overfit.** RDF histograms (300 features), full Coulomb matrix eigenvalues, and Behler-Parrinello symmetry functions correlate heavily with simpler features and inflate dimensionality. Keep geometry features under ~20–30 curated descriptors, or use careful regularization if using 200+.
- **Chained multi-output regression is fragile.** Using formation energy predictions as bandgap input (run 20260210_003718) showed promise in CV but produced 0.1345 on test—likely information leakage or metric computation bug.
- **Pipeline stalls:** 3/11 runs died before producing any evaluation (design phase hangs, manager failures). Robust pipeline initialization is critical.

## Feature engineering playbook
- **Tier 1 (essential):** Lattice volume from vectors/angles, composition percentages (Al/Ga/In), spacegroup (target-encoded or embedded), atomic density
- **Tier 2 (geometry, high impact):** Per-element-pair coordination numbers at multiple cutoff radii (2.0–5.0 Å), mean/std nearest-neighbor distances by element pair, O-Metal-O and Metal-O-Metal bond angles, per-element atom counts from .xyz
- **Tier 3 (geometry, gold-tier):** Element-type n-grams along NN chains (unigrams/bigrams/trigrams), local environment signatures, fractional coordinate statistics, oxygen coordination distributions, lattice anisotropy ratios
- **Tier 4 (physics-informed):** Electronegativity differences, ionic radii, lattice distortion metrics — delivered measurable gains (0.070 → 0.063 in one run)
- **Avoid/limit:** Coulomb matrix eigenvalues (marginal gain vs. complexity), RDF histograms at 300+ bins (overfit risk), polynomial composition interactions (al×ga, cubic terms—mostly noise on this dataset)
- **Distance cutoffs are domain-critical:** Metal-O bonds ~2.0 Å, metal-metal ~3.5 Å; mistuned cutoffs destroy graph topology

## Tuning insights
- **Formation energy HP regime:** lr=0.02–0.06, max_depth=6–8, reg_lambda=2.0–3.5, conservative regularization
- **Bandgap HP regime:** lr=0.015–0.08, max_depth=8–11, reg_lambda=0.8–2.0, moderate regularization with deeper trees
- **Early stopping:** 50–200 rounds; 100 is a good default. Prevents overfitting on 2,160 samples.
- **Subsample/colsample:** 0.6–0.8 range combats feature dominance in high-dimensional geometry feature sets
- **Tuner typically plateaus after 3–5 rounds.** Diminishing returns beyond that; switch to ensemble/revision strategy.
- **Aggressive HP reduction (high lr + deep trees + low regularization) causes catastrophic bandgap overfitting.** Always monitor per-fold and per-target RMSLE, not just aggregate.
- **n_estimators=500–1000 with early stopping** is safe; let early stopping decide actual count.

## Open questions
- **Bandgap bottleneck persists across all runs** (~0.06–0.09 RMSLE vs. 0.03–0.04 for formation energy). Is this inherent task difficulty or a feature gap? Target-specific feature selection/engineering pipelines untested.
- **Voronoi-based NN graphs** vs. distance-cutoff heuristics never tested; could be more robust for coordination features.
- **Pre-built crystal descriptor libraries** (SOAP, ACSF via dscribe/pymatgen) never successfully used due to environment constraints; could unlock richer representations.
- **Physics-aware constraints** (monotone_constraints, interaction_constraints in LightGBM/XGBoost) unexplored; forcing formation_energy monotonicity in composition could reduce noise.
- **Per-target feature selection** (L1/Boruta) could reduce overfitting when using 200+ geometry features—never systematically tested.
- **CatBoost with native categorical handling** showed promise in one run (0.054 per-spacegroup models) but was not combined with best geometry features.
