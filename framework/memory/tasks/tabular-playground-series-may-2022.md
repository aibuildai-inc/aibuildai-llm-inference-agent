# Task Memory: tabular-playground-series-may-2022

## Task profile
- Type: Binary classification (balanced ~51/49)
- Dataset: 800k rows, mixed tabular — 16 continuous, 14 low-cardinality integers, 1 near-unique 10-char string (f_27)
- Metric: ROC-AUC (maximize)
- Medal thresholds: Bronze 0.99818

## Best result
- Score: 0.99718 | Medal: none | Run: 20260419_162421
- Approach: L2 logistic regression stacker over 5 heterogeneous bases (LightGBM 0.9949, FT-Transformer 0.9960, XGBoost 0.9881, CatBoost 0.9873, MLP 0.9905) with f_27 decomposed into ch_0..ch_9 character tokens. Nested stratified CV for meta-learner C-selection; logit-transformed base probabilities.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260418_194956 | 0.99659 | none | ResNet-MLP w/ FT-Transformer-style tokenization; 2 GBDTs killed by config.py refusal |
| 20260418_210053 | 0.99526 | none | 3-model stack (LGBM+XGB+MLP), f_27 tokenization+bigrams |
| 20260419_162421 | 0.99718 | none | 5-model stack (+FT-Transformer+CatBoost), nested CV meta, logit transform |
| 20260420_193448 | 0.99660 | none | Deeper FT-T (6L/192d) + pseudo-labeling + ternary region features; regressed vs prior |
| 20260422_163322 | 0.99383 | none | Char-Transformer two-tower (6L/d128) + numeric MLP + masked-char aux loss (0.99505 CV); mlebench regressed |

## Proven strategies
- **f_27 character-level decomposition**: 10 positional tokens + 9 bigrams + unique-char count; essential across all model families. Whole-string / generic hashing strictly inferior.
- **Heterogeneous 5-model stacking**: LGBM + XGB + CatBoost + MLP + FT-Transformer; GBDT–neural mix critical. Single-model caps ~0.986; stacking lifts to 0.997+. Replicable anchor (~0.995 CV floor).
- **FT-Transformer as top single base**: 0.9960 AUC; attention over f_27 tokens finds interactions tree splits miss. Scaling depth/width fragile without meta-retuning.
- **Char-Transformer two-tower**: 6L/d128 transformer on f_27 + numeric MLP matched 5-base stack at 0.99505 CV. String features are towers to route early, not late-embedded features.
- **Auxiliary masked-char reconstruction loss** (5% weight): implicit regularization for char transformer; outperforms explicit dropout/L2 inflation in two-tower fusion.
- **Logit transformation before stacking** + **nested stratified CV meta-learner** (L2 LR, tuned C): stable, outperforms shallow LGBM meta.
- **Ternary region interaction features**: f_02×f_21, f_05×f_22, f_00+f_01 (Kaggle EDA); not discoverable from univariate stats.
- **Pseudo-labeling (single-round soft)**: +0.0005–0.001 per base with strict fold separation; multi-pass narrowing confidence never lifted.
- **Smoke phase triage is reliable**: candidates with <0.55 AUC on 2k-sample smoke never recovered in full training.
- **No class balancing needed**: ~51/49 split converges cleanly.

## Failure patterns
- **Aggressive FE regresses at saturation**: polynomial expansion + MI filtering dropped Cand 3 from 0.9963→0.9951 CV. Above 0.995 AUC, feature correlation interference and selection bias dominate; conservative FE wins.
- **Non-fold-stratified MI filtering**: MI ranked on all-data leaks selection bias into outer folds; must compute per-fold and intersect top-K.
- **Joint end-to-end Bayesian co-tuning**: 5-base+meta joint search space too large without warm-start; Cand 2 smoke AUC 0.53 (random).
- **Large polynomial feature grids**: Cand 4 (30 numeric + polynomials + GBDT meta) OOM/compute-blocked.
- **GATE / differentiable-tree standalones**: Cand 5 smoke AUC 0.47; complexity without POC not viable under budget.
- **Multi-pass pseudo-labeling**: confirmation bias / teacher overfit; Cand 6 stuck at 0.9867. Single-round PL only.
- **Deeper FT-T without meta retuning**: 6L/192d + PL regressed stack by 0.0006.
- **OOF label leakage / stale fold caches**: always assert OOF AUC <1.0 and fold sizes after mode transitions.
- **XGBoost GPU hist**: CUDA OOM at 2GB free; CPU hist fallback (41-min wall time).
- **Homogeneous GBDT ensembles**: plateau at 0.985–0.991; require neural diversity.
- **Sub-agent refuses to edit config.py** misinterpreting system reminder — killed candidates in 3/4 runs (20260418_194956); requires explicit pre-authorization via task config.
- **Bronze (0.99818) never reached**; best gap 0.0010.

## Feature engineering playbook
- **f_27 (10-char string)**: 10 positional features (char→int), 9 consecutive bigrams, unique-char count. Identical tokenization across all bases for fold consistency.
- **Ternary region features**: f_02×f_21, f_05×f_22, f_00+f_01 interaction terms (Kaggle EDA).
- **Numeric features**: sign indicators, threshold crossings, pairwise interactions on originals. Keep conservative — polynomial grids regress at >0.995.
- **Untested extensions**: fold-stratified MI-filtered polynomials, trigrams on f_27, learned char embeddings, intra-string cross-position attention.

## Tuning insights
- **Single-model ceilings**: LGBM ~0.9949, FT-T ~0.9960, XGB ~0.9881, CatBoost ~0.9873, MLP ~0.9905, Char-Transformer+numeric-MLP two-tower ~0.99505.
- **Stacking ceiling so far**: L2 LR on 5-base logits → 0.99718 (best CV/mlebench consistent).
- **Two-tower parity**: Char-Transformer two-tower matches 5-base stack in CV (~0.99505) but mlebench 0.99383 suggests less robust to distribution shift than heterogeneous ensemble.
- **Aux-loss weight**: 5% masked-char reconstruction stable; untested sweep.
- **Smoke→full fidelity**: weak smoke (<0.55) never recovers; strong smoke (>0.99) holds.
- **Total: 15 tuner runs + 2 reviser runs across 4 runs**; architecture diversity saturated.

## Open questions
- Bronze gap 0.001. Char-Transformer two-tower (0.99505 CV) + 5-base stack (0.99502 CV) soft ensemble or stack-of-stacks untested; correlated but may add 1–3 bp.
- **Fold-stratified MI filtering** for polynomial feature commits — retry Cand 3 FE revision under this protocol.
- Intra-string cross-position attention beyond bigrams; dedicated char transformer already helped, deeper positional structure possible.
- GBDT meta-learner on stacked probs + raw features not conclusively beaten L2 LR; dedicated tuning pass warranted.
- End-to-end co-optimization (base + meta jointly) needs warm-starting from known-good stack, not cold Bayesian.
- Aux-loss weight sweep (1%–20%) on two-tower for potential further stabilization.
- Pre-authorize tuner config edits via task config (not text clarification) to avoid repeated sub-agent refusals.
