# Task Memory: protein-compound-interaction

## Task profile
- Type: Binary classification (protein-compound binding interaction)
- Dataset: ~114K training samples; paired protein sequences + compound structures
- Metric: AUC-ROC (higher is better)
- Medal thresholds: Target ~0.8 AUC (gold-like); gradient-boosting ceiling observed ~0.78

## Best result
- Score: 0.7792 AUC-ROC | Medal: ungraded | Run: 20260315_050632
- Approach: XGBoost with Morgan fingerprints + amino acid composition + top-500 frequency-ranked protein k-mers (k=2,3,4), trained under 5-fold stratified CV. K-mer features broke the ~0.75 plateau set by fingerprints+AA alone.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260315_030604 | 0.7563 | None | LightGBM + Morgan fingerprints + RDKit descriptors + protein k-grams |
| 20260315_050632 | 0.7792 | ungraded | XGBoost + Morgan + AA composition + top-500 freq-ranked k-mers (k=2–4); +2.3 pp |

## Proven strategies
- **Gradient boosting (LightGBM/XGBoost) dominates deep learning** on this ~114K-sample task; both runs confirm GBDT > CNN/transformer/GNN baselines
- **Morgan fingerprints** as molecular baseline: stable compound signal across both runs
- **Protein k-mers (k=2–4), frequency-ranked, top ~500**: the single biggest lift observed (+~0.02 AUC over fingerprints+AA alone); breaks the ~0.75 plateau
- **5-fold stratified CV**: reliable hyperparameter selection, no CV→test overfitting detected in run 1
- RDKit descriptors + AA composition as cheap complements to fingerprints/k-mers

## Failure patterns
- **Deep learning consistently fails**: CNN, transformers, end-to-end encoders scored <0.55 AUC or 0.0 (convergence failures) in run 1; abandoned per manager guidance
- **ESM-2 integration is brittle**: environment loading errors (run 1), too slow for 114K samples (run 2); never cleanly integrated despite being the most promising avenue
- **GNN molecular encoders**: computational overhead exceeds 4h wall budget
- **Ensemble pitfall**: XGBoost + LightGBM with different feature sets (one with k-mers, one without) produced shape mismatches; ensemble attempt abandoned in run 2
- Candidates repeatedly killed mid-tuning due to dependency/library issues (runs 1 and 2)

## Feature engineering playbook
- Compounds: Morgan fingerprints (mandatory baseline) + RDKit descriptors; MACCS keys untested at success
- Proteins: AA composition alone is insufficient (~0.75 ceiling); add top-500 frequency-ranked k-mers with k ∈ {2,3,4} for +~0.02 AUC
- Align preprocessing across models BEFORE ensembling — feature shape parity is a hard requirement
- Hand-crafted features beat learned representations at this data scale

## Tuning insights
- XGBoost and LightGBM perform near-equivalently (0.7324–0.7792 range); either is a safe default
- 5-fold CV stabilizes tuning on 114K samples; single train-test splits give brittle estimates
- CV→test transfer is stable — selection decisions reliable without held-out drift

## Open questions
- Does **cached ESM-2 embeddings** (not fine-tuned) close the ~2 pp gap to 0.8 AUC if environment issues are fixed?
- Would **class-enriched k-mer selection** (active vs inactive) outperform frequency-ranked k-mers?
- Can XGBoost + LightGBM **ensembling with aligned features** exceed either alone?
- Is **MPNN + readout on compound graphs** feasible within the 4h budget, and would it beat Morgan fingerprints?
