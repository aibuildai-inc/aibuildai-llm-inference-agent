# Pattern: classification-challenges-general

## When this applies
- Small-to-medium tabular or audio datasets (~250–10k samples) with mixed feature types, structured raw fields, or class imbalance
- Multi-label, multi-class, or binary classification with held-out test set susceptible to distribution shift

## Proven patterns
- **Tree-based models (LightGBM/XGBoost) are the reliable baseline**: spaceship-titanic XGBoost 0.8149 vs MLP 0.772; mlsp-2013-birds LightGBM ensemble reached 0.8593 pre-CNN
- **K-fold CV ensemble averaging beats single best model**: spaceship-titanic 5-fold KNN ensemble hit 0.8536 (best); mlsp-2013-birds N_FOLDS 5→12 gave +0.047 AUC
- **Domain-aware feature decomposition of structured strings**: spaceship-titanic Cabin→deck/room/side and PassengerId→group features; mlsp-2013-birds segment aggregations (mean/max/std/percentile)
- **Pre-computed features + caching**: tensorflow-speech mel-spectrograms gave 10–50× speedup; mlsp-2013-birds histogram-of-segments covers 100% of recordings as cheap baseline
- **Stratified/grouped CV is mandatory**: tensorflow-speech speaker-grouped 5-fold prevents 15–20pt leakage; mlsp-2013-birds full k-fold beats 52-sample holdout (0.12 AUC overestimate)

## Common pitfalls
- **Massive CV/val-to-test gap** is the dominant cross-task failure: mlsp-2013-birds 0.966 CV→0.717 test; tensorflow-speech 98.71% val→0.353 test; spaceship-titanic untested gap risk
- **Deep learning from scratch underperforms on small data**: mlsp-2013-birds CNNs scored ≤0.73 without pretraining; spaceship-titanic Transformer collapsed to 0.50 (random)
- **Learned/stacked meta-ensembles overfit**: mlsp-2013-birds simple averaging beat logistic-regression stacking; complex hybrids killed by judge in spaceship-titanic
- **Dimensionality explosion on small samples**: mlsp-2013-birds 813d librosa→0.579 AUC; keep features ≤ ~3× sqrt(n_samples)
- **Class prior mismatch / collapse**: tensorflow-speech predicted 64% unknown vs ~10% expected; mlsp-2013-birds extreme imbalance needs scale_pos_weight
- **Pipeline init failures eat budget**: spaceship-titanic 43% of runs failed at init; tensorflow-speech late-stage calibration patches shipped unvalidated

## Model selection guide
- Tabular, <10k rows → LightGBM/XGBoost first (max_depth=4–6, lr=0.05, early stop 50)
- Audio/spectrogram → ImageNet-pretrained ResNet-18 with two-phase finetune; avoid scratch CNNs and SincNet
- Always: 5-fold (or 12-fold if n<300) stratified/grouped CV ensemble with simple averaging
- Avoid: Transformers, deep MLPs, learned stackers, MIL attention unless n>5k

## Feature engineering patterns
- Decompose structured strings (Cabin, PassengerId, filenames) into typed sub-features
- Aggregate variable-length groups via mean/max/std/min/median/percentile statistics
- Engineer domain-rule anomaly flags (CryoSleep×spending, zero-segment recordings)
- Group-level aggregates (per-group spending, per-recording histogram) add cheap signal
- Cap feature count: mlsp-2013-birds ≤400d on 258 samples; spaceship-titanic 13→59 hit best score

## Tuning patterns
- **N_FOLDS and feature engineering > hyperparameter search**: mlsp-2013-birds folds 5→12 gave +0.047; spaceship-titanic 13→59 features dominated model choice
- Always log per-class predictions and submission class distribution before submitting (tensorflow-speech failure mode)
- Use TEST_MODE sanity runs before full training to catch pipeline bugs
- Validate late-stage calibration/reviser patches; never ship unvalidated last-step changes
