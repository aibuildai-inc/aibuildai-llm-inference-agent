# Task Memory: mlsp-2013-birds

## Task profile
- Type: Multi-label classification (19 bird species, binary relevance per species)
- Dataset: 258 training samples (322 total recordings), pre-computed features: 100-dim histogram-of-segments, 38-dim variable-length segment features; raw audio/spectrograms available. 52% of recordings have zero detected segments. Extreme class imbalance: 2–49 positives per species; 43.8% of training recordings have no species labels.
- Metric: ROC AUC (macro-averaged across 19 species), higher is better
- Medal thresholds: Bronze ≈ 0.87372; historical competition winner achieved ~0.954 AUC

## Best result
- Score: 0.8698 | Medal: none (above median) | Run: 20260309_194125
- Approach: Ensemble stacking combining LightGBM on rich tabular features (histogram + segment aggregations + librosa), transfer-learning ResNet-18 on mel-spectrograms, and attention MIL on segments. Simple averaging of predictions. Best single model was ResNet-18 at 0.956 CV AUC.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260210_060123 | 0.7517 | none | CNN on mel-spectrograms; budget exhaustion, no completion |
| 20260210_171743 | unknown | ungraded | EfficientNet AUC=0.0 (NaN bug); run stalled |
| 20260210_193634 | unknown | ungraded | LightGBM 0.7536 CV; stalled before revision |
| 20260210_203414 | 0.7166 | none | **Breakthrough**: segment-level multi-instance pooling → 0.966 CV AUC but 0.717 test (massive overfit) |
| 20260210_232340 | 0.7416 | none | LightGBM 387-dim features, 0.865 CV → 0.742 test (val overfit on 52-sample holdout) |
| 20260211_004401 | unknown | ungraded | Run terminated before submission |
| 20260211_010557 | unknown | ungraded | LightGBM 678-dim, CV 0.856; run incomplete |
| 20260211_040607 | 0.8593 | none | LightGBM ensemble, N_FOLDS=12, 0.902 CV → 0.859 test; best pre-CNN result |
| 20260222_230935 | 0.8353 | none | Complex stacking ensemble regressed; librosa feature dimension bug (205→144d) |
| 20260226_060804 | 0.8101 | ungraded | GBM on histogram + 6 engineered features; no tuning budget used |
| 20260309_194125 | 0.8698 | none | ResNet-18 transfer learning + ensemble; above median for first time |

## Proven strategies
- **LightGBM one-vs-rest per species** is the reliable baseline; consistently reaches 0.85–0.90 CV AUC
- **Segment-level aggregation (mean/max/std/min/percentile)** over variable-length segments beats pre-aggregated histograms and raw flattening
- **Transfer learning CNN (ResNet-18)** with frozen-then-finetune, SpecAugment, mixup achieved best single-model CV AUC (0.956)
- **Per-species scale_pos_weight** (square-root dampened) essential for handling 2–49 occurrence range
- **Stratified k-fold CV** (5–12 folds) on full 258 samples; single holdout (52 samples) gives unreliable estimates
- **Increasing N_FOLDS** (5→12) was a bigger lever than hyperparameter tuning (+0.047 AUC gain)
- **Simple averaging** of ensemble predictions beats learned meta-learners on <300 samples
- **Histogram-of-segments (100d)** covers 100% of recordings vs. segment features (~48%); always include as baseline
- **Metadata features** (site one-hot, temporal from filenames) add cheap predictive signal

## Failure patterns
- **Massive CV-to-test gap**: Repeatedly observed 0.10–0.25 AUC drops (e.g., 0.966 CV → 0.717 test; 0.865 CV → 0.742 test). Primary failure mode of this task.
- **Dimensionality explosion kills models**: 500–800+ features on 258 samples causes overfitting (librosa 813d → 0.579 AUC; ExtraTrees 392d → 0.793 AUC)
- **Deep learning from scratch fails**: CNNs without pretraining score 0.73 or worse; EfficientNet produced NaN/0.0 AUC
- **Stacking/learned ensembles overfit**: Logistic regression meta-learner, scipy-optimized weights, and stacked ensembles consistently underperform simple averaging
- **MIL attention underperforms**: 52% zero-segment recordings starve attention models; simpler pooling statistics match or beat MIL
- **Feature extraction code fragility**: Librosa pipeline refactor (205d→144d) caused 4% AUC drop in one run
- **Budget exhaustion**: Multiple runs (4 of 11) terminated before completion; tree methods train in <5 min, CNNs consume 30+ min

## Feature engineering playbook
- **Core features (always include)**: 100-dim histogram-of-segments + aggregated segment stats (mean/std/max/min/median/skew/kurtosis over 38 dims → ~266d)
- **Derived histogram features**: L1 norm, entropy, nonzero count, max value, std, zero-flag (6 features, cheap and helpful)
- **Librosa features (use carefully)**: MFCCs + deltas, chroma, spectral centroid/rolloff/contrast/flatness, ZCR, RMS, tonnetz, mel stats → target 205 dims. Critical: preserve all dimensions; do NOT simplify aggregation
- **Spectrogram features**: Mel-spectrograms for CNN path; BMP-derived features (306d) available but risk dimensionality issues
- **Metadata**: Site one-hot encoding, temporal features from filenames
- **Keep total dims ≤ 400**: Beyond this, overfitting dominates on 258 samples; if using librosa, apply feature selection (variance threshold, mutual information)

## Tuning insights
- **N_FOLDS is the #1 hyperparameter**: 5→12 folds gives +0.047 AUC; plateau around 12–15 folds on 258 samples
- **Revision > hyperparameter tuning** when score is far from target: redesigning features/architecture beats grid search
- **LightGBM > XGBoost ≈ HistGBM > ExtraTrees** on this task; per-species best-model selection across ensemble members adds robustness
- **Early stopping (50 rounds)** prevents overfitting; max_depth=4 for shallow trees
- **ResNet-18 two-phase training**: Phase 1 frozen backbone (higher LR), Phase 2 fine-tune all (lower LR); dropout 0.5, EMA, label smoothing all necessary
- **Do NOT trust small holdout validation**: Use full k-fold CV; 52-sample holdout led to 0.12 AUC overestimate

## Open questions
- **CV-to-test gap is the unsolved problem**: Best CV AUC 0.956 but best test AUC only 0.8698. Distribution shift between train/test? Overfitting to CV folds? Investigate recording-level vs. site-level splitting.
- **Audio-specialized pretraining (PANNs, CLAP, BirdNET)** untested; likely higher ceiling than ImageNet-pretrained ResNet for spectrograms
- **Pseudo-labeling / semi-supervised**: 43.8% of recordings have zero labels; untapped signal for calibration
- **Per-species threshold tuning**: ROC AUC is threshold-agnostic but calibrated probabilities may matter for submission format
- **Species co-occurrence patterns**: Multi-task learning exploiting species correlations untested
- **Why does ensemble underperform best single model on test?** ResNet-18 alone (0.956 CV) may be more robust than ensemble (0.8698 test); weaker base models may add noise
- **Confidence-weighted averaging** (weight by per-species CV AUC) as middle ground between simple averaging and learned stacking
