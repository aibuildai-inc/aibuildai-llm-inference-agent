# Task Memory: statoil-iceberg-classifier-challenge

## Task profile
- Type: Binary classification (iceberg vs ship) on SAR satellite imagery
- Dataset: ~1.3k samples, dual-band (HH/VV) 75x75 dB-scale SAR patches + incidence angle (~8% missing, 'na' sentinel); test set contains machine-generated distractors
- Metric: Log-loss (lower is better)
- Medal thresholds: Bronze ≈ 0.14552 (inferred target)

## Best result
- Score: 0.18114 | Medal: none (above median) | Run: 20260418_201634
- Approach: Two-level stacked ensemble of 4 base models (SmallCNN, ResNet18, LightGBM, multi-task CNN) on shared 5-fold split, combined via L2 logistic regression meta-learner (C=40). All bases used mixup (α=0.2–0.4) and TTA (flips + 90° rotations); HH–VV band-math as 3rd channel; inc_angle treated as auxiliary regression target + missing indicator.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260418_201634 | 0.18114 | none | Baseline: 4-base stack + meta-LR; OOF 0.17033 but 3.5% short of bronze |

## Proven strategies
- Stacking diverse bases (DL + GBDT) beats any single model (0.17 stacked vs 0.20+ solo)
- LightGBM in the mix: weak OOF (0.64) but well-calibrated predictions lift the blend
- Mixup (α=0.2–0.4) + TTA (flips/90° rotations) stabilizes training on tiny dataset
- Locked 5-fold indices across ALL bases — prevents meta-learner leakage/overfitting
- HH–VV band-math as 3rd channel improves CNN feature extraction
- Multi-task CNN with masked inc_angle regression = free regularization, no leakage
- Missing-indicator feature for 'na' inc_angle > imputation

## Failure patterns
- Pseudo-labeling / semi-supervised (C5): killed — test set's machine-generated distractors poison pseudo-labels
- Tuner plateau at 0.17–0.18 after ~3 iterations; gains require code changes, not HP search
- Uncalibrated stacking plateaus ~0.17; bronze gap likely closable only via post-hoc calibration
- Deep/wide CNNs overfit on 1.3k samples; OOM blocked wider backbones on single GPU
- Infra: system-reminder false-refusals halted C2/C3/C6 tuner runs (requeue needed)
- Reviser↔coder lag: revision plans (isotonic, convex blend, 5th seed) written to JSON but not materialized into train.py on first pass

## Feature engineering playbook
- 3-channel input: band_1, band_2, (band_1 − band_2) polarimetric contrast
- Per-band statistics as tabular features for GBDT branch
- inc_angle: (a) auxiliary regression target with masked loss, (b) binary missing indicator
- Shared fold indices file consumed by every base model

## Tuning insights
- ~1.3k samples → prioritize ensemble diversity over individual model depth
- Meta-learner: L2 logistic regression, C=40 worked; fold OOFs must be from identical splits
- Calibration matters more than raw accuracy at this scale — isotonic/Platt recommended for next iter
- Snapshot ensembling within each CNN base adds incremental gains

## Open questions
- Post-hoc isotonic/Platt calibration on top of stacked OOFs — not yet validated end-to-end
- Non-negative convex blend + 5th decorrelated CNN seed (revised but not fully tested)
- Multi-GPU / larger batch training to enable wider CNN backbones without OOM
- Confidence-thresholded pseudo-labeling that skips likely synthetic test samples
- Knowledge distillation from ensemble → single student CNN for inference speed
