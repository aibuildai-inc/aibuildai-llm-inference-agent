# Task Memory: stanford-covid-vaccine

## Task profile
- Type: Sequence-to-sequence regression (per-position, multi-target) on RNA
- Dataset: ~2160 RNA sequences, length 107; scored positions 0–67; 811 noisy samples (SN_filter=0), 1349 clean (SN_filter=1); small-data regime
- Metric: MCRMSE / multi-column log loss (lower is better)
- Medal thresholds: Gold achievable at ≤0.339 (observed); best run 0.21573

## Best result
- Score: 0.21573 | Medal: Gold | Run: 20260301_162658
- Approach: Ensemble of 4 diverse architectures (BiGRU+BPP, CNN+GCN, Transformer, Structured Transformer) all trained only on SN_filter=1 samples with BPP structural features. Single-model baseline converged to 0.232–0.235; ensemble blend dropped to 0.21573.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260228_031619 | 0.33861 | Gold | Bi-GRU, 10-fold CV, SN-weighted, target clipping; sequence-only |
| 20260301_162658 | 0.21573 | Gold | SN_filter=1 hard cutoff + BPP features + 4-model ensemble |

## Proven strategies
- **Filter training data to SN_filter=1** (1349 clean samples): single largest lever, 0.67→0.24 (~65% reduction). Beats L2/weighting on noisy rows.
- **BPP (base-pair probability) structural features** from ViennaRNA/RNAfold: mandatory; sequence-only underperforms by >20% once data is clean.
- **Ensemble diverse architectures at convergence**: BiGRU, CNN+GCN, Transformer converge to 0.232–0.235 individually; blending yields ~1.5% gain.
- **K-fold CV (≥10-fold)**: stabilizes per-position regression, required to detect leakage in seq2seq setup.
- **Target outlier clipping** (vs deletion): preserves 37% of data that would otherwise be dropped; prevents NaN/divergence.
- **Position-masked loss**: train on all 107 positions but optimize only scored 0–67.

## Failure patterns
- **Training on full noisy dataset**: models score 0.48–0.67 regardless of regularization; overfits to noise.
- **Transformer + BPP without filtering**: 0.669; complexity ≠ capability on small data.
- **Pretrained RNA LMs**: 0.497 → 0.6989 across runs; transfer learning consistently ineffective at N≈2k.
- **Complex dual-stream / WaveNet+LSTM**: Conv-Transformer 0.320, WaveNet+LSTM timed out; architectural complexity adds no gain.
- **Val/LB gap on run 1** (0.193 val → 0.339 LB): likely from missing BPP features + not filtering by SN, causing distribution mismatch with high-SNR test set.

## Feature engineering playbook
- Compute BPP matrices via ViennaRNA as structural input — do not skip.
- Hard-filter train to SN_filter=1; do not weight/soft-filter noisy rows.
- Clip target outliers to train-time quantiles (observed targets range to ±44).
- Mask loss to scored positions 0–67 only; feed full 107-length sequence.

## Tuning insights
- **Preprocessing > architecture > hyperparameters**: once clean data + BPP in place, model family is ~irrelevant (all hit 0.23–0.24 ceiling).
- **Small data favors RNN/CNN over Transformer/GNN/pretrained**: Bi-GRU beat Transformers and pretrained RNA models in run 1.
- **Regularization (dropout, weight decay)** matters more than capacity on N=2160.
- After reaching single-model ceiling, shift budget from tuning → ensembling / TTA.

## Open questions
- SN_filter threshold sensitivity: is strict =1 optimal, or does loosening to include some SN_filter=0 rows (with weighting) retain more signal?
- TTA (reverse-complement, masking) untried; potential to push ensemble <0.21.
- Checkpoint aggregation strategy: best-epoch blending vs re-trained meta-learner unexplored.
- Is the run-1 val/LB gap (0.193→0.339) eliminated purely by SN-filter+BPP, or does a submission-pipeline bug persist?
