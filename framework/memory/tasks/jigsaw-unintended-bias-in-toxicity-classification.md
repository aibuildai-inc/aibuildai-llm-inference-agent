# Task Memory: jigsaw-unintended-bias-in-toxicity-classification

## Task profile
- Type: Binary text classification (toxicity) with bias-aware evaluation
- Dataset: ~1.8M comments, text modality; fractional toxicity labels (soft targets), 6 auxiliary targets, 9 identity subgroup annotations (~50% of samples have identity labels)
- Metric: Custom bias-aware AUC — overall AUC + power mean (p=-5) of 3 subgroup bias AUCs (Subgroup, BPSN, BNSP) across 9 identity groups; p=-5 harshly penalizes worst subgroup. Higher is better.
- Medal thresholds: median benchmark ~0.94088 task score; mlebench_grade of 0.76917 achieved but still below medal range

## Best result
- Score: 0.76917 (mlebench_grade) / 0.94088 (task metric) | Medal: None | Run: 20260221_191402
- Approach: RoBERTa-base (125M params) with focal loss + identity-balanced batch sampling + mean pooling. Focal loss naturally handles 92:8 class imbalance; identity-balanced sampling ensures toxic identity-mentioning samples get sufficient gradient signal. 2–3 epochs feasible within time budget.

## Score progression
| Run | Score (mlebench) | Medal | Key change |
|-----|-------------------|-------|------------|
| 20260212_172125 | None | Ungraded | All 6 candidates timed out at coder step; no training completed |
| 20260218_223856 | 0.7929 | None | DeBERTa-v3-large attempted; collapsed to 0.5 predictions due to fp16/LR bugs |
| 20260221_191402 | 0.76917 | None | RoBERTa-base focal loss + identity sampling; first successful end-to-end run |

## Proven strategies
- **DeBERTa-v3-base is the throughput sweet spot**: 800–1000 samples/sec enables 2–3 full epochs on 1.8M samples within time budget; large/xlarge models risk <1 epoch and underfitting
- **Focal loss works**: `(1 - p_t)^gamma` handles 92:8 imbalance without manual reweighting; produces well-calibrated probabilities; matched or beat explicit identity-weighted BCE on mlebench_grade
- **Identity-weighted BCE also works**: 4-quadrant sample weighting (toxic+identity, toxic+no-identity, non-toxic+identity, non-toxic+no-identity) directly targets the 3 bias sub-metrics; achieved same 0.94088 task score as focal loss
- **Soft fractional labels >> hard binary**: Fractional toxicity targets carry genuine information; hard thresholding discards it
- **Mean pooling over all tokens > CLS-only pooling**: Used in best-scoring candidate
- **Multi-candidate filtering before coding saves budget**: 7→3 or 6→3 filtering consistently catches overengineered approaches early (contrastive debiasing, 900M param models)

## Failure patterns
- **Coder timeout at 1200s is the #1 killer**: Hit in all 3 runs; data pipeline on 1.8M samples + validation metric computation is the bottleneck, not model architecture
- **LR scheduler misconfiguration causes catastrophic divergence**: Warmup steps computed for 5 epochs but only 1 epoch runs → LR diverges. Always compute `total_steps = samples / batch_size * actual_epochs`
- **DeBERTa-v3-large fp16 weight loading trap**: Loads weights in fp16 by default; optimizer cannot update fp16 tensors → predictions collapse to 0.5. Must use explicit `model.float()` or `torch_dtype=torch.float32`
- **bf16+autocast causes logit collapse**: Use fp16+GradScaler instead for DeBERTa family
- **Multi-task auxiliary heads underperform in low-epoch regime**: DeBERTa-v3-large with identity+sub-attribute auxiliary heads scored 0.78141 vs 0.94088; tasks compete for limited gradient budget when <2 epochs
- **Large models (DeBERTa-v3-large 304M, DeBERTa-v2-xlarge 900M) are net negative**: Throughput too low for full convergence; debugging overhead (fp16, memory, contention) eats time budget
- **GPU contention causes spurious failures**: cuda:1 at 97% utilization from other processes caused random divergence; use explicit `CUDA_VISIBLE_DEVICES`

## Feature engineering playbook
- **Identity-balanced batch sampling**: Ensure each batch contains sufficient toxic+identity-mentioning samples; directly improves worst-subgroup AUC (the p=-5 bottleneck)
- **4-quadrant sample weighting**: Weight samples by (toxic/non-toxic) × (identity-mentioning/not) quadrant to target Subgroup, BPSN, BNSP AUC components independently
- **Token length management**: MAX_LENGTH=192 works but 128 may suffice with token budgeting (preserve identity mentions at full length, truncate rest) to speed forward pass
- **Validation on identity-annotated subset only (~50%)**: Bias metric only computable on identity-labeled samples; skip non-annotated samples during eval to halve validation time

## Tuning insights
- **BERT-base defaults**: EPOCHS=3, BATCH_SIZE=64, LR_BACKBONE=2e-5 are reasonable starting points
- **Measure actual step time from a short sanity run**: Hardcoded estimates (0.20s vs actual 0.034s) corrupt LR schedules by 6x; always time empirically
- **Progressive unfreezing / layer-wise LR decay not yet explored**: May help identity-bias mitigation by selectively fine-tuning upper layers + classification head
- **Time-budget-based scheduling preferred over epoch-based**: For quota-limited compute, adapt schedule to wall-clock rather than fixed epoch count
- **Implement divergence detection callback**: Auto-rollback to last checkpoint on NaN loss or logit collapse rather than wasting remaining budget

## Open questions
- **Why does focal loss beat identity-weighted BCE on mlebench_grade (0.76917 vs 0.76631) when task scores are identical (0.94088)?** Likely better probability calibration; investigate submission distributions
- **Ensemble of 3 RoBERTa-base with different seeds/augmentation untested**: Smaller models are faster to debug; ensemble may push past current plateau
- **Vectorized power-mean (p=-5) in PyTorch not implemented**: Current sklearn-based eval is a compute bottleneck; vectorized version could save significant validation time
- **1-epoch + 10% data sanity check not yet implemented**: Would catch timeout issues before committing full budget
- **No run has completed training without timeout**: Need tighter per-candidate time budgets or cheaper validation loops; full training must never timeout if design is viable
