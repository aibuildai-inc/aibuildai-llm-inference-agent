# Task Memory: dogs-vs-cats-redux-kernels-edition

## Task profile
- Type: Binary image classification (dog vs. cat)
- Dataset: 25K training images (balanced 50/50), 12.5K test images; variable-resolution JPEGs resized/padded to 224×224
- Metric: LogLoss (lower is better)
- Medal thresholds: target <0.06127 LogLoss mentioned; no explicit medal tiers recorded

## Best result
- Score: ungraded | Medal: ungraded | Run: none completed training
- Approach: No run has completed a full train→evaluate→submit cycle. Closest was run 20260207_182214 where ResNet50 + heavy augmentation (Candidate 2) survived judge filtering as the sole viable implementation, but training was never executed within the agent-call budget.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260207_003651 | ungraded | — | 5 architectures coded & verified (dry-run only); agent calls exhausted before training |
| 20260207_180239 | ungraded | — | API credit exhaustion (402) during design phase; 0 candidates instantiated |
| 20260207_181310 | ungraded | — | Total initialization failure; 0 candidates, 0 agent calls |
| 20260207_182214 | ungraded | — | 4 designed, 1 survived judge (ResNet50); 3 killed by bugs. Still no training executed |
| 20260207_215941 | ungraded | — | 5 designed, 4 survived judge; 1 tuner run attempted. Ensemble killed for time violation. No training completed |

**Trajectory:** Zero scored submissions across 5 runs. Persistent bottleneck is exhausting agent calls / tokens / time before any candidate completes training.

## Proven strategies
- **ResNet50 frozen-layer fine-tuning** is the most robust single-model approach: freeze conv1→layer2, fine-tune layer3→layer4. Survived judge in multiple runs without bugs.
- **BCEWithLogitsLoss (raw logits, no sigmoid in model)** is the correct loss setup—confirmed across runs; double-sigmoid is a lethal bug.
- **Heavy augmentation pipeline** (RandomResizedCrop, horizontal flip, ColorJitter, GaussianBlur, RandomGrayscale) consistently designed and validated.
- **EfficientNet-B0 / EfficientNetV2-S** are the best speed-accuracy tradeoff under the 20-min CPU constraint (~5.3M params).
- **Quick-test verification** (import check, forward pass, loss computation on mini-batch) catches bugs before committing to full training.
- **ImageNet normalization** is standard and universally applied.
- **80/20 stratified train/val split** is the consensus data split strategy.

## Failure patterns
- **Agent call / token budget exhaustion before training**: Occurred in 4 of 5 runs. Design phase consumes 20–25K tokens for 5 candidates; leaves nothing for training.
- **Double-sigmoid bug**: Model includes sigmoid + BCELoss expects probabilities → incorrect gradients. Hit EfficientNetV2-S in run 20260207_182214.
- **Scheduler.step() in minibatch loop instead of epoch loop**: Recurred in Candidates 3 and 4 across runs. Decays LR 625× faster than intended, preventing convergence.
- **Ensemble infeasibility**: Multi-model ensembles consistently killed for exceeding 20-min CPU training budget (estimated 45–300 min). Tempting but impractical.
- **Label smoothing declared but not implemented**: Config says ε=0.1 but loss function uses standard BCE. Seen in multiple candidates.
- **ViT-B/16 too expensive**: Warmup/gradient-clipping overhead + large model size makes it impractical for 20-min CPU runs.

## Feature engineering playbook
- **Resize strategy**: Pad/resize to 224×224; aspect-preserving crop mentioned but inconsistently enforced—standardize this.
- **Augmentation stack**: RandAugment or manual (RandomResizedCrop, HFlip, ColorJitter, GaussianBlur, RandomGrayscale). MixUp/CutMix designed but never tested.
- **Label smoothing (ε=0.1)**: Expected to improve log-loss calibration by 0.01–0.02; never actually tested.
- **Prediction clipping**: Clip probabilities away from 0/1 to avoid log-loss explosion. Consensus recommendation, untested.
- **TTA (Test-Time Augmentation)**: Designed but never executed; expected marginal log-loss improvement.

## Tuning insights
- **Optimizer**: SGD+momentum is the default; no Adam/AdamW comparison attempted.
- **Scheduler**: ReduceLROnPlateau is safe; CosineAnnealing designed but untested. **Critical**: call `scheduler.step()` per epoch, never per batch.
- **Frozen layers**: For ResNet50, freezing through layer2 and fine-tuning layer3+ is the sweet spot for 25K images.
- **Epochs**: 5–10 epochs feasible on CPU for lightweight models (EfficientNet-B0, ResNet50). Profiling per-epoch wall time is essential.
- **Dropout**: Applied in classification head; rate not systematically tuned.

## Open questions
- **Can any candidate actually train and submit within budget?** Zero runs have completed training. Must allocate ≥3 of 5 agent calls to training/eval/submission.
- **Reduce design phase cost**: Use 1–2 candidates max (EfficientNet-B0 or ResNet50) instead of 5 to preserve budget for training.
- **EfficientNet-B0 vs. ConvNeXt-Tiny**: No empirical comparison exists on this dataset under 20-min CPU constraint.
- **Label smoothing + clipping impact**: Estimated 0.01–0.02 log-loss improvement; needs actual measurement.
- **Per-epoch wall time on target hardware**: Unknown. Must profile before committing to architecture. If <2 min/epoch, ensemble of 2 small models may become feasible.
- **Why did run 20260207_181310 produce zero agent calls?** Initialization failure never diagnosed.
- **Ensemble diversity**: Does multi-seed + different augmentation actually reduce log-loss vs. single best model with TTA?
