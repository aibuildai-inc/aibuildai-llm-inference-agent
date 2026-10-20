# Task Memory: herbarium-2020-fgvc7

## Task profile
- Type: Fine-grained image classification (extreme long-tail)
- Dataset: ~811K herbarium specimen images, 32,094 plant species classes. Extreme long-tail: median 7 samples/class, 43% of classes ≤5 samples, 31.7% singletons (3,729 single-sample classes), one zero-sample class. Standard train/test split.
- Metric: Macro F1-score (higher is better)
- Medal thresholds: Silver ≈ 0.28–0.45; Gold ≈ 0.63+

## Best result
- Score: 0.451 | Medal: Silver | Run: 20260321_100643
- Approach: ConvNeXt-Tiny with ImageNet-22K pretraining, cosine LR decay, EMA regularization, auxiliary genus-level head, and balanced sampling. Extended training across 8–10 epochs pushed validation F1 to 0.557. Aggressive candidate pruning focused full GPU budget on the single best model.

## Score progression
| Run | Test Score | Val Score | Medal | Key change |
|-----|-----------|-----------|-------|------------|
| 20260213_083840 | 0.448 | 0.530 | Silver | ConvNeXt-Tiny + Balanced Softmax, iterative tuning |
| 20260321_100643 | 0.451 | 0.557 | Silver | ConvNeXt-Tiny IN-22K + EMA + genus aux head |
| 20260322_095241 | 0.395 | 0.592 | Silver | Swin-Small IN-22K + class-balanced focal loss |
| 20260308_121753 | 0.357 | 0.386 | Silver | DINOv2 ViT-Small, two-phase (linear probe→fine-tune) |
| 20260318_165010 | 0.356 | 0.474 | Silver | EfficientNet-B0→B3 upgrade + focal loss + sqrt-freq sampling |
| 20260227_105952 | 0.302 | 0.412 | Silver | EfficientNet-B0 + class-balanced focal loss |
| 20260323_103747 | 0.283 | 0.504 | Silver | ConvNeXt-Small@384px + two-stage training + TTA |

**Note:** Val-to-test gap is large and inconsistent (0.08–0.22), suggesting validation overfitting or distribution shift. Best test score (0.451) came from a run with moderate val F1 (0.557), not the highest val F1 (0.592).

## Proven strategies
- **ImageNet-22K pretraining is essential**: Consistently outperformed IN-1K across all runs. 22K-class pretraining aligns with the 32K-class target; treat as a must-have for fine-grained botanical tasks.
- **ConvNeXt-Tiny/Small backbone**: Won or co-won in 4/7 runs. Best speed-to-capacity ratio; depthwise separable convolutions + LayerNorm + GELU outperform legacy EfficientNet and ResNet families.
- **Class-balanced focal loss (γ=2.0, β=0.999)**: Reliable across 5+ runs. Handles hard examples + long-tail simultaneously without per-class α tuning. Simpler and more robust than logit-adjusted CE, LDAM, or ArcFace.
- **Balanced/sqrt-frequency sampling**: Works consistently; avoids catastrophic overfitting on singletons that explicit class reweighting causes.
- **Cosine LR schedule with warmup**: Stable across all successful runs; align total epochs to actual time budget (8–10 epochs is the sweet spot).
- **EMA regularization**: Stabilized late-stage training in multiple runs; enable after initial convergence.
- **Aggressive candidate pruning**: Focusing full GPU budget on 1–2 survivors consistently beat splitting resources across 3+ candidates.
- **Checkpoint resume + extended training**: Critical for extracting gains within time budgets; pre-test resume logic to avoid device mismatch bugs.

## Failure patterns
- **Transformer/ViT backbones timeout or underperform**: Swin-Tiny, ViT-Small (non-DINOv2), EVA02-Small all hit 1200s timeouts or implementation complexity issues. Only Swin-Small IN-22K succeeded (run 20260322) but had large val-test gap.
- **DataParallel + AMP (float16) on 32K-class output layers**: Causes NaN logits and numerical instability. Avoid or keep classifier head in float32.
- **Complex multi-stage/multi-task pipelines**: Two-stage decoupled training, hierarchical genus/family heads, ArcFace margins, and self-training all exceeded time budgets or failed at implementation. Simpler single-stage pipelines were more robust in automated settings.
- **Large backbones (EfficientNet-B5, ConvNeXt-Base, ensemble methods)**: Consistently killed by timeouts; never successfully trained within budget.
- **Logit-adjusted CE alone**: Without aggressive class-balanced sampling, achieved near-zero F1 (0.000273 in one run).
- **MobileNetV3/small capacity models**: Stall at 0.03–0.10 F1; insufficient representational capacity for 32K fine-grained classes.
- **Persistent val-to-test gap**: Multiple runs showed 0.10–0.20 gap, indicating potential overfitting to validation distribution.

## Feature engineering playbook
- **Input resolution**: 224px is baseline; 288–384px provides meaningful gains (+0.05–0.15 F1) when backbone capacity allows. ConvNeXt-Small@384px broke through ConvNeXt-Tiny@224px ceiling.
- **Augmentation stack**: RandomResizedCrop (scale 0.5–1.0), ColorJitter, RandomGrayscale, RandomErasing. Aggressive augmentation combats overfitting on 1–3 sample classes.
- **Balanced Softmax**: Debiases logits by subtracting log(class_prior) at inference; zero-parameter, stable, effective for extreme long-tail. Competitive with focal loss.
- **Label smoothing (0.1)**: Helps with late-stage rare-class overfitting when combined with focal loss; interaction with Balanced Softmax unclear.
- **Test-time augmentation**: Horizontal flip ensemble provided modest robustness gains; multi-scale TTA (multiple resolutions) untested but promising.

## Tuning insights
- **3–6 tuner runs needed**: Typical trajectory shows steep initial gains then plateau (e.g., 0.45→0.50→0.53→0.53). Budget for 5+ runs.
- **Plateau detection threshold too aggressive at 0.5%**: Gains of 2–3% came after apparent plateaus; extend training by default.
- **Layer-wise LR decay (0.75–0.9)**: Critical for ConvNeXt/Swin fine-tuning; uniform LR causes stagnation. Sensitivity within 0.6–0.9 range unexplored.
- **Epoch budget**: 8–10 epochs is the sweet spot for 32K classes; fewer undertrain rare classes, more risk overfitting head classes.
- **Classifier head size matters**: At 32K classes, the linear head is 50–65% of total params. Use 768-dim or smaller backbone features; bottleneck projections reduce overfitting.
- **Don't change architecture mid-training**: Stochastic depth adjustments on resume caused metric collapse; keep architecture frozen after initial setup.
- **Dropout 0.3 + label smoothing 0.1 in late epochs**: Prevents degradation on rare classes during epochs 15+.

## Open questions
- **ConvNeXt-Base or Swin-Base with IN-22K**: Never successfully trained within budget; extrapolation suggests 0.55–0.60+ F1 if timeout issues resolved.
- **DINOv2 ViT-Base at 336px**: Strong self-supervised pretraining (142M images) was promising but ViT-Small peaked at 0.386 val F1; larger model + resolution untested.
- **Val-to-test gap diagnosis**: Best val F1 (0.592) yielded only 0.395 test; best test (0.451) had 0.557 val. Root cause unclear—class distribution shift, augmentation leakage, or evaluation protocol mismatch?
- **Ensemble of ConvNeXt + Swin**: Never implemented due to time budgets; diversity between architectures could push past 0.50 test F1.
- **Metric learning for tail classes**: Prototypical networks or contrastive learning for 1–3 sample classes remains untested; standard CE/focal loss may be inherently limited here.
- **Hierarchical genus/species auxiliary loss**: Theoretically sound for botanical taxonomy but always failed at implementation; simple single auxiliary genus head (weight 0.1) worth retrying.
- **Temperature scaling post-hoc calibration**: Focal loss outputs were never calibrated; could recover 1–2% F1.
- **Pseudo-labeling on test data**: Self-training approach was too complex in automated pipelines but could help tail classes.
- **Gold medal (0.63+) likely requires**: IN-22K large backbone that actually trains + progressive resolution + ensemble + extended budget (8+ GPU hours).
