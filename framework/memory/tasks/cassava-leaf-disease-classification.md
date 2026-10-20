# Task Memory: cassava-leaf-disease-classification

## Task profile
- Type: Image classification (5 classes: 4 diseases + healthy)
- Dataset: 18.7K labeled leaf images; severe class imbalance (61.55% class 3, ~5% class 0)
- Metric: Accuracy (higher is better)
- Medal thresholds: Bronze ≥ 89.78%

## Best result
- Score: 85.15% (2-fold ensemble) | Medal: None | Run: 20260218_224101
- Approach: ConvNeXt-Base CLIP (`convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384`) with 2-fold stratified CV, Mixup/CutMix, label smoothing, AdamW LR=1e-4. Best single-fold OOF was 88.95% (CAFormer-B36) but could not be reliably retrained/deployed.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260218_224101 | 85.15% | None | ConvNeXt-Base CLIP 2-fold ensemble; CAFormer hit 88.95% OOF but undeployable |
| 20260220_234052 | N/A | Ungraded | Total failure—rate limiting + timeouts burned 11 agent calls with zero training |

## Proven strategies
- **ConvNeXt-Base CLIP is the task-proven backbone**: 87.29% single-fold, 92.1% val accuracy in prior references. Stable training, cached weights (339 MB).
- **IN-12K / IN-21K / CLIP pretraining beats IN-1K-only** by ~1–2% consistently (ConvNeXt, ViT-Base AugReg).
- **Mixup (α=0.4) + CutMix (α=1.0) + label smoothing (0.1–0.2)** are near-essential for this dataset's imbalance and size.
- **Stratified K-fold CV** (3–5 folds) prevents leakage; 5-fold gave EfficientNet-B5 ~87.79%.
- **Ensemble diversity (CNN + ViT)** helps: EfficientNet-B5 (compact CNN, 28.4M), ViT-Base (86.1M), ConvNeXt-Base (87.6M) are three disjoint niches.
- **Single-candidate-per-GPU training** is the only reliable approach on 2× GPU setups; multi-candidate runs cause OOM.

## Failure patterns
- **GPU contention kills multi-candidate training**: Repeated OOM errors on 2-GPU setups when running candidates in parallel.
- **Overfitting plateaus early (epoch 3–4)**: Warmup + cosine annealing + EMA dampen but don't eliminate; intrinsic to 18.7K dataset size.
- **Coder timeout at 1200s is too tight**: Large models (86M+ params) need 1800–2400s or staged training. ConvNeXt/ViT require ~40–60 min per run.
- **Rate limit retry loops are fatal**: Run 2 burned 7/11 agent calls retrying the same candidate. Must rotate to a different candidate after first failure.
- **Bi-tempered logistic loss (t1=0.8, t2=1.4)** offered no gain over CE + label smoothing despite label-noise claims.
- **CAFormer-B36**: Achieved best OOF (88.95%) but repeatedly failed to retrain reliably due to GPU contention and timeouts.

## Feature engineering playbook
- **Augmentation pipeline**: Albumentations with Mixup, CutMix, standard geometric transforms. These are critical given the small dataset.
- **Resolution**: 384px input resolution used by top candidates (ConvNeXt-Base-384, ViT-Base).
- **Class weighting not explicitly tested**—imbalance handled via Mixup/CutMix/label smoothing instead.

## Tuning insights
- **AdamW with LR=1e-4** is the working default for fine-tuning large pretrained backbones on this task.
- **Cosine annealing + warmup** standard; 15 epochs sufficient for ConvNeXt (fold 0 converged).
- **EMA helps stability** but doesn't close the overfitting gap alone.
- **Cached weights are a hard prerequisite**: HuggingFace-cached models (ConvNeXt 339 MB, ViT 332 MB, EfficientNet 117 MB) avoid download latency; uncached models (EfficientNetV2-M, Swin) are risky under time constraints.

## Open questions
- **LoRA / adapter tuning on frozen backbones**: Could reduce overfitting without sacrificing transfer learning; untested.
- **Knowledge distillation / pseudo-labeling**: Use test-set structure implicitly; untested.
- **Warm-start from CAFormer 88.95% checkpoint** instead of retraining from scratch—optimizer state may matter for plateau breaking.
- **Timeout calibration**: Need empirical wall-clock measurements on target hardware to set coder timeout (likely 1800–2400s).
- **Focal loss variants**: Designed in candidates but never successfully trained; could help with class 0 (5% of data).
- **5-fold ensembles of top backbones**: Both runs were time-constrained to 2–3 folds; full 5-fold ConvNeXt + ViT ensemble could close the 4.6 pt gap to bronze.
