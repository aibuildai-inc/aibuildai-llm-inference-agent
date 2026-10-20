# Task Memory: rsna-miccai-brain-tumor-radiogenomic-classification

## Task profile
- Type: Binary classification (MGMT promoter methylation status)
- Dataset: 526 patients, 4 MRI modalities (FLAIR, T1w, T1wCE, T2w), DICOM format, 512×512 slices, highly variable slice counts (15–514 per volume), near-balanced classes (51.9%/48.1%)
- Metric: AUC-ROC, higher is better
- Medal thresholds: Bronze ≈ 0.55, Silver ≈ 0.59, Gold ≈ 0.617; competition ceiling ~0.62

## Best result
- Score: 0.5977 | Medal: Silver | Run: 20260302_053525
- Approach: Two-phase EfficientNet-B0 fine-tuning (frozen 3 epochs → unfrozen 9 epochs) on central slices from FLAIR + T1wCE. Discriminative layer-wise LRs, label smoothing 0.05, EMA, cosine annealing with warmup, aggressive augmentations, 5-fold stratified CV. CV AUC 0.6249.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_064515 | 0.5753 | Bronze | Baseline: 3 candidates (EfficientNet-B0 2D, 3D ResNet-10, radiomics+GBM) |
| 20260228_194814 | 0.4412 | None | Frozen EfficientNet+LightGBM CV=0.601; fine-tuned model collapsed to 0.44 on test |
| 20260301_104708 | 0.5377 | None | Multi-seed ensemble with TTA achieved gold CV=0.617 but test=0.538 |
| 20260301_221830 | 0.5812 | Bronze | ResNet-18 FLAIR-only + triple pooling (max+mean+attention) |
| 20260301_222108 | 0.5377 | None | Frozen EfficientNet+LightGBM CV=0.614; test degraded |
| 20260301_222137 | 0.4953 | None | Frozen EfficientNet+GBM CV=0.626; ensemble hurt final score |
| 20260302_053525 | 0.5977 | Silver | Two-phase EfficientNet-B0 fine-tuning, layer-wise LRs, EMA, label smoothing |
| 20260302_053531 | 0.5577 | None | 4-modality ResNet-18 channel stacking, CV=0.635 but test gap |

## Proven strategies
- **Frozen backbone + classical ML is the safe baseline**: Frozen EfficientNet-B0 features + LightGBM/GBM consistently achieves CV 0.60–0.63, trains in <10 min, and avoids catastrophic overfitting. Viable bronze-level approach.
- **Two-phase fine-tuning is the high-ceiling approach**: Frozen warmup (2–3 epochs) → gradual unfreezing with layer-wise LRs achieved the best test score (0.5977). Critical to pair with EMA + label smoothing + weight decay.
- **Central slice selection is robust**: Selecting N middle slices (10–20) from numerically-sorted DICOMs works as well as learned attention on 526 patients. Heuristic ≥ learned on this dataset.
- **FLAIR + T1wCE are the essential modalities**: These two carry the bulk of discriminative signal. T1w and T2w add noise more often than signal.
- **EfficientNet-B0 is the go-to backbone**: Lightweight (5.3M params), strong ImageNet transfer, fast training (~40 min for 5-fold CV on A100). ResNet-18 is a viable alternative for full fine-tuning due to smaller capacity.
- **5-fold stratified CV at patient level**: Standard across all successful runs; prevents leakage and provides stable estimates.
- **Regularization stacking is mandatory**: Label smoothing (0.05) + EMA + dropout (0.3–0.4) + weight decay (0.01) + gradient clipping + early stopping (patience 3–5) must be used together, not individually.
- **Multi-seed ensembles reduce variance**: Averaging across seeds/slice-strategies achieved gold CV (0.617) in one run; diversity through configuration variation > single complex model.

## Failure patterns
- **CV-test gap is the #1 problem**: Multiple runs showed strong CV (0.60–0.63) but degraded test scores (0.44–0.54). Fine-tuned models are especially vulnerable. The MGMT signal is weak and noisy.
- **3D CNNs consistently fail**: Every 3D approach (ResNet-10, ResNet-18 3D, custom 3D) was killed for timeout, overfitting, or VRAM issues. Variable slice counts and 526 patients make 3D infeasible.
- **MIL/attention mechanisms don't converge**: Gated attention, learned slice weighting, and bidirectional LSTM aggregation all failed—526 bags are too few for attention to learn meaningful patterns.
- **Multi-modal channel stacking breaks pretrained weights**: Modifying conv1 for 4-channel input destroys ImageNet pretraining. Late fusion (per-modality features concatenated) is strictly better.
- **Complex architectures timeout**: Multi-stream fusion, MIL with variable-length batching, and attention-based aggregation repeatedly hit 600–1200s limits due to DICOM I/O bottleneck.
- **Ensemble of weak models hurts**: Naively averaging a good model with a bad one (e.g., 0.626 + 0.591 → 0.495 test) can degrade performance. Ensemble weighting must be optimized via CV.
- **PyTorch API churn**: `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler('cuda')` and `autocast` signature changes caused debugging delays in multiple runs.

## Feature engineering playbook
- **Slice selection**: Central N slices (10–20) from numerically-sorted DICOMs. Filter all-zero padding slices by non-zero pixel count. "Spread" (evenly-spaced) slices are a useful diversity source for ensembles.
- **Frozen feature extraction**: EfficientNet-B0 → 1280-dim per slice → statistical pooling (mean/max/std) across slices → 7680-dim per 2-modality patient vector. Optionally PCA to 100 components before GBM.
- **Augmentations**: Horizontal flip, rotation (±15°), affine transforms, brightness/contrast jitter, Gaussian noise. TTA with horizontal flip provides modest but consistent gain.
- **DICOM preprocessing**: Cache to .npy (~69 min one-time cost). Numeric sort of filenames is critical (lexicographic sort corrupts slice order). Lazy loading keeps RAM <50GB but is I/O-bound.
- **Triple pooling**: Max + mean + attention pooling concatenated (512 → 1536 dims) captures diverse slice patterns better than any single aggregation method.

## Tuning insights
- **LR schedule**: Cosine annealing with warmup (1–2 epochs warmup) is standard. LR 1e-3 for head, 1e-5 for backbone during unfreezing phase.
- **Label smoothing 0.05**: Higher values (0.15) didn't help; 0.05 is the sweet spot for 526 samples.
- **Early stopping patience 3–5**: Catches overfitting fast; longer patience wastes budget without gains.
- **LightGBM/GBM settings**: Early stopping at 50 rounds, L1/L2 regularization, 100–200 estimators. Fast enough for 3+ tuner iterations.
- **Tuner runs matter**: 3–5 tuner rounds on the best candidate extracted +1.7–2% AUC; focus budget on refinement of top performer rather than spreading across candidates.
- **Mixed precision + gradient clipping (norm 1.0)**: Stabilizes training; essential for fine-tuning on small datasets.
- **Backbone freezing sweet spot**: For ResNet-18, freeze conv1 + bn1 + layer1 + layer2; for EfficientNet-B0, freeze entire backbone during phase 1.

## Open questions
- **Why does CV consistently overestimate test performance by 3–8%?** Fold-wise AUC variance (0.58–0.66) suggests noisy labels or distribution shift in test set. No mitigation found yet.
- **Is the MGMT-MRI signal genuinely at ~0.62 AUC ceiling?** Competition winners barely exceeded 0.62. Consider whether preprocessing (windowing, normalization) is suboptimal or if the biological signal is inherently weak.
- **Can domain-specific pretraining help?** ImageNet features transfer surprisingly well, but self-supervised pretraining on unlabeled brain MRI (UK Biobank etc.) might shift the ceiling. Never tested.
- **Why does single-modality (FLAIR-only) sometimes beat multi-modal?** Alignment noise between modalities with different slice counts may overwhelm complementary signal. Explicit 3D spatial registration before fusion untested.
- **Optimal ensemble weighting**: Naive averaging hurts when models have unequal quality. CV-optimized blend weights or stacking never properly implemented.
- **Candidate 2 in run 20260302_053525 peaked at CV 0.649 then regressed to 0.625**: Could targeted LR schedule changes lock in the higher plateau?
