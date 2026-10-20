# Task Memory: ranzcr-clip-catheter-line-classification

## Task profile
- Type: Multi-label classification (11 binary labels)
- Dataset: ~27K chest X-ray images from 3,202 unique patients (avg 8.5 images/patient); grayscale; 16,261 segmentation annotations (~60% coverage). Severe class imbalance: ETT-Abnormal 0.28% → CVC-Normal 70.82% (250× range).
- Metric: Macro-averaged ROC AUC (higher is better)
- Medal thresholds: Bronze ≥ 0.9709

## Best result
- Score: 0.96375 AUC | Medal: none | Run: 20260423_221321
- Approach: ConvNeXt-Small (IN-22k) + 4-channel FPN segmentation-auxiliary decoder (ETT/NGT/CVC/Swan), masked BCE+Dice seg loss (weight 0.1), per-class pos_weight BCE (cap 10.0), GeM pooling, GroupKFold-by-PatientID, 5-TTA rank-averaged across multiple folds. OOF 0.9499 → final 0.96375.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_072321 | 0.9386 | none | EfficientNet-V2-S 512×512; tuner degraded by 7 simultaneous changes |
| 20260302_234811 | 0.92949 | none | EfficientNet-B4 (NS) 512×512 + per-label weights + CLAHE |
| 20260305_194307 | 0.93436 | none | EfficientNet-B4 + GeM + GroupKFold-3 + OneCycleLR |
| 20260305_194253 | 0.93595 | none | DenseNet169 + GeM + 5-TTA + per-class threshold opt |
| 20260419_063507 | 0.96285 | none | EfficientNet-V2-S + SegAux 4-fold + EMA + early stop |
| 20260420_154900 | 0.95778 | none | OOF stack V2-S+SegAux + MaxViT; weak member dragged blend |
| 20260423_221321 | **0.96375** | none | **ConvNeXt-S (IN-22k) + SegAux + rank-averaged TTA across folds** |

## Proven strategies
- **Segmentation-auxiliary decoder is the ceiling-breaker**: 4-class (ETT/NGT/CVC/Swan) FPN decoder with masked BCE+Dice (weight 0.1) regularizes regardless of backbone; broke 0.93–0.94 plateau (+0.027 AUC).
- **Masked seg loss handles partial annotations**: Only 60% of images have polylines; ignoring non-annotated samples in seg loss prevents coupling to unlabeled negatives.
- **IN-22k pretraining > IN-21k > IN-1k**: ConvNeXt-S IN-22k 0.9499 OOF beats EfficientNet-V2-S IN-21k 0.9223 OOF (same SegAux pipeline). Prioritize pretraining scale over backbone family.
- **Rank-averaging TTA + folds**: Beats arithmetic mean by 0.01–0.02 AUC; implement early when multi-fold budget exists.
- **Patient-level GroupKFold (SEED=42)**: 86.9% of patients repeat; random splits inflate AUC ~0.02–0.03.
- **Per-class pos_weight in BCE, capped at 10.0**: Robust to 250× imbalance; uncapped causes instability.
- **GeM pooling (p≈3.0)**: Consistently +0.002–0.004 AUC.
- **CLAHE → channel replication → ImageNet norm**: Enhances thin tube contrast.
- **ModelEMA + early stop epoch 5–6**: Saves 0.005–0.01 AUC vs. unregularized longer training.
- **Persistent fold-prediction caching**: Enables resume across tuner calls and clean rank-average aggregation at submit.

## Failure patterns
- **Robust Asymmetric Loss + margin loss on imbalanced multi-label**: Saturated sigmoid on 3/11 classes, AUC→0.5 (Run 20260423 C2 killed). RAL designed for single-label/balanced; stick with weighted BCE.
- **Submission-columns bug**: C1 wrote 9 of 11 label columns, destroying grading until late fix. Validate column count pre-submit.
- **EMA state not fully restored on fold-resume**: Blocks fold-1 convergence; serialize EMA shadow weights with checkpoint.
- **Manager cap on backbone families (≤2)** blocked CXR-pretrained ResNet50 (C4 fallback) despite local weights existing. Lift cap or prioritize domain-pretrained single backbone.
- **Weak ensemble members degrade stacking**: Enforce member OOF macro-AUC floor (≥0.95).
- **Design-phase termination with 0 agent calls**: Pipeline must auto-launch surviving candidates post-judge-filter.
- **Tuner degradation from simultaneous hyperparameter changes**: Adjust 1–2 params at a time.
- **TTA shape-mismatch / dual-stream bugs**: Validate output shapes before submission.
- **Large models (B5/B7, ResNet200D, Swin-Base)**: Exceed 30-min budget or unstable LR/batch interaction.

## Feature engineering playbook
- **CLAHE on raw grayscale → channel replicate → ImageNet normalize**.
- **Augmentation**: HorizontalFlip, ±10–15° rotation, brightness/contrast, CoarseDropout. Hold stable mid-tuning.
- **Segmentation masks**: Rasterize polylines into 4-family binary masks; lightweight FPN/U-Net decoder; masked BCE+Dice; weight 0.1.
- **5-TTA stack**: original, H-flip, ±10° rotation, brightness; aggregate via rank-average.
- **Progressive resize 320→512**: +0.03 AUC historically; revisit with SegAux.

## Tuning insights
- **Backbone ranking (with SegAux)**: ConvNeXt-S IN-22k (0.9499) > EfficientNet-V2-S IN-21k (0.9482) > DenseNet169 (0.9296) > MaxViT (0.9301) > ConvNeXt-V2 distill (0.8726).
- **Resolution sweet spot**: 512×512; 384×384 ~30% faster, minor loss; 640×640 overfits.
- **Epochs**: Peak at 5–6 with EMA; longer only helps without EMA.
- **Seg loss weight 0.1**: Sweet spot; higher dominates classification gradient.
- **Discriminative LRs (1e-4 / 3e-4 / 1e-3)**: Beat single LR.
- **Single-task CNN + SegAux** beats multi-task transformers and distillation under 30-min budgets.

## Open questions
- **Gap to bronze is 0.0072 AUC** (0.96375 → 0.9709): SegAux ceiling near ~0.965 with single backbone; need stacking of ≥2 strong (≥0.95 OOF) SegAux models with diverse pretraining (IN-22k + CXR).
- **CVC-Borderline systematic floor (0.79–0.81 per-class AUC)**: Unresolved across SegAux, RAL, hard-neg mining, architecture swaps. Try CutMix/label-aware mixup, CXR-pretrained backbone, or class-specific curriculum.
- **CXR-domain pretraining (CheXpert/NIH/MIMIC)**: Local weights confirmed available; blocked by backbone-family cap. Highest-priority untried lever.
- **Multi-fold ConvNeXt-S SegAux ensemble**: Only partial fold combination achieved; complete 5-fold rank-average likely +0.003–0.005.
- **AMP for V2-M/L with SegAux**: Could fit larger backbones in budget.
- **Pipeline robustness**: Validate submission column count; persist EMA shadow weights in checkpoints; auto-launch surviving candidates.
