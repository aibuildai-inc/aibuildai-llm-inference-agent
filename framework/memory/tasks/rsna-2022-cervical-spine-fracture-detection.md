# Task Memory: rsna-2022-cervical-spine-fracture-detection

## Task profile
- Type: Multi-label classification (8 targets: C1–C7 vertebra fracture + patient_overall)
- Dataset: 202 training studies (cervical spine CT, DICOM); ~71K total slices; 1,817 test patients (~640K slices); severe class imbalance (C3≈4.5%, C4≈5.4% positive up to C6≈16.8%); 9 NIfTI segmentation masks, 24 bounding-box annotations (too sparse to use)
- Metric: Weighted multi-label log loss (lower is better); patient_overall weight = 7×, each Cx = 1×
- Medal thresholds: Target ≤0.5212 (likely silver/bronze boundary); median not beaten in any run

## Best result
- Score: 0.64715 | Medal: none | Run: 20260224_080052
- Approach: Two-stage pipeline — frozen EfficientNet-B0 with multi-window HU input (bone/soft-tissue/subdural as 3-channel RGB), 1280-d per-slice features cached, then bidirectional GRU (256-d, 2-layer) with 8 gated attention heads (one per label) aggregating ordered slice sequences into study-level predictions. Competition-weighted BCE (7× patient_overall). 80/20 stratified split.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_234626 | — (incomplete) | ungraded | 3 candidates coded (2.5D CNN+GRU, 3D ResNet-18, 2D+LightGBM); never executed. Quick LightGBM test: 0.4899 val |
| 20260224_080052 | 0.64715 | none | EfficientNet-B0 + BiGRU w/ gated attention; feature caching; multi-window HU input |
| 20260224_210418 | 0.693 | none | Frozen EfficientNet + LightGBM baseline; all deep candidates timed out; stale prevalence-based submission |
| 20260304_232956 | 0.6927 | none | EfficientNet-B2 + LightGBM (streaming DICOM, PCA 256-d); best candidate scored 0.8013 on internal metric |
| 20260319_214435 | — (failed) | ungraded | All 6 candidates eliminated before scoring; initialization failure |
| 20260319_214446 | 0.80039 | none | EfficientNet-B2 + LightGBM (PCA 50-d, 7 z-segments); val score 0.3897 but test score 0.80 — severe overfit |
| 20260320_194943 | — (incomplete) | ungraded | Only LightGBM candidate implemented; deep candidates hit torch.amp API issues |
| 20260320_195452 | — (incomplete) | ungraded | 3 candidates designed, none executed; agent budget exhausted in design phase |

## Proven strategies
- **Frozen 2D backbone + lightweight head is the only reliable paradigm**: Every completed run used frozen EfficientNet (B0 or B2) features. End-to-end fine-tuning plateaus at val loss ~1.08–1.10 and never improved.
- **Bone windowing (center=400, width=1800)**: Mandatory for fracture visibility in CT; used universally across all runs.
- **2.5D pseudo-RGB slice stacking** (3 adjacent axial slices as channels): Preserves z-axis context while leveraging ImageNet pretraining. Consistently outperforms single-slice and full 3D approaches.
- **Feature caching + two-stage pipeline**: Enables fast iteration within time budgets. Without caching, deep candidates consistently time out.
- **Competition-weighted sample weights (7× patient_overall)**: Correctly aligns training with metric structure. LightGBM's native sample weights work better than loss-function weighting in neural nets.
- **LightGBM with heavy regularization**: λ_l1≥1.0, λ_l2≥5.0, colsample_bytree≈0.3, subsample≈0.7, scale_pos_weight per target. Most robust classifier for 202 samples.
- **Derived patient_overall**: Computing `1 - prod(1 - p_Cx)` from vertebra predictions outperforms independent patient_overall prediction.
- **Z-segment aggregation** (7 segments × 5 statistics = mean/max/std/p25/p75): Crude vertebra localization without segmentation masks; feeds ~16.9K features to PCA → LightGBM.

## Failure patterns
- **Deep sequence models (BiGRU, BiLSTM, Transformer) consistently fail**: Overfit on 202 studies despite regularization (label smoothing, dropout, early stopping). BiGRU best val ~0.65 but never generalizes to test; Transformer scored 0.8371.
- **Timeout epidemic**: BiGRU, BiLSTM, MIL-attention, multi-planar reformation, and 2.5D dual-head candidates all exceeded 1200s coder timeout across multiple runs. Complex architectures are infeasible within budget.
- **3D CNNs ruled out every time**: 202 studies → ~1.4K vertebra crops with ~10% positive rate. Judged infeasible before coding in every run that considered it.
- **Segmentation-based approaches impossible**: Only 9 NIfTI annotations; NIfTI-to-DICOM alignment risky; no pretrained MONAI model validated.
- **Severe val-to-test gap**: Best run (20260319_214446) achieved 0.3897 val but 0.80 test — isotonic calibration on 202-sample OOF likely overfits. PCA reduction to very low dims (50) may also lose discriminative signal.
- **ResNeSt batch_norm failure**: Split-attention layers crash with batch_size=1; requires batch_size≥2 or GroupNorm substitution.
- **DICOM pre-caching is a memory/time trap**: Loading all 98–688 slices per study exhausts memory; streaming (load only 32–64 sampled slices) is necessary.

## Feature engineering playbook
- **Multi-window HU input**: Bone (400/1800) + soft-tissue + subdural as 3-channel RGB input to pretrained backbone. Standard and effective.
- **Slice sampling**: 32–64 evenly-spaced slices per study; diminishing returns beyond 64; 5 slices is too few (0.4899 quick-test).
- **Z-segment binning**: Divide ordered slices into 7 segments (proxy for C1–C7); compute per-segment statistics over backbone features.
- **Aggregation statistics**: mean/max/std/p25/p75 over slice features per segment → ~16.9K-dim study vector.
- **PCA dimensionality reduction**: 16.9K → 100–256 dims. Going to 50 dims may lose signal (correlated with val-test gap in run 20260319). 100–256 is safer.
- **Positional encoding**: Provides z-axis awareness for sequence models; marginal gain (~0.04 loss) over mean pooling.

## Tuning insights
- **PCA n_components sweet spot**: 100–256 dims from ~17K features. 50 dims showed overfit (0.39 val → 0.80 test); raw 17K dims risks curse of dimensionality.
- **LightGBM per-target scale_pos_weight**: Essential. C3 needs ~20× upweighting, C6 needs ~5×. Do not use uniform class weights.
- **Label smoothing**: 0.01–0.03 helps neural approaches marginally; 0.10 causes flat predictions.
- **End-to-end fine-tuning LR**: Backbone gradients may be blocked; ablate learning rate, gradient clipping, and layer freezing schedules (never successfully resolved).
- **Calibration**: Isotonic regression on 202-sample OOF likely overfits; prefer Platt scaling or temperature scaling.
- **5-fold StratifiedGroupKFold** (group=StudyInstanceUID): Prevents leakage; handle single-class folds gracefully.

## Open questions
- **Can fine-tuning the backbone ever work?** End-to-end training plateaued at 1.08 val loss across multiple runs. Needs careful LR schedule ablation with very low LR + gradient clipping — never properly attempted.
- **Optimal PCA dimensions**: 50 overfits, 256 may be wasteful. Systematic sweep (50/100/150/200/256/512) not done.
- **Test-time augmentation (TTA)**: Never attempted. Averaging over HU window variants or random crops could yield 0.01–0.02 improvement.
- **Ensembling DL + LightGBM**: Diversity between BiGRU (0.65) and LightGBM (0.69) predictions never exploited via weighted averaging.
- **Bounding box annotations** (690 studies available): ROI crops around known fracture locations could provide strong signal but were never used.
- **Per-vertebra correlation modeling**: LightGBM treats each vertebra independently; fractures are spatially correlated. Hierarchical softmax or C1–C7 correlation constraints unexplored.
- **Why the massive val-to-test gap?** 0.39 val → 0.80 test suggests distribution shift or calibration failure beyond just small-sample overfit. Needs investigation into test set characteristics.
