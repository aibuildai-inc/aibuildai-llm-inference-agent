# Task Memory: uw-madison-gi-tract-image-segmentation

## Task profile
- Type: 3D medical image segmentation (multi-organ: large_bowel, small_bowel, stomach)
- Dataset: ~76 cases, ~76K slices, 16-bit MR-Linac scans (raw 140–3500/65535); 70% empty masks; large_bowel >> small_bowel; corrupt PNGs in cases 101/102/107
- Metric: dice-hausdorff-combo = 0.4×Dice + 0.6×normalized 3D Hausdorff (higher better); **Hausdorff dominates**, per-slice 2D OR-of-classes, z/num_slices + sqrt(3) normalization
- Medal thresholds: bronze ≈ 0.86699

## Best result
- Score: 0.7653 | Medal: none | Run: 20260307_014030
- Approach: EfficientNet-B2 U-Net 2.5D with grader-aligned metric (z-normalized Hausdorff ÷ sqrt(3)), DataParallel training, 48 epochs. Training plateaued ~0.76 — structural single-stage limitation, not tuning.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260307_014030 | 0.7653 | none | EffNet-B2 2.5D + grader-aligned metric normalization |
| 20260418_194317 | 0.70283 | none | FPN + EffNet-B3 + Lovasz aux |
| 20260418_205951 | 0.60454 | none | Baseline U-Net+EffB1, 2.5D, BCE+Dice, hflip TTA |
| 20260419_162356 | 0.55987 | none | ConvNeXt-Tiny, 384px, deep supervision, EMA |
| 20260420_193428 | 0.71710 | none | +3D morph closing, CC filtering, per-class thresholds, 256px |
| 20260421_182549 | 0.51064 | none | ImageNet norm bug capped Dice at 0.30 |
| 20260421_191223 | 0.65752 | none | Class-specialist binary U-Nets + 5-fold (val 0.878→sub regression) |
| 20260422_163302 | 0.53119 | none | Anchor-EffB1-2.5D w/ RLE validator; val 0.82→sub 0.53 |
| 20260423_164342 | 0.53733 | none | 4-candidate explore; C1 won @ val 0.4945 |
| 20260424_180701 | 0.68891 | none | EffB2-NS 2.5D + aux classifier gate; 2 two-stage candidates timed out at 1200s |
| 20260425_010821 | 0.55308 | none | nnU-Net-style EffB3 2.5D; multi-axis TTA regressed 0.7471→0.7122 |
| 20260425_011525 | 0.4484 | none | nnU-Net v2 auto-config beat custom 2-stage; C1 anchor collapsed 0.7653→0.0874 |

## Proven strategies
- **nnU-Net auto-configuration is the mandatory anchor for medical imaging**: 20260425_011525 C6 nnU-Net v2 (spacing-aware patch/depth tuning from filename metadata) scored 0.4484, beating all 5 custom 2.5D/two-stage designs by 5–60×. Make nnU-Net the default baseline.
- **Two-stage classify-then-segment remains structurally promising but unrealized**: 5 consecutive attempts have failed (20260424×2, 20260425×3 — C2/C4 lost to API 529, C5 cbDice/FiLM scored 0.0072). Pre-cache `positive_slice_indices.npy` offline; consider nnU-Net-classifier + nnU-Net-segmenter variant.
- **Auxiliary classifier head ≠ true two-stage gating**: -0.009 delta; tight encoder coupling defeats gate purpose.
- **nnU-Net-style preprocessing (percentile clip 0.5/99.5 + foreground z-score) > per-image-max** on 16-bit MR.
- **2.5D 3-slice channel-stack** (s-1, s, s+1) sufficient on segmenter; 5-slice marginal; 3D regresses on 76 cases.
- **Mirror-flip TTA only** (~+1-2%); **multi-axis xz/yz TTA regresses 3.6%** without matched boundary-DoU loss.
- **Per-class loss weighting**: large_bowel 2.0–2.5×, small_bowel 1.5× vs stomach.
- **Case-level GroupKFold** on case_id (non-negotiable).
- **SWA + EMA stack** (+0.005–0.007 each); **SGD+poly-LR matches Adam**.
- **Sigmoid multi-label head** > softmax for 3-class; **per-class sigmoid (unified) > specialist binary**.

## Failure patterns
- **Sanity-test parity_delta=0.0 does NOT guarantee production validity**: 20260425_011525 C1 anchor (claimed 0.7653 baseline) passed all parity diagnostics yet collapsed to per-class Dice 0.013/0.012/0.0002 in production. RLE round-trip alone is insufficient — must add actual Dice/Hausdorff metric computation in sanity phase.
- **Detector-crop fragile in production**: 20260425_011525 C3 EfficientDet-D0 scored 0.0261 despite +0.015 Dice prior; mit-b4 transformer detector also regressed (C3: 0.4762, run 20260423). Profile detector confidence + crop coverage on dev set first.
- **Cross-slice attention + differentiable gating + cbDice unstable**: 20260425_011525 C5 scored 0.0072 (worst of run); FiLM modulation or cbDice gradient quality suspect.
- **API 529 overload kills 20–33% of candidates**: 20260424×2, 20260425_011525×2 (C2 EffB3→B5, C4 SegFormer mit-b5) lost mid-coder. Need retry-with-backoff or simpler-design fallback.
- **Monolithic single-stage segmenter plateaus 0.70–0.76** across 8+ runs; architectural tweaks flat or negative.
- **Val-to-submission gap persists (6+ runs)**: 0.4945→0.537, 0.82→0.53, 0.878→0.658, 0.7011→0.6889, 0.7122→0.5531, **0.7653-claimed→0.0874 (20260425_011525, -0.68!)**. Worsening; calibration/parity audit critical.
- **mit-b4 / 3D volumetric on 76 cases**: regresses (init cost + data scarcity).
- **ImageNet norm on 16-bit MR**: caps val Dice 0.30.
- **Combined losses on single-stage** (Dice+Tversky+BCE): degenerate empty-mask predictions.
- **GSL warmup, Hausdorff direct loss, UNet++, 3D refiners, full nnU-Net pipeline (custom-built), pseudo-labels, class-specialist binary U-Nets**: plateau or regress.
- **Design surface dilution**: ≤2-3 candidates w/ 1 pure reproduction preferred; 6-candidate explore (20260425_011525) saw 4/6 collapse or die.

## Feature engineering playbook
- **nnU-Net v2 default baseline**: spacing extracted from filenames, auto patch/depth.
- **Two-stage**: detector crop → classifier (all slices) → segmenter (positive only, gated); cache `positive_slice_indices.npy` offline; consider nnU-Net for both stages.
- **Preprocessing: percentile clip 0.5/99.5 + foreground z-score** > per-image-max.
- 2.5D 3-channel stack (s-1, s, s+1); segmenter 384–512px, classifier 256–320.
- Augs: HFlip, ShiftScaleRotate, ElasticTransform (α=120, σ=6), GridDistortion, Mixup.
- TTA: mirror-flip ONLY by default; validate any axis-TTA on held-out fold first.
- RLE: `order='F'`; submission-parity sanity test day-1 + per-class Dice in sanity (not just RLE round-trip).
- num_workers=0 on corrupt PNG cases (101/102/107).
- Per-channel classifier thresholds + per-class segmentation thresholds swept independently on OOF.

## Tuning insights
- **nnU-Net beats hand-tuned designs**: 20260425_011525 confirmed automated configuration > manual; structural saturation at 0.70–0.76 is for *custom* single-stage only.
- **Sanity test must include metric computation**, not parity-only: C1 passed parity_delta=0.0 then collapsed -0.68 in production.
- **Two-stage prerequisite: lean dataloader + offline positive-index cache**; 5 consecutive two-stage attempts have died.
- **Post-hoc gating shortcut**: train single-stage + standalone classifier separately, compose at inference (avoids joint-training pitfall).
- **Val→sub gap is the dominant blocker** (now -0.68 worst-case): instrument per-channel calibration, test-set normalization parity, case_id ordering BEFORE more architecture tuning.
- **Hausdorff metric (0.6 weight)**: normalization (slice-z + sqrt(3)) parity between val harness and grader unverified.
- **Iteration discipline**: ≤2-3 candidates, 1 reproduction; coder sanity <600s for two-stage; implement API 529 retry-with-backoff.
- FP16 + batch=4 + 2–3 folds; nnU-Net v2 auto-configured is now cheapest competitive baseline.

## Open questions
1. **Root-cause C1 0.7653→0.0874 collapse**: per-class Dice in sanity, RLE column-major audit, val augmentation disabled check. Pre-flight before any retry.
2. **nnU-Net classifier + nnU-Net segmenter two-stage variant**: leverage 20260425_011525 C6's 0.4484 success; could finally unlock the two-stage paradigm.
3. **Hausdorff normalization parity audit** between local harness and Kaggle grader — explains worsening val→sub gap.
4. **API 529 retry-with-backoff in agent loop**: 4 candidates lost across last 2 runs to overload errors.
5. **Multi-axis TTA failure root cause**: z-spacing (3mm vs 1.5mm in-plane) or ensemble weighting?
6. **Per-class threshold sweep + boundary-DoU loss** as direct Hausdorff attack.
7. **Bronze (0.86699) plausible only if val-to-sub gap closes**; nnU-Net auto-config opens new ceiling above custom 0.76.
