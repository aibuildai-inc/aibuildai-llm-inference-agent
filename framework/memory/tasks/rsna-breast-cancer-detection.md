# Task Memory: rsna-breast-cancer-detection

> ⚠️ Data fix on 2026-04-21 made test_images available. First post-fix run (20260421_190819) produced real test scores. Pre-fix scores (0.02–0.05) were prior-fallback constants — ignore for calibration.
>
> 🚨 **HARD RULE — val_metric formula MUST match grader exactly**. 5 post-fix runs all reported val pF1 0.30–0.59 but mlebench-graded 0.02–0.07 (5–27× gap). Root cause: each candidate computed val with a DIFFERENT formula than the grader (image-level vs breast-level, balanced subset vs full, per-fold-mean vs concatenated). The grader is `pfbeta` (`mle-bench/mlebench/competitions/rsna-breast-cancer-detection/grade.py`):
> ```python
> # On (prediction_id, cancer) pairs aligned to sample_submission.csv (one row per patient_id+laterality):
> pTP = sum(p_i * y_i)
> pFP = sum(p_i * (1 - y_i))
> precision = pTP / (pTP + pFP)
> recall    = pTP / sum(y_i)
> pF1       = 2*precision*recall / (precision + recall)
> ```
> Every train.py MUST compute val_metric using this exact code on breast-level OOF, and end with `assert abs(val_pF1_reported - val_pF1_grader) < 0.02` to catch drift. Submission MUST also pass `assert abs(submission["cancer"].mean() - 0.021) < 0.02` before being written, with logit adjustment or isotonic-on-held-out-slice applied to satisfy the gate. Without these two assertions all val numbers are unreliable.

## Task profile
- Type: Binary classification (image-level → patient-breast aggregation)
- Dataset: ~49K mammography DICOMs (JPEG2000 lossless), 2 sites; ~2.1% positive rate (~213 positive patient-breasts); multi-view (CC+MLO)
- Metric: Probabilistic F1 (pF1), higher better — calibration-sensitive
- Medal thresholds: Bronze ≥ 0.41 pF1

## Best result
- Score: **0.0732 mlebench** (OOF pF1 0.3835) | Medal: none | Run: 20260421_190819 (C1, tuner_run2)
- Approach: ROI-Crop ConvNeXt-Small + multi-task heads (BCE cancer + aux density/biopsy), GroupKFold, balanced sampler, post-hoc temperature + isotonic calibration. Earlier run1 checkpoint hit val pF1 **0.4149/0.4155** (bronze line) but was lost when TEST_MODE collapse overwrote best_model.pth.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_064429 | 0.0487* | none | EffNet-B0 multi-view + metadata; pre-fix |
| 20260303_205731 | 0.0467* | none | ConvNeXt-S 512² + label smoothing |
| 20260303_075948 | 0.0433* | none | EffNet-B4 + BCE pos_weight=48 |
| 20260303_085925 | 0.0398* | none | ConvNeXt-S vs EffNetV2-S; 1200s timeout |
| 20260303_205705 | 0.0222* | none | EffNet-B4 + Focal; val overhead catastrophic |
| **20260421_190819 C1** | **0.0732** | none | ConvNeXt-S ROI-crop + multi-task; bronze-line lost to revert bug |
| 20260421_190819 C3 | 0.0447 | none | EffNetV2 + sigmoidF1 + density CE; DataLoader hang |
| 20260421_190819 C2 | 0.0230 | none | ConvNeXt-V2 frozen + LightGBM stack; no progress |
| 20260423_210347 C1 | 0.0450 | none | ConvNeXt-S + learnable Platt temp head; calibration collapse (mean 0.4254 vs base 0.021) |
| 20260423_210347 C3 | ~0.0 | none | TEST_MODE collapse again (80 train, 14 steps) — constant 0.525 output |

*pre-data-fix fallback constants

## Proven strategies
- **Offline DICOM→PNG persistent manifest cache**: saves ~1.5h/round; non-negotiable
- **VOI LUT → MONOCHROME1 invert → CLAHE → Otsu breast crop → 512²** preprocessing
- **StratifiedGroupKFold by patient_id**: prevents 4-views-per-patient leakage
- **Breast-level max-pool aggregation** over (patient_id, laterality)
- **BCE + pos_weight (47–48×)** more stable than Focal for 49:1 imbalance
- **ROI-Crop ConvNeXt-Small + multi-task aux heads (density/biopsy/invasive)** marginally beats single-task (~+0.03 val pF1); peaked val 0.4155
- **EMA + WeightedRandomSampler** prevents negative-class collapse
- **Versioned `best_model_v{N}.pth` + max-val selector** eliminates save-best-revert (worked this run)

## Failure patterns
- **TEST_MODE collapse — recurring (3rd time)**: production train.py must contain NO `TEST_MODE`/`DEBUG_MODE`/`SMOKE_TEST` flag; tuner must `assert not TEST_MODE` on run 1+. Always expose MAX_TRAIN_IMAGES/MAX_TEST_PATIENTS as config
- **Calibration collapse dominates test pF1**: C1 val 0.295 → test 0.045 because submission mean 0.4254 vs base rate 0.021 (20× too high). pos_weight 47× + WeightedRandomSampler skew training to ~50/50; in-graph Platt layer alone cannot pull inference back to base rate
- **Self-certifying revert loop**: temperature search on same fold as revert validation → T=0.3 collapsed probs → false revert. Use a dedicated calibration slice (5–10% by patient_id) never used for gradients OR T-search; or fix T=1.0 in training and revert on raw pre-temp val pF1
- **DataLoader stuck-inference deadlock**: 8 workers in `do_wait`/`futex_wait`; correlates with /data3 contention. Gate tuner on disk idle; reduce num_workers; hard inference timeout
- **Frozen-backbone + GBDT (C2) fails on pF1**: post-hoc isotonic can't match end-to-end at 2.1% imbalance
- **Three-term loss (BCE + sigmoidF1 + density CE)**: weights unstable; converged only to val 0.318
- **Differentiable pF1 / siamese (C2 20260423)**: unstable without BCE warmup curriculum
- **/data3 I/O starvation kills new candidates (C4 MaxViT)**: ~150 PNG/min decode, exhausts budget before training; saturation environmental, not algorithmic

## Feature engineering playbook
- DICOM metadata (WindowCenter/Width, BodyPartThickness, CompressionForce) as auxiliary fusion only
- ROI-crop before resize (not just Otsu full-breast) improves signal density at 512²
- Multi-task aux supervision (density, biopsy, invasive) regularizes at 2.1% positive rate
- Label smoothing 0.05 on positives marginally reduces FP penalty

## Tuning insights
- **512² is practical ceiling** under 24h budgets
- **pF1 weights absolute probability — calibration is THE bottleneck**: prefer **logit adjustment at inference** (`logits -= log((1-π)/π)`, π=0.021) and/or post-hoc Platt/isotonic on held-out slice over in-graph learnable temperature
- **Sanity gate before submission**: assert `|submission.mean() - 0.021| < 0.02`
- **Convergence needs ≥3000 wall-clock steps**; ConvNeXt-S peaked step 3400 in run1; C1 20260423 stopped at step 1318 (2× slower per-step under I/O contention)
- **sigmoidF1 alone insufficient**; needs BCE warmup curriculum (10 ep BCE → linear ramp)
- **Batch 32 + grad accumulation** at 512²
- Pre-decode entire 49k corpus in offline init step before any tuner call

## Open questions
- Recover lost run1 best_model.pth (val 0.4149) from snapshot — alone likely scores bronze
- **Logit adjustment + held-out calibration slice**: highest-leverage fix per latest evidence
- Frozen post-hoc Platt (fitted after final fold) vs in-graph learnable temperature
- Differentiable pF1 (Yacouby-Axman) with BCE warmup curriculum on EffNetV2 siamese
- Site-1 JPEG2000 decoding (pylibjpeg + gdcm fallback) to expand 21K→49K
- Hard timeout + last-checkpoint fallback on inference phase (20-min cap)
- Exclusive disk-I/O gating for /data3 — every stuck-inference + decode-starve correlates
- Focal/OHEM re-evaluation post-data-fix at 2.1% positive rate
