# Task Memory: hms-harmful-brain-activity-classification

## Task profile
- Type: Multi-class soft-label classification (6 classes: seizure, LPD, GPD, LRDA, GRDA, other)
- Dataset: ~15,396 unique EEG IDs from ~1,910 patients (~96k rows with duplicates); dual modality — pre-computed Kaggle spectrograms (4 brain regions: LL/RL/LP/RP) + raw EEG parquets (18–20 channels); soft labels from expert vote counts; heavy class imbalance (42% "other", LRDA as low as ~847 samples)
- Metric: KL-divergence (lower is better)
- Medal thresholds: Bronze ≈ 0.375

## Best result
- Score: 0.4947 | Medal: none | Run: 20260307_205025
- Approach: Two-branch multimodal EfficientNet-B0 — spectrogram branch on 4-region spectrograms + EEG branch with 1D convolutions on 18-channel double-banana montage. Late fusion via concatenation. KL-divergence loss weighted by √(vote_count). Single-fold validation.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_142210 | 0.637 | none | EfficientNet-B1, 8-ch dual spectrogram, 5-fold CV + TTA |
| 20260214_125453 | 0.576–0.578 | none | ConvNeXt-Tiny/EfficientNet-B2, two-stage training, vote weighting |
| 20260215_105913 | 0.665 | none | WaveNet + EfficientNet-B0 ensemble (weaker run) |
| 20260217_114430 | 0.688 | none | EfficientNet-B0 spectrogram-only, 5-fold, undertrained (7 epochs) |
| 20260221_192104 | 0.706 | none | EfficientNet-B0 tiled spectrograms, timeout-limited |
| 20260306_172019 | 1.024 | none | 8-ch dual spectrogram, submission possibly wrong candidate |
| 20260307_205025 | **0.4947** | none | **Multimodal EfficientNet-B0 (spec + raw EEG 1D CNN)** |
| 20260325_123757 | 1.719 | none | EfficientNet-B0 8-ch, severe timeout; OOF plateau 0.62–0.67 |

**Trend**: Spectrogram-only approaches consistently plateau at KL ≈ 0.58–0.72. The only run to break below 0.55 was the multimodal run (20260307) that fused raw EEG via 1D CNN with spectrograms, reaching 0.4947. Still 0.12 above bronze.

## Proven strategies
- **Multimodal fusion is mandatory**: Spectrogram + raw EEG 1D CNN fusion (0.4947) decisively beat spectrogram-only (~0.58–0.72). Every spectrogram-only variant across 8 runs hit the same ceiling regardless of backbone (B0/B1/B2, ConvNeXt-Tiny, EfficientNetV2-S).
- **Double-banana bipolar montage for EEG**: Differential signals between electrode pairs reduced noise; clinically meaningful and consistently used in best-performing candidates.
- **Patient-level GroupKFold**: Mandatory — multi-sample patients cause silent leakage with random splits. Confirmed across all runs.
- **KL-divergence loss with vote-count weighting**: Direct metric optimization + √(votes) or votes/mean weighting focuses learning on high-agreement samples. Table-stakes for this task.
- **Soft label normalization**: votes → probability distribution (votes/total_votes). One-hot or raw vote counts fail badly.
- **Log1p + per-region z-score normalization** on spectrograms: Consistently outperformed raw intensity or min-max scaling.
- **EfficientNet-B0 as workhorse backbone**: 5.3M params, 3–8 min/epoch, enabled rapid iteration; larger models (B2/B4, ViT) added no benefit within time budgets.
- **Precomputed spectrogram caching**: Eliminated I/O bottleneck (30 min/epoch → 2–3 min), critical for iteration speed.
- **SpecAugment + left-right brain flip augmentation**: Confirmed as most impactful augmentations.

## Failure patterns
- **Spectrogram-only ceiling at KL ≈ 0.58–0.72**: Persistent across all runs, all backbones, all augmentation schemes. This is the single most important finding — no amount of tuning breaks this ceiling.
- **EEG-as-spectrogram does NOT help**: Adding EEG mel-spectrograms or custom spectrograms as extra channels (8-ch input) provided zero or marginal lift over Kaggle spectrograms alone (runs 20260306, 20260325). The EEG signal must be processed differently (e.g., 1D CNN on raw waveform).
- **Vision Transformers underperform CNNs**: ViT/TinyViT killed in two runs (20260215, 20260307) — overfit on ~15K samples, hyperparameter-sensitive.
- **1200s coding timeout kills complex candidates**: Dual parquet I/O + montage transforms + multi-branch architectures frequently exceed wall-clock budgets. Precompute aggressively.
- **fp16 training causes inf/NaN in KL loss**: Multiple runs hit this. Use bfloat16 or float32 for divergence-based losses.
- **All-rows (96K) vs. deduplicated (15K) trade-off**: Using all rows causes overfitting (KL 0.93→1.14); deduplication necessary but leaves model undertrained.
- **Two-stage training (all data → high-confidence) is marginal**: Stabilized training but did not lift the architecture ceiling in any run.
- **Tuning saturation**: Label smoothing, dropout, mixup, weight decay all explored extensively — marginal gains at best once architecture ceiling is reached.
- **5-fold CV does not help when per-fold KL is uniform**: All folds converge to identical ~0.63, confirming model is capacity-limited, not overfit.

## Feature engineering playbook
- **Spectrogram input**: 4-channel (LL/RL/LP/RP regions), log1p transform, center crop around `spectrogram_label_offset_seconds`, resize to 128×256, per-channel z-score. Skipping these degrades KL by 0.1–0.3.
- **EEG input**: 18-channel double-banana bipolar montage processed via 1D convolutions (NOT converted to spectrogram — that doesn't help). Raw waveform temporal features complement spectrograms.
- **Window alignment is brittle**: Off-by-one in offset extraction causes silent ~0.05 KL degradation.
- **NaN handling**: Fill NaN in spectrograms before normalization.

## Tuning insights
- **AdamW + cosine annealing** is the default optimizer combo; no run found alternatives superior.
- **GeM pooling > global average pooling** for EfficientNet backbone (confirmed run 20260217).
- **Dropout 0.4** worked well; higher caused underfitting.
- **15–20 epochs needed for convergence**; most runs were budget-limited to 5–10 epochs, leaving KL on the table.
- **Batch size limited by multi-branch memory**: Dual-branch models forced gradient accumulation; single-branch allows larger batches.
- **Multi-scale temporal inputs (3× resolutions)** are memory-inefficient (3–5× slower epochs with B4); single-scale more practical.
- **EMA checkpointing** preserves best model per fold — use it.

## Open questions
- **How to properly extract EEG features**: Top solutions reportedly use bandpass filtering (0.5–20 Hz), ICA artifact removal, frequency band power features (delta/theta/alpha/beta/gamma), or spatial filtering (Laplacian montage). None of these were attempted across 8 runs.
- **Learned STFT / CWT front-end**: Trainable frequency decomposition (designed in run 1, never implemented) could outperform fixed EEG-to-spectrogram pipelines.
- **Multi-task / auxiliary losses**: Training on seizure presence/absence or LRDA subtypes as auxiliary tasks may force richer EEG representations.
- **Ensemble of diverse backbones**: All runs used single-family ensembles (multiple folds, same architecture). Cross-architecture ensembling (ResNet + EfficientNet + 1D CNN) via learned gating or stacking was never tried.
- **3D convolutions**: Fusing time × frequency × spatial dimensions in a single kernel may capture cross-modal interactions missed by separate 2D + 1D branches.
- **Closing the 0.12 gap to bronze**: Best score 0.4947 vs. target 0.375. Likely requires: (a) proper EEG domain preprocessing, (b) multi-fold ensembling of the multimodal architecture, and (c) TTA — none of which were combined in a single run.
