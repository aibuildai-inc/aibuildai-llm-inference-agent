# Pattern: brain-signal-analysis

## When this applies
- EEG/biosignal classification with multi-modal inputs (raw waveforms + spectrograms)
- Soft-label / vote-distribution targets evaluated with KL-divergence
- Patient-grouped datasets (~10–20K unique IDs) with class imbalance
- Strict wall-clock budgets (≤1200s coding timeouts)

## Proven patterns
- **Multimodal fusion (spectrogram + raw EEG 1D CNN) beats spectrogram-only**: hms-harmful-brain-activity best run 20260307_205025 hit KL 0.4947 vs. 0.58–0.72 ceiling for all 7 spectrogram-only runs (20260213, 20260214, 20260217, 20260221, 20260306, 20260325).
- **Patient-level GroupKFold is mandatory**: random splits cause silent leakage when patients have multiple windows.
- **KL-loss with √(vote_count) weighting**: directly optimizes metric while focusing on high-agreement labels (used in best run 0.4947).
- **Soft labels via votes/total normalization**: one-hot or raw counts consistently fail.
- **Log1p + per-region z-score on spectrograms**: outperforms raw or min-max in all hms runs.
- **Double-banana bipolar montage** for raw EEG channels: noise reduction, clinically meaningful.
- **Precompute & cache spectrograms**: cut epoch time 30min→2–3min in hms runs, enabling iteration.

## Common pitfalls
- **Spectrogram-only ceiling (~0.58–0.72 KL)**: persists across B0/B1/B2, ConvNeXt-Tiny, EfficientNetV2-S — no tuning breaks it (8 hms runs confirm).
- **EEG-as-extra-spectrogram-channel doesn't help**: 8-ch dual spectrogram runs (20260306, 20260325) gave zero lift; raw EEG must go through 1D CNN.
- **ViT/TinyViT overfit** on ~15K samples (failed runs 20260215, 20260307 ViT branch).
- **fp16 → inf/NaN in KL loss**: use bf16 or fp32 for divergence losses.
- **Using all 96K duplicated rows** overfits (KL 0.93→1.14); deduplicate to unique EEG IDs.
- **Complex multi-branch + parquet I/O** blows the 1200s timeout (run 20260325 hit severe timeout, KL 1.719).
- **Two-stage training (all → high-confidence)**: marginal, doesn't break architecture ceiling.

## Model selection guide
- Default backbone → **EfficientNet-B0** (5.3M params, 3–8 min/epoch); larger (B2/B4) gave no lift in hms.
- Spectrogram-only baseline → expect KL ~0.6 floor; do **not** stop here.
- To break the ceiling → add **raw-EEG 1D CNN branch** with late concat fusion.
- Avoid ViT/TinyViT unless dataset >50K.
- Pooling: **GeM > GAP** (confirmed run 20260217).

## Feature engineering patterns
- Spectrograms: 4-region (LL/RL/LP/RP), log1p, center-crop around label offset, resize 128×256, per-channel z-score, NaN-fill before normalization.
- Raw EEG: 18-ch double-banana bipolar montage → 1D conv stack (NOT spectrogram conversion).
- Augmentation: **SpecAugment + left-right brain flip** are the high-impact pair.
- Watch for off-by-one in `*_label_offset_seconds` — silently costs ~0.05 KL.

## Tuning patterns
- **AdamW + cosine annealing** is default; no superior alternative found.
- Dropout ~0.4; higher underfits.
- Need **15–20 epochs** for convergence; budget-limited 5–10 epochs leaves score on the table (run 20260217 undertrained at 7 epochs → 0.688).
- Multi-branch models force gradient accumulation; budget batch size accordingly.
- EMA checkpointing per fold — use it.
- 5-fold CV with uniform per-fold KL = capacity-limited, not overfit → invest in architecture, not regularization.
