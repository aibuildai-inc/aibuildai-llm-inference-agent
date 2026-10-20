# Pattern: natural-disaster-geophysics-prediction

## When this applies
- Seismic/geophysical signal data (waveforms, seismic images) with small labeled sets (3K-4K samples)
- Tasks predicting physical phenomena: time-to-eruption regression, salt body segmentation
- Domain priors matter (gravity, frequency content); raw data needs heavy preprocessing or feature engineering

## Proven patterns
- **Domain-specific feature/loss alignment beats generic approaches**: XGBoost with `reg:absoluteerror` directly optimizing MAE hit Gold (2.47M) on volcanic eruptions; Lovasz-hinge Phase B after BCE+Dice warmup gave +0.025 on TGS Salt
- **Phase A → Phase B training (warmup then domain loss)**: Both tasks benefit — TGS Salt (BCE+Dice → Lovasz), volcanic (default → MAE-aligned). Single-phase training underperforms
- **Stratified k-fold by target distribution mandatory at small sample sizes**: TGS Salt salt-coverage buckets prevented 15-20% inflation; volcanic 5-fold CV with early stopping (patience=100) prevented overfitting on 3,987 samples
- **Signal-processing features (FFT, Welch PSD, Hilbert envelope, spectrogram stats) outperform raw statistics**: Volcanic best run (2.47M) used ~1,297 such features; mean/std baselines plateau higher

## Common pitfalls
- **Deep learning fails at <5K samples**: Volcanic MLP (3.36M), 1D-CNN (failed), LSTM (12.6M) all 1.5-7x worse than GBDT; TGS Salt weak CNN backbones plateau at 0.466
- **Over-engineering regresses scores**: TGS Salt UNet++ + 8-way TTA + post-hoc pseudo-labels: 0.8224 → 0.7811; volcanic XGBoost with 679 unfiltered features: 9.1M MAE
- **Ignoring physical priors**: V-flip/rotation TTA on seismic images violates gravity prior; aggressive waveform downsampling destroys temporal patterns
- **Self-supervised pretraining on grayscale/seismic domain underperforms ImageNet supervised** (TGS Salt FCMAE/SAM candidates killed)
- **Train/test feature mismatch**: TGS Salt depth metadata is zero at test (use only for stratification, never as feature)

## Model selection guide
- Tabular features from time-series → **GBDT (XGBoost/LightGBM) with MAE-aligned objective** (volcanic Gold: 2.47M)
- Image segmentation, ~3K samples → **Transformer backbone (Segformer MiT-B2/B3) at ≥256×256** (TGS Salt best: 0.8449); SE-ResNeXt50 U-Net is best CNN baseline (0.82 tier)
- Avoid: pure deep learning on raw waveforms <5K samples; linear models (volcanic Ridge: 5.4M); SAM+LoRA adapters

## Feature engineering patterns
- **Resolution/scale matters most**: TGS Salt 128→256 was largest single lever (+0.02); volcanic full 60K timesteps preserved beats downsampled
- **Signal decomposition stack**: FFT (spectral centroid/bandwidth/entropy), Welch PSD bands, Hilbert envelope, autocorrelation at multiple lags, rolling stats at multiple scales (500-6000)
- **Null indicators + native tree handling** > complex imputation (volcanic 11-28% missing sensors)
- **Feature count sweet spot 500-1,300 after selection**; correlation filter at 0.98 + zero-variance removal mandatory above 500
- **Heavy augmentation with domain constraints**: HFlip-only, ShiftScaleRotate ≤±15° for seismic images

## Tuning patterns
- **Early stopping patience 20-100** (segmentation 20-30, GBDT 100); models converge in 2K-5K rounds
- **Per-fold threshold tuning** recovers +0.008 on segmentation (TGS Salt fold 4 ≈ 0.58 vs fold 2 ≈ 0.48)
- **SWA in Phase B with cosine→1e-7** stabilizes final convergence (TGS Salt run 20260423)
- **Ensemble via probability/OOF averaging, not majority vote**: All TGS Salt top-5 solutions use soft averaging; volcanic LightGBM+XGBoost OOF blend is the obvious untried lift
- **Family orthogonality > seed diversity**: TGS Salt transformer+CNN gave +0.015 vs +0.003 seed diversity
- **Default GBDT configs already medal**: Volcanic tuning squeezed only ~5% (2,624K → 2,490K); features dominate
