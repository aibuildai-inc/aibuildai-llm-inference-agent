# Pattern: medical-image-segmentation

## When this applies
- Small dataset (12–76 cases), pixel/voxel imbalance (1–4% foreground or 70% empty)
- Heterogeneous file formats (multi-layout TIFF, 16-bit MR PNG) requiring unified loaders
- Dice or Dice+Hausdorff metrics with strict val→submission parity demands

## Proven patterns
- **Pre-load/pre-extract data eliminates I/O bottleneck**: hubmap RAM pre-load 40min→2–4min/epoch (run 20260302_002937, 0.94426 Bronze); offline JPG/PNG tile extraction had similar effect (20260221, 0.9445).
- **EfficientNet-B2/B4 + U-Net is the reliable backbone**: hubmap winner used EffB4 (0.94426); UW-Madison best used EffB2 2.5D (0.7653). Both beat heavier ConvNeXt/MiT/B6-B7 variants on small data.
- **Dice+BCE (0.5/0.5) or Focal+Dice**: hubmap 0.94426 used Dice+BCE; UW-Madison stable runs used BCE+Dice. Reliable across foreground ratios.
- **Inference stack adds +1–3% Dice**: D4/mirror TTA + Gaussian-weighted stitching + post-hoc threshold sweep (hubmap 20260219 threshold 0.503; consistently +0.001–0.008).
- **Patient/case-level GroupKFold is non-negotiable**: random splits leak (8–9 patients hubmap, 76 cases UW-Madison).

## Common pitfalls
- **Transformer encoders (MiT-B2/B3/B4, SegFormer) collapse on small data**: hubmap Dice 0.04–0.77; UW-Madison mit-b4 detector regressed to 0.4762.
- **Lovász loss is fragile**: hubmap MAnet+Lovász→0.0008; FPN+EffB3+Lovász UW-Madison 0.7028. Needs BCE warmup if used.
- **Multi-axis/3D TTA regresses without matched loss**: UW-Madison xz/yz TTA -3.6% (0.7471→0.7122).
- **Sanity-test parity ≠ production validity**: UW-Madison C1 parity_delta=0.0 then collapsed 0.7653→0.0874. Add real Dice/Hausdorff in sanity, not just RLE round-trip.
- **1200s coder timeout kills heterogeneous-format exploration**: ~10 hubmap runs lost; ≥2000s and pre-provided readers needed.
- **ImageNet normalization on raw medical intensities**: UW-Madison capped Dice at 0.30 (run 20260421_182549).

## Model selection guide
- ≤15 images, binary mask → UNet+EffB4 (hubmap 0.94426) or UNet++ ResNeXt50 (0.9445)
- ~76 cases, multi-class 2.5D → nnU-Net v2 auto-config FIRST (UW-Madison 0.4484 beat 5 custom designs); EffB2 2.5D as fallback (0.7653)
- Avoid: vanilla UNet (underperforms UNet++/DeepLabV3+ when data ample), transformers, 3D volumetric on <100 cases, MAnet+Lovász, ConvNeXt-Tiny/Base

## Feature engineering patterns
- **Unified loader is task-zero**: hubmap needs HWC/CHW/multi-page/YCbCr-JPEG handler; UW-Madison needs 16-bit percentile clip 0.5/99.5 + foreground z-score (NOT per-image-max, NOT ImageNet stats).
- **Positive oversampling ~0.7**: required for 1–4% foreground (hubmap); class weights large_bowel 2.0–2.5× (UW-Madison).
- **2.5D 3-slice stacking** for volumetric tasks (UW-Madison); **512×512 tiles** for WSI (hubmap sweet spot at 290px mean object).
- **Aug stack**: D4/HFlip + ShiftScaleRotate + ElasticTransform + GridDistortion; stain norm (Macenko) for histology, no color aug for MR.

## Tuning patterns
- **Multi-phase LR decay + checkpoint resume**: hubmap +0.004–0.005 (20260215, 0.9393→0.9443); essential under timeouts.
- **Best+latest checkpoint blend**: free +0.001 Dice.
- **SWA + EMA**: UW-Madison +0.005–0.007 each.
- **Threshold sweep post-training**: +0.001–0.008 reliably; per-class thresholds for multi-class.
- **Inference optimization > retraining near saturation**: hubmap stride/Gaussian closed 0.972 tile vs 0.924 full gap.
- **Val→submission gap is the dominant blocker on multi-class**: UW-Madison gap ranged -0.05 to -0.68; instrument calibration parity BEFORE architectural changes.
