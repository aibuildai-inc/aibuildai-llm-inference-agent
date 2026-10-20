# Pattern: medical-image-classification

## When this applies
- Medical imaging tasks (CT, MRI, X-ray, dermoscopy, histopathology, mammography) with binary or multi-label classification
- Small-to-medium datasets (200–50K studies/patients) with severe class imbalance (1–10% positive)
- Patient-level grouping required; multiple images per patient common
- Calibration-sensitive metrics (pF1) or AUC-based metrics

## Proven patterns
- **GroupKFold/StratifiedGroupKFold by patient_id is non-negotiable**: prevents 0.02–0.20 AUC inflation (rsna-breast, siim-isic 44% patient overlap, ranzcr 86.9% patient repeat, melanoma)
- **Frozen ImageNet backbone + classical head as safe baseline**: rsna-miccai (CV 0.60–0.63), rsna-spine (0.69 best), histopathology (early runs); fast and resistant to overfit on <1K samples
- **Two-stage feature-cache + lightweight aggregator**: rsna-spine (EffNet-B0 features → BiGRU 0.647), siim-isic (GeM + metadata MLP); enables iteration within budget
- **EMA (decay 0.999–0.9995) + label smoothing (0.05)**: consistent +0.005–0.03 lift across histopathology (0.9981 gold), siim-isic (0.9491 gold), ranzcr (0.96375)
- **TTA (8-fold flips/rotations) + rank averaging**: +0.005–0.02 AUC; histopathology, ranzcr, siim-isic all confirm rank > arithmetic
- **Multi-task auxiliary heads regularize tiny positive sets**: ranzcr SegAux (+0.027 AUC), siim-isic 9-class diagnosis (+0.02–0.03), rsna-breast density/biopsy aux
- **Domain preprocessing pipelines**: CLAHE + windowing (CT/X-ray), VOI LUT + MONOCHROME1 invert + Otsu crop (mammo), DICOM→PNG persistent cache (saves 1–2h/round)

## Common pitfalls
- **3D CNNs almost always fail under budget**: rsna-spine, rsna-miccai, siim-covid all killed for OOM/timeout/overfit on <1K studies
- **Vision Transformers fail at small input sizes**: histopathology DeiT at 96×96, Swin-V2-T at 256px (0.890 vs 0.945 EVA02 at 448px)
- **Submission/calibration bugs dominate test gap**: siim-covid (0.77 internal → 0.46 test), rsna-breast (val 0.41 → test 0.07, 5–27× gap), siim-isic submission picked wrong candidate (lost silver)
- **In-graph learnable temperature collapses**: rsna-breast self-certifying revert loop; use held-out calibration slice instead
- **Aggressive resampling without calibration**: pos_weight 47× + WeightedRandomSampler skews inference mean 20× above base rate (rsna-breast)
- **Focal/ASL/PolyLoss often regress vs weighted BCE**: ranzcr RAL collapsed to 0.5, siim-isic ASL 0.914 vs focal 0.945, histopathology pos_weight sufficient
- **MIL/attention with <1K bags doesn't converge**: rsna-miccai gated attention failed, rsna-spine BiGRU overfits
- **TEST_MODE/DEBUG flags leaking into production**: rsna-breast collapsed 3× this way

## Model selection guide
- Dataset <1K studies → frozen backbone + LightGBM/GBM (rsna-spine 0.69, rsna-miccai 0.60–0.63 baseline)
- Dataset 1K–10K, fine-tuning viable → EfficientNet-B0/B2 or ResNet-18 with two-phase unfreezing (rsna-miccai 0.5977 silver, siim-covid)
- Dataset >10K with strong augmentation → EfficientNet-B2/B3 or ConvNeXt-Small IN-22k (histopathology 0.998 gold, ranzcr 0.964)
- Dermoscopy/natural-image-like → EVA02-Base IN-22k MIM at 448px (siim-isic 0.949 gold beats Large at 560px)
- Avoid: 3D CNNs, ViTs <224px input, single-stage detectors for variable-scale objects, frozen backbone for >10K data

## Feature engineering patterns
- **Domain windowing first**: bone (400/1800) for CT, VOI LUT for mammo, multi-window stacking as 3-channel RGB
- **Center/ROI crop > full-image**: histopathology center-aware, rsna-breast ROI-Crop ConvNeXt
- **Metadata fusion via embeddings + MLP concat after GeM**: siim-isic +1–2% AUC, ranzcr per-class pos_weight
- **2.5D pseudo-RGB (3 adjacent slices)** beats single-slice and full 3D for volumetric data (rsna-spine, rsna-miccai)
- **Persistent PNG/npy cache**: mandatory; per-epoch DICOM decoding kills throughput

## Tuning patterns
- **Resolution sweet spot 448–512px**: histopathology 96², ranzcr 512², siim-isic 448²; >560px rarely beats well-tuned smaller
- **Capacity 7–15M params optimal**: EfficientNet-B2/B3, ConvNeXt-Small, DenseNet-121; larger overfits
- **Epoch budget 5–8 with patience 2–3 + EMA**: longer plateaus or overfits across all 6 tasks
- **bf16 autocast for ViTs** (fp16 NaNs on EVA02/CAFormer); grad clip 1.0 + 200-step warmup
- **Layer-wise LR decay**: head 1e-4 to 2e-4, backbone 1e-5 to 2e-5
- **Change 1–2 hyperparameters per tuner round**: ranzcr degraded with 7 simultaneous changes
- **Multi-fold rank-averaged ensemble of diverse architectures > single model tuning past plateau**: histopathology 4-model ensemble (0.998), siim-isic 5-fold (0.949), ranzcr ConvNeXt + EffNet-V2 SegAux
