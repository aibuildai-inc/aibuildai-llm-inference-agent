# Task Memory: hubmap-kidney-segmentation

## Task profile
- Type: Semantic segmentation (binary, histopathology WSI)
- Dataset: 12 training TIFFs (13K–50K px/side), 10 test; glomeruli ~1–3.65% foreground; ~2,719 glomeruli total; mixed fresh-frozen vs FFPE stains; 3 TIFF layouts (HWC interleaved, CHW planar, multi-page grayscale, YCbCr/JPEG compressed)
- Metric: Dice coefficient, higher is better
- Medal thresholds: Bronze 0.9439 (median ~0.94)

## Best result
- Score: 0.94426 (mlebench grade, val Dice 0.9542) | Medal: Bronze | Run: 20260302_002937
- Approach: UNet + EfficientNet-B4, 512×512 tiles, Dice+BCE (0.5/0.5). Critical breakthrough: pre-load all 12 TIFFs to RAM (~20–25GB) before training, eliminating rasterio I/O bottleneck (epoch time 40min → 2–4min). TTA + threshold optimization added ~3% Dice post-training.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_043135 → 20260214_222343 | N/A | — | All coder timeouts at 1200s; no training |
| 20260215_191723 | 0.9443 | Bronze | UNet+SE-ResNeXt50, 1024 tiles, Focal+Dice, 3-phase LR decay + best/latest ensemble |
| 20260217–20260218 (multiple) | N/A | — | Persistent 1200s timeouts; TIFF I/O bottleneck |
| 20260219_002836 | 0.9494 val | ungraded | DeepLabV3+ R50 ASPP, 68 epochs, Lovász+Dice+BCE |
| 20260221_234434 | 0.9445 | Bronze | UNet++ ResNeXt50; pre-extracted JPG/PNG tiles offline; 8-fold TTA + Gaussian stitching |
| 20260222_224432 | 0.9406 | — (above median) | UNet++ ResNeXt50, 512 tiles, Focal+Dice, stain normalization |
| 20260226_190737 | 0.9353 | — | UNet++ SE-ResNeXt50 checkpoint resume; inference overlap 192→256 + Gaussian TTA |
| 20260227_211852 | 0.9414 | — | UNet++ SE-ResNeXt50, two-phase pre-cache pipeline, cortex-mask filtering |
| 20260301_063114 | 0.93977 | — | UNet++ ResNeXt50 + deep supervision (val 0.9520) |
| 20260301_114909 | 0.94106 | — | UNet++ SE-ResNeXt50, Focal+Dice, CLAHE (val 0.9515) |
| **20260302_002937** | **0.94426** | **Bronze** | **UNet+EffB4, RAM pre-load, 512 tiles, TTA+threshold** |
| 20260302_031343 | 0.94265 | — | DeepLabV3+ EffB7, 1024 tiles, ASPP multi-scale |
| 20260321_154245 | 0.93866 | — (above median) | UNet+EffB4, BCE+Dice, 512 tiles, checkpoint resume across 6 tuner runs |

## Proven strategies
- **Pre-load/pre-extract TIFFs**: RAM pre-load (20–25GB) OR offline JPG/PNG tile extraction; eliminates rasterio per-tile I/O which dominates wall-clock. Single biggest speedup (e.g., 11hr→5min/epoch in 20260221).
- **512×512 tiles**: Sweet spot for ~290px mean glomeruli diameter; balances context, batch size, and tile count (~50k). 1024 works when paired with large encoders (DeepLabV3+ B7) and full RAM.
- **UNet++ or DeepLabV3+ > vanilla UNet > FPN** when not RAM-constrained; vanilla UNet+EffB4 wins when data-scarce (12 imgs).
- **Focal+Dice or Dice+BCE (0.5/0.5)**: Reliable for 1–3.65% foreground. Focal γ=2 addresses imbalance directly.
- **Encoders**: EfficientNet-B4 (19M) and SE-ResNeXt50/ResNeXt50 (~28M) dominate; B5–B7 feasible only with strong I/O setup.
- **Inference stack**: 8-fold D4 TTA + Gaussian-weighted tile stitching + threshold sweep reliably adds +1–3% Dice (tile-level often 0.97 while full-image 0.92 — stitching is the gap).
- **Checkpoint resume**: Essential under 1200s coder/tuner timeouts; enables multi-phase LR decay (0.9393→0.9430→0.9443 in 20260215).
- **Patient/image-level CV split**: Only 8–9 unique patients; random split leaks.

## Failure patterns
- **1200s coder timeout (dominant failure across ~10 runs)**: Agents burn time exploring heterogeneous TIFF formats before writing code. No code reaches training. Mitigation: pre-provide unified TIFF reader template; pre-extract tiles; raise timeout to 2000–3000s.
- **Transformers (MiT-B2/B3, Segformer) collapse on 12 images**: Sanity checks 0.04–0.77 Dice; data-hungry, no inductive bias for fine boundaries.
- **MAnet + Lovász theoretical match to 1st place → Dice 0.0008** (20260301_114909): repeated unexplained collapse suggests Lovász warmup/scheduling is fragile on tiny data.
- **FPN+SE-ResNeXt unstable**: 0.7006 in 20260219 vs 0.9500 in 20260301; loss/attention interaction suspected.
- **Large models (EfficientNet-B6/B7, ConvNeXt-Base, ResNet101) + 1024 tiles**: OOM or per-epoch budget overruns.
- **Manager "one action per iteration"** blocks parallel coder launches; forces serial timeouts.
- **RLE encoding off-by-one / float32 JSON serialization / autocast kwargs**: recurring silent-failure bugs worth unit-testing.

## Feature engineering playbook
- **Unified TIFF reader first**: detect HWC / CHW / multi-page-grayscale / YCbCr-JPEG; convert to uniform 3-ch HWC uint8 at load; rasterio+GDAL+tifffile+imagecodecs all needed. Use OpenCV with `OPENCV_IO_MAX_IMAGE_PIXELS` for speed.
- **Positive-tile oversampling (~0.7 ratio)**: non-negotiable for 1–3.6% foreground; random tiling collapses to trivial.
- **Cortex mask filtering** (from competition GeoJSON): focuses on glomerulus-rich regions.
- **Stain normalization (Macenko/Reinhard) or CLAHE**: handles FFPE vs fresh-frozen divergence; more effective than geometric aug alone.
- **Aug stack**: D4 flips/rotations, elastic, grid distortion, color jitter; mask-morphology erosion/dilation for boundary variation.
- **Inference**: stride ≤ tile/2, Gaussian blend kernel, 8-fold D4 TTA, post-hoc threshold sweep (~0.503 optimal in 20260219), morphological small-object removal.

## Tuning insights
- **Multi-phase LR decay (baseline → LR 1e-5 → continue from best)**: predictable +0.004–0.005 Dice (20260215).
- **Best-checkpoint + latest-checkpoint 50/50 blend**: cheap +0.001 Dice, no retraining.
- **Threshold sweep post-training**: +0.001–0.008 Dice consistently.
- **Deep supervision (UNet++ aux losses)**: implicit regularization critical on 12 images (20260301_063114 val 0.9520).
- **Inference optimization > retraining near saturation**: overlap 192→256 + Gaussian blending closed 20260226's gap (tile Dice 0.972 vs full 0.924).
- **Cosine warm-restart**: enables clean checkpoint resumes.
- **Batch 6–8 on A100 at 512×512** with EffB4/SE-ResNeXt50.

## Open questions
- Can ensemble of top-3 candidates (e.g., UNet+EffB4 + UNet++ResNeXt50 + DeepLabV3+) exceed 0.945 toward silver? Best run had only 1 alive candidate at end.
- Why does Lovász loss repeatedly collapse (MAnet 0.0008, candidate 2 divergence in 20260321) despite being the 1st-place competition loss? Test BCE warmup first, then phased Lovász.
- Would per-image LOO CV + learned per-image thresholds beat global 0.5 threshold?
- Can domain-pretrained encoders (TCGA histology) beat ImageNet init?
- Is the ~2% tile-level vs full-image Dice gap reducible below 1% with learned stitching or CRF refinement?
