# Pattern: document-text-image-analysis

## When this applies
- High-resolution document/manuscript imagery (historical text, papyrus scans) requiring pixel-level or character-level localization
- Tasks with severe distribution shift across few source documents/fragments (≤3 training fragments)
- Long-tail class distributions or volumetric (Z-stacked) inputs from imaging modalities (CT, X-ray)
- Metrics that reward precision (F0.5, point-in-box F1)

## Proven patterns
- **Two-stage decoupling beats end-to-end** when class count is large: kuzushiji-recognition went 0.096→0.94 by separating class-agnostic detection from N-class classification
- **Domain-aware front-end + 2D decoder with depth pooling** for volumetric inputs: 5/5 Vesuvius top-5 teams use 3D-aware encoders + mean/max depth-pool; our 2D-z-as-channels capped at F0.5 0.54 vs ~0.67 cluster
- **Tight central windows beat full volumes**: vesuvius z=15–49 or 16-channel z=24–39 outperformed full 65 slices; analogous to crop-size discipline in kuzushiji (96×96 patches)
- **Per-source normalization is essential for cross-document transfer**: per-fragment z-score in vesuvius (run 20260227 0.5424); analogous to per-page tile normalization
- **Heatmap/center-point regression aligns with point-based metrics**: CenterNet (F1=0.94) > Faster R-CNN (F1=0.88) on kuzushiji because metric scores center-in-bbox
- **TTA in probability space (4-way rot/flip)**: consistently +0.01–0.02 in vesuvius runs; D4 TTA standard across top solutions
- **Long-tail oversampling via RepeatFactorSampler + focal loss + label smoothing 0.1 + mixup α=0.8**: kuzushiji rare-class trio

## Common pitfalls
- **End-to-end multi-thousand-class detection**: kuzushiji F1=0.096 with 4,113-class Faster R-CNN; per-class heatmaps infeasible (~400MB/sample)
- **Pure 3D U-Net end-to-end**: vesuvius run 20260218 collapsed to F0.5=0.0; 3rd-place Vesuvius writeup explicitly rejects this
- **2D z-as-channels for volumetric inputs**: 0/5 Vesuvius top-5 use it; caps ~0.13 below domain-pretrained 3D encoders
- **Wider z-windows dilute signal**: vesuvius 24-ch=0.4902 vs 16-ch=0.5375
- **Submission encoding bugs hide huge gaps**: vesuvius run 1 internal 0.559 vs LB 0.101 (5.5× gap from RLE/label binarization)
- **Late-stage redesigns**: vesuvius run 1 step-16 redesign introduced bugs across 4 candidates
- **Zip-based on-the-fly loading bottlenecks workers**: pre-extract gives 4× speedup (kuzushiji)
- **Default threshold 0.5 on F0.5 metrics**: optima land at 0.7–0.93 (percentile-in-mask) or as low as 0.26+min-size

## Model selection guide
- Volumetric input (Z>1)? → Domain/Kinetics-pretrained 3D encoder (ir-CSN, 3D-ResNet) + depth-pool + 2D U-Net decoder
- Many-class character/symbol detection? → Class-agnostic detector (CenterNet preferred for point metrics) + EfficientNet-B0 classifier on 96×96 crops
- Single-fragment/page semantic segmentation? → EfficientNet-B3/B5 SMP-UNet with per-source z-score
- Avoid: end-to-end multi-class detection at scale, pure 3D-UNet, FCOS/RetinaNet/DETR with thousands of classes

## Feature engineering patterns
- Per-source (per-fragment / per-page) intensity normalization, NOT global
- Central z-window selection (16–32 of 65 for vesuvius); k-means anchor clustering for kuzushiji bbox sizes (6–520px)
- Tile + overlap + NMS for high-res pages; geometric-mean blending in overlaps (Vesuvius rank-1)
- Pre-extract archives to disk before training; profile data loading first
- Sub-split largest fragment to balance LOO folds (4/5 Vesuvius top-5 split fragment 2)
- ImageNet 3-ch → 16-ch first-conv channel adaptation is stable

## Tuning patterns
- Detector saturates fast (~5 epochs in kuzushiji); shift budget to classifier/decoder
- Sweep threshold on holdout; precision is typically the bottleneck for F0.5/F1
- Percentile-within-mask thresholds (0.87–0.93) adapt to per-source calibration drift better than fixed 0.5
- Cosine annealing + 500-step (or 1-epoch ×10 multiplier) warmup; AdamW wd 1e-6 to 1e-4
- Soft probability ensemble averaging (6–22 members), never majority voting; diversity from {backbone, z-config, tile size}, not seeds
- Validate submission encoding end-to-end early; large internal-vs-LB gaps signal pipeline bugs not model issues
