# Pattern: object-detection

## When this applies
- 2D/3D detection tasks with bounding boxes, mAP-based metrics
- Severe class imbalance (84% cars in Lyft; Aortic 6479 vs Pneumothorax 203 in VinBigData)
- Heavy/non-standard input formats (DICOM, multi-LiDAR point clouds) requiring preprocessing

## Proven patterns
- **Use mature, native detector frameworks over custom implementations**: torchvision Faster R-CNN R50-FPN v2 hit 0.429 on VinBigData (76% above bronze); CenterPoint anchor-free head won all 3 Lyft runs (best 0.28526 Gold)
- **Anchor-free / center-based heads** dominate when imbalance is severe: CenterPoint won every Lyft attempt (0.233–0.285)
- **Offline preprocessing to disk is mandatory** for slow I/O formats: DICOM→PNG cut epochs from 100+ min; LiDAR→BEV transforms needed pre-validation
- **Annotation fusion before training**: Weighted Boxes Fusion for multi-radiologist VinBigData boxes; multi-LiDAR sensor fusion (TOP+FRONT_L/R) for Lyft
- **Iterative fine-tuning from strong checkpoints**: 6 rounds drove Lyft winner from baseline to 0.28526; single-shot leaves headroom
- **Test-time augmentation** provided measurable lift (VinBigData winning submission)

## Common pitfalls
- **Custom backbone integrations consistently fail**: EfficientNet-B4 4→5 channel mismatch with FPN; Swin-T required modifying torchvision internals; custom PointPillars repeatedly scored 0.0 mAP
- **Coordinate frame transforms are the #1 recurring bug** in 3D (sensor→ego→world)
- **Validation→test gap is unreliable**: Lyft run 1 improved 2.2× val→test, runs 2–3 regressed up to −56%; don't pick submissions on val mAP alone
- **Aggressive mid-run rewrites** (augmentation overhauls, architecture pivots) degraded performance vs. fine-tuning the leader
- **Lightweight statistical baselines** (range-view 2D, pillar-only) score ~0 and waste budget
- **Class imbalance often left unaddressed** despite being known (Lyft 84% cars: focal loss / class-balanced sampling untried)
- **Environment friction** (spconv, mmdet3d, numba, concurrency races) burned 7 consecutive attempts in VinBigData

## Model selection guide
- Sparse 2D medical/natural detection → **Faster R-CNN R50-FPN v2** baseline (3–5 min/epoch, two-stage handles high background ratio)
- 3D LiDAR detection → **CenterPoint + PointPillars backbone on BEV** (never custom pillar encoders)
- Multi-modal 3D → stack as multi-channel BEV (13-ch LiDAR+camera lifted +0.018 val mAP)
- Only escalate to RetinaNet/FCOS/Swin/EfficientNet **after** the native baseline is fully tuned

## Feature engineering patterns
- Convert exotic formats to standard images/tensors on disk as step 1 (DICOM→PNG; LiDAR→BEV)
- Validate coordinate transforms on toy subsets before full training
- Fuse redundant annotators/sensors via WBF or multi-sensor stacking; preserve rare classes during fusion
- Stack modalities as extra channels rather than designing late-fusion architectures

## Tuning patterns
- Budget 3–6 fine-tuning rounds on the leading candidate (highest ROI lever in Lyft)
- Drop candidates scoring ~0 mAP early to free compute
- Prefer architecturally-principled candidate over val-mAP outliers when val/test gap is unstable
- TTA at inference is cheap and consistently helps
- Resist designing v2/v3 revisions that never reach training—force the strongest checkpoint into submission
