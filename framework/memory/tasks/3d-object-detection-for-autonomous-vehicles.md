# Task Memory: 3d-object-detection-for-autonomous-vehicles

## Task profile
- Type: 3D object detection (LiDAR + optional camera fusion)
- Dataset: Lyft/nuScenes-format; 45K–106K points/scan (up to ~66K typical), 5 channels (x,y,z,intensity,ring); multi-lidar (1–3 lidars, ~20% samples have all three: TOP + FRONT_LEFT/RIGHT); 9 classes; 502K annotations with heavy imbalance (car=84%)
- Metric: mean Average Precision (mAP), higher is better
- Medal thresholds: Gold achievable at ~0.23–0.29 test mAP; bronze baseline ~0.046

## Best result
- Score: 0.28526 | Medal: Gold | Run: 20260221_163156
- Approach: BEV CenterPoint with PointPillars-style backbone. LiDAR points encoded to bird's-eye-view via sensor→ego→world coordinate transforms, multi-lidar fusion where available, CenterPoint head regressing 3D boxes for 9 classes. 6+ iterative training attempts on the winning candidate.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260221_163156 | 0.28526 | Gold | BEV CenterPoint + PointPillars backbone, multi-lidar fusion, 6 training iterations |
| 20260227_175108 | 0.23316 | Gold | CenterPoint anchor-free, custom PyTorch (no spconv/mmdet3d), 3 tuning rounds |
| 20260228_024459 | 0.23940 | Gold | 13-ch BEV (LiDAR+camera) + EfficientNet-B0 + BiFPN + CenterPoint head |

## Proven strategies
- **CenterPoint-style anchor-free head on BEV** consistently wins across all 3 runs; beats anchor-based, range-view 2D, and lightweight baselines
- **BEV representation** outperforms range-view 2D and 1D pillar-only baselines
- **Multi-lidar fusion** (TOP + FRONT_LEFT/RIGHT) improves coverage where available
- **Iterative fine-tuning from strong checkpoints** (3–6+ attempts) reliably squeezes gains; single-shot training leaves headroom
- **Camera+LiDAR fusion via multi-channel BEV** (13-ch in run 3) gave ~0.018 val mAP lift over 3-ch ResNet-18 baseline in the same run
- **EfficientNet-B0 + BiFPN** scales better than ResNet-18 for dense 3D BEV tasks
- **Early dropout of non-competitive candidates** (mAP≈0) frees compute for winners

## Failure patterns
- **Coordinate frame transforms** (sensor→ego→world) are the #1 recurring bug; require pre-validation on small subsets
- **Custom PointPillars implementations** repeatedly score 0.0 mAP (pillar encoding bugs); avoid without mature libs
- **Lightweight/statistical BEV baselines** score ~0 and waste budget
- **Validation→test gap is massive and inconsistent**: run 1 improved 0.128→0.285 (2.2×); runs 2–3 regressed 0.143→0.233 and 0.548→0.239 (−56%). Val scores are NOT reliable selection signal
- **Range-view 2D** (run 1 C6) scored high on val (0.5395) but was correctly rejected—classic overfitting
- **Design-to-code handoff failure**: v1/v2 revision plans (GT sampling, aggressive aug, LR schedules) never reached training in run 3
- **Environment setup friction**: missing spconv/mmdet3d, numba install issues, structured-output errors during submission
- **Aggressive augmentation / mid-run architecture pivots** degraded perf; fine-tuning beat rewrites

## Feature engineering playbook
- Encode LiDAR to BEV with proper sensor→ego→world transforms (validate on toy examples first)
- Stack modalities as extra BEV channels: LiDAR depth/intensity/height + camera RGB → 13-ch input
- Fuse multiple lidars when available rather than using TOP only
- Keep the baseline augmentation pipeline; resist aggressive GT sampling rewrites mid-run

## Tuning insights
- 3–6 iterative tuning rounds on the winning candidate is the highest-ROI lever
- Fine-tune from strongest checkpoint rather than retraining from scratch on new hyperparams
- Do not use validation mAP alone to pick final submission—val/test gap is unreliable in both directions; prefer the architecturally-principled candidate (BEV + CenterPoint) over a val outlier
- Class imbalance (84% cars) is NEVER addressed in any run; focal loss / class-balanced sampling / per-class heads are untried

## Open questions
- Root cause of large and inconsistent val→test gap: domain shift on private set, metric recalibration, or test-time post-processing?
- Would focal loss or class-balanced sampling lift minority classes (non-car = 16%)?
- Would deeper EfficientNet (B2/B3), heavier BiFPN, or test-time augmentation / ensembling push past 0.285 test mAP?
- Why do custom PointPillars implementations keep scoring 0.0—encoding bug, augmentation pipeline gap, or loss scaling?
- Can v1/v2 design revisions (GT sampling + LR schedule) be forced through the coder to actually train, not just be designed?
