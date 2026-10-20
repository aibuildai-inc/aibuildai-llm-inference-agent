# Task Memory: siim-covid19-detection

## Task profile
- Type: Multi-task object detection + classification (image-level opacity bounding boxes + study-level 4-class classification)
- Dataset: ~5,696 DICOM chest X-ray images (2330×2783px typical), ~1,244 test samples. Study-level labels: Negative (27%), Typical (47%), Indeterminate (18%), Atypical (8%). Image-level: 68% have opacity boxes. Large opacities (mean 706×1026px, 21% width / 37% height of image).
- Metric: mAP @ IoU > 0.5 (higher is better)
- Medal thresholds: Bronze ≥ 0.601

## Best result
- Score: 0.46356 | Medal: none | Run: 20260320_201842
- Approach: Faster R-CNN ResNet101-FPN with detection-derived study classification (leveraging strong opacity-label correlation). Reached 0.7697 mAP during training but degraded to 0.46356 on submission—likely inference/formatting bug. Best validated internal score across all runs.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_193205 | N/A | ungraded | Setup timeout; no candidates ran |
| 20260213_223339 | ungraded | ungraded | 2 candidates designed (ResNet50 FRCNN + FRCNN w/ aux head); GPU OOM killed tuner at 0.507 mAP |
| 20260217_222418 | 0.387 | none | FCOS anchor-free + multi-task head reached 0.549 internal mAP; reviser crashed |
| 20260305_193950 | 0.331 | none | Faster RCNN ResNet50-FPN, 10 epochs @ 512px; plateaued |
| 20260305_194005 | N/A | ungraded | 6 candidates designed, all killed at step 2 |
| 20260306_072458 | 0.260 | none | Only working candidate scored 0.4708 internal; final submission degraded |
| 20260306_072511 | 0.0 | none | All-none baseline; DL candidates killed by SPEED-FIRST policy |
| 20260306_074404 | 0.288 | none | CSV-only baselines; per-class stratified > majority class by +0.05 |
| 20260320_201842 | 0.464 | none | Faster RCNN ResNet101-FPN; 0.7697 internal mAP, submission degraded |

## Proven strategies
- **Two-stage detectors (Faster R-CNN) consistently outperform single-stage** (RetinaNet, YOLO) on large, variable-scale opacity objects; RoI Align precision matters. RetinaNet plateaued at 0.058 mAP in one run.
- **FCOS anchor-free detectors are viable** and avoid anchor mismatch issues; reached 0.549 internal mAP (best single-architecture internal score before ResNet101 run).
- **Detection-derived study classification** exploits strong opacity-label correlation (Negative=0% opacity, Typical/Indeterminate~95%) and saves training time vs. two-model approaches.
- **GroupKFold by StudyInstanceUID** is non-negotiable; prevents ~0.15–0.20 mAP inflation from study-level leakage.
- **DICOM preprocessing pipeline**: VOI LUT → MONOCHROME1/2 inversion → CLAHE → normalization → 3-channel stacking → PNG cache. Caching is critical; per-epoch DICOM decoding kills throughput (~0.5–1s per image).
- **ResNet101 > ResNet50** backbone for medical imaging capacity on large X-ray images.
- **Frozen ImageNet features are insufficient**; backbone fine-tuning is mandatory for medical imaging domain adaptation.
- **Empirical class-frequency baseline** (per-class stratified confidence) outperforms majority-class baseline by ~0.05 mAP; useful as a floor.

## Failure patterns
- **Submission degradation**: Internal mAP consistently much higher than leaderboard score (0.7697→0.464, 0.549→0.387, 0.4708→0.260). Inference pipeline, bbox coordinate rescaling, or submission formatting bugs are the likely cause. This is the single biggest recurring issue.
- **GPU OOM / timeouts**: Multiple runs killed by memory exhaustion or 600s timeouts during training/inference. Batch size, image resolution, and model size must be carefully balanced.
- **Reviser/judge crashes**: `judge_output.json does not exist`, `train.py not found`, silent training failures—infrastructure fragility consistently blocks pipeline completion.
- **Agent call exhaustion**: Runs consistently exhaust agent calls before meaningful tuning occurs; too many calls spent on design/setup, too few on refinement.
- **RetinaNet consistently fails**: Diverged in one run (high LR), plateaued at 0.058 in another. Single-stage detectors struggle with this task's large, variable-scale objects.
- **Anchor scale mismatch**: Default COCO anchors (5–350px) miss large opacities (30–70% image width); manual recalibration never fully implemented.

## Feature engineering playbook
- **DICOM windowing**: Apply VOI/Modality LUT, handle MONOCHROME1 inversion, variable bit depths, JPEG Lossless compression.
- **CLAHE enhancement**: Applied after normalization, before PNG caching; improves contrast for opacity detection.
- **3-channel stacking**: Grayscale→3-channel for ImageNet-pretrained backbone compatibility. Works but wastes capacity; consider single-channel input with modified first conv layer.
- **Image resolution**: 512px is the minimum viable; 768–1024px recommended by competition winners. 256px is too coarse. Multi-resolution training (512→640 fine-tune) could balance speed and quality.
- **Bounding box rescaling**: Must correctly map from resized training coordinates back to original pixel coordinates for submission. Errors here directly cause submission degradation.

## Tuning insights
- **Learning rate sensitivity**: FCOS and RetinaNet diverge with aggressive LR schedules; use conservative warmup + cosine decay. AdamW preferred.
- **Class weighting is critical**: Study-level imbalance (47% Typical vs. 8% Atypical) + image-level (68% with opacities) demands focal loss + explicit class weighting. Cross-entropy alone is insufficient.
- **Multi-task loss balancing**: Detection vs. classification loss ratio needs careful tuning; auxiliary classification head risks gradient dominance.
- **EMA + gradient accumulation**: Effective batch=8 with accumulation; EMA stabilizes training on small dataset.
- **Training budget**: 10 epochs is insufficient for convergence (trajectory still rising at epoch 10). 20+ epochs with proper scheduling needed.
- **Fewer, longer-trained candidates beat many lightly-trained ones**: I/O bottleneck from DICOM loading means more candidates = less training per candidate. Allocate budget to 1–2 candidates with full tuning.

## Open questions
- **Why does submission score consistently degrade vs. internal validation?** This is the #1 blocker. Need explicit submission validation step: verify bbox coordinate space, CSV format, study/image ID mapping, confidence calibration.
- **Would 768px+ resolution with ResNet101 FRCNN close the 0.14 gap to bronze (0.464→0.601)?** Internal scores suggest the architecture can reach 0.77 mAP; the gap is in inference/submission, not modeling.
- **External data (ChexPERT, NIH Chest-14, VinBigData, RSNA Pneumonia)**: Competition winners relied heavily on pretraining with external medical data + pseudo-labeling. Pure in-domain training may have a ceiling below 0.60.
- **TTA and soft-NMS**: Neither has been properly evaluated; horizontal flip + multi-scale inference could lift mAP without retraining.
- **DICOM I/O optimization**: Multi-threaded loading (ThreadPoolExecutor) or pre-cached resized PNG dataset could free significant training budget.
- **Is the anchor-free (FCOS) or anchor-based (FRCNN) path more promising?** FCOS avoids anchor mismatch but dense predictions need NMS tuning; FRCNN has higher ceiling but higher memory cost. Best internal scores favor FRCNN with ResNet101.
