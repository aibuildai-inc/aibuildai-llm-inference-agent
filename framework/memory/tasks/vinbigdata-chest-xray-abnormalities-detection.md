# Task Memory: vinbigdata-chest-xray-abnormalities-detection

## Task profile
- Type: Object detection (medical imaging, multi-class abnormality localization)
- Dataset: ~13.5k DICOM chest X-rays; multi-radiologist annotations (5915 single / 3218 double / 4757 triple-agreed); severe class imbalance (Aortic enlargement 6479 vs Pneumothorax 203)
- Metric: mAP@IoU>0.4, higher is better
- Medal thresholds: bronze ~0.243 (silver/gold unknown)

## Best result
- Score: 0.429 | Medal: ungraded (above bronze by 76%) | Run: 20260218_210551
- Approach: Native torchvision **Faster R-CNN ResNet50-FPN v2** trained on DICOM→PNG-converted images. Multi-radiologist boxes merged via **Weighted Boxes Fusion (WBF)** before training. TTA at inference.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260218_210551 | 0.429 | ungraded (>bronze) | Faster R-CNN R50-FPN v2 + WBF + DICOM→PNG + TTA |

## Proven strategies
- **Native torchvision detectors** (FasterRCNN, RetinaNet, FCOS) — zero integration risk vs custom timm/Swin backbones
- **Weighted Boxes Fusion** for merging variable multi-annotator boxes — non-optional for this dataset
- **Two-stage Faster R-CNN** beat one-stage RetinaNet (focal loss) and anchor-free FCOS despite class imbalance favoring focal loss; sparse medical detection has high background ratio that two-stage handles well
- **DICOM→PNG offline conversion** to disk before training (DICOM read is ~445ms/image → 100+ min/epoch otherwise)

## Failure patterns
- **Custom backbones burn budget**: EfficientNet-B4 had 4→5 channel mismatch with FPN; Swin-T + Focal Loss required modifying torchvision FasterRCNN internals — both killed at design phase
- **Coder infrastructure concurrency error** crashed 7 consecutive implementation attempts in the same run; root cause undiagnosed (transient cluster vs. code-side race)
- On-the-fly DICOM reading is infeasible at this dataset scale

## Feature engineering playbook
- Convert all DICOMs to PNG on disk as first pipeline step (mandatory)
- Apply WBF to collapse the 3 radiologists' annotations per image into consensus boxes before feeding the detector
- Preserve class-imbalanced tail classes (Pneumothorax etc.) — do not drop rare classes during WBF

## Tuning insights
- Faster R-CNN R50-FPN v2 at ~3–5 min/epoch gives fastest iteration loop; prefer it as baseline before exploring one-stage/anchor-free alternatives
- TTA at inference provided measurable lift (applied in winning submission)
- Architectural diversity (RetinaNet/FCOS) was explored but none beat the Faster R-CNN baseline in this run

## Open questions
- Would fully-trained RetinaNet or FCOS close the gap to Faster R-CNN, or is the 2-stage advantage structural for sparse medical detection? (N=1 completion, margin uncertain)
- Root cause of persistent coder concurrency failures — need to distinguish transient infra from reproducible code-side bug before trusting ambitious candidates
- Silver/gold thresholds unknown — unclear how much headroom 0.429 leaves; next runs should target stronger backbones (R101-FPN, Swin via mmdetection) once infra is reliable
- Multi-scale training / larger input resolution impact not yet measured
