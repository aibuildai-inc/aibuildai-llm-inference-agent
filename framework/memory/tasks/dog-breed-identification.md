# Task Memory: dog-breed-identification

## Task profile
- Type: image classification (120 dog breeds)
- Dataset: Kaggle "dog-breed-identification" competition; image modality
- Metric: unknown (likely multi-class log loss, per Kaggle standard for this competition)
- Medal thresholds: unknown

## Best result
- Score: N/A | Medal: ungraded | Run: none successful
- Approach: No candidate has been trained. Both runs (20260208_065702, 20260208_065806) failed before any model was designed or evaluated.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260208_065702 | N/A | ungraded | Complete pipeline stall — 0 candidates, 0 agent calls |
| 20260208_065806 | N/A | ungraded | Kaggle API 401 auth failure during setup; no data downloaded |

## Proven strategies
- (none yet — no successful training run to date)

## Failure patterns
- **Pipeline stall (run 065702):** Zero candidates generated, zero agent calls dispatched despite `max_agent_calls=8`. Likely silent failure in initialization/dispatch logic.
- **Kaggle auth failure (run 065806):** 401 Unauthorized from Kaggle API even with `~/.kaggle/kaggle.json` present. API key expired or revoked.
- **W&B temp-dir permissions:** Permission denied errors on W&B temp dirs observed in run 065806; may cause downstream issues even after auth is fixed.
- Both runs burned wall-clock time with zero useful signal — pre-flight checks are essential.

## Feature engineering playbook
- (nothing tested yet)

## Tuning insights
- (nothing tested yet)

## Open questions
- Why did the agent dispatch loop produce 0 calls in run 065702? Check initialization logic and candidate generation trigger.
- Regenerate Kaggle API key and verify with `kaggle competitions list` before next run.
- Add a fast-fail credential + data-access validation step at pipeline start (before agent loop).
- Consider pre-downloading / caching the competition dataset to avoid runtime credential dependency.
- Clean up or reconfigure W&B temp-dir permissions for W&B logging.
- Once data access is resolved, explore standard image-classification baselines: fine-tuned EfficientNet / ConvNeXt / ViT families with standard augmentation (RandomResizedCrop, horizontal flip, color jitter).
