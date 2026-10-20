# Task Memory: aerial-cactus-identification

## Task profile
- Type: Binary image classification (cactus present vs. absent)
- Dataset: ~17,500 total images (14,175 train / 3,325 test), 32×32 RGB aerial imagery captured at ~100m altitude; 3:1 class imbalance (75.1% positive)
- Metric: AUC-ROC (higher is better)
- Medal thresholds: Unknown

## Best result
- Score: 0.9999 AUC | Medal: Ungraded | Run: 20260310_193628
- Approach: Candidate 1 (attempt_2) on a 3-candidate run. Exact architecture not fully logged, but consistent with compact CNN or hybrid ViT strategy. Two of three candidates scored >0.9996, confirming the task is near-solvable with well-tuned approaches. Prior run (20260207_213428) achieved 0.9986 with a documented compact ViT + conv stem architecture.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260207_013306 | — | — | EfficientNet-B0 & ViT-Tiny designed; training hung/failed (GPU contention) |
| 20260207_170747 | — | — | ViT w/ 4×4 patches coded & verified on toy batch; run halted before eval |
| 20260207_200808 | — | — | CNN + focal loss & ViT designed; never executed |
| 20260207_204101 | — | — | 2 candidates killed at init; 0 agent calls |
| 20260207_211447 | — | — | Same init failure pattern |
| 20260207_213428 | 0.9986 (val) | Ungraded | Compact ViT (conv stem + 6 blocks + multi-CLS) with focal loss α=0.75 |
| 20260227–20260310 (×12) | — | — | Repeated infrastructure/logging failures; 0 agent calls, empty metrics |
| 20260310_185537 | 0.6304 | Ungraded | Only 1/6 candidates finished; weak baseline, agent budget exceeded |
| 20260310_193628 | **0.9999** | Ungraded | 3 candidates, best two ≈0.9999 & 0.9997; candidate 2 crashed to 0.93 |

## Proven strategies
- **Compact ViT with convolutional stem** on native 32×32 input: 2-layer conv stem → 4×4 patch embedding → 6 transformer blocks (4-head attention) → multi-CLS token pooling. Achieved 0.9986 val AUC (run 213428). Avoids wasteful 32→224 upsampling.
- **Focal loss with α=0.75** for majority class: correctly handles 3:1 imbalance. Misconfiguring α (e.g., 0.25) significantly hurts performance.
- **Strong augmentation pipeline**: Mixup/CutMix, random erasing, color jitter, flips, rotations. Provides larger gains than architecture changes on this small-image aerial task.
- **Stochastic depth regularization**: Prevents overfitting on ~14K training samples, especially for ViT-based models lacking convolutional inductive bias.
- **AdamW + cosine annealing with warmup + gradient clipping**: Stable training within 5-minute budget.
- **Iteration/refinement**: Top candidates in run 193628 both used attempt_2, confirming that a debug/tuning pass improves scores.
- **Keeping model compact (~110K–156K params)**: Sufficient for this task; larger pretrained models add latency without benefit.

## Failure patterns
- **Infrastructure/logging failures dominate**: 12+ runs (Feb 27–Mar 10) produced zero metrics despite designing 4–6 candidates each. Likely caused by `max_agent_calls = -1` misconfiguration or manager initialization crash after design phase.
- **GPU contention in shared cluster**: First run (013306) stalled during training due to background processes (vscode servers). Always verify exclusive GPU access with `nvidia-smi`.
- **EfficientNet-B0 with 32→224 resize is wasteful**: Killed by judge in run 213428; computational overhead without benefit on tiny images.
- **ViT without conv stem struggles**: Pure ViT with 4×4 patches on 32×32 gives only 64 tokens—insufficient diversity. Conv stem is essential.
- **Agent budget overruns**: Runs 185537 (8/5 calls) and 193628 (8/6 calls) exceeded budgets. With 6 candidates and limited calls, most candidates never finish. Reduce to 3–4 candidates.
- **Single design choice can crater score**: In run 193628, candidate 2 scored 0.93 vs. 0.9999 for candidates 1 & 3—a 7pp gap from one bad architectural/augmentation decision.

## Feature engineering playbook
- **No resize to ImageNet scale**: Keep native 32×32; upsampling wastes compute and blurs features.
- **Dataset-specific normalization** preferred over ImageNet statistics for domain-specific aerial imagery.
- **Augmentation is king**: Flips, 90°/180°/270° rotations (aerial images have no canonical orientation), color jitter (varying sun angle/shadow), Mixup/CutMix, random erasing. These matter more than backbone choice.
- **Stratified train/val split**: Maintain 3:1 class ratio in both splits.

## Tuning insights
- **Focal loss α/γ sensitive**: α=0.75 (majority-weighted) with γ=2 works; α=0.25 is a common misconfiguration for this imbalance direction.
- **BCEWithLogitsLoss** is a safe fallback if focal loss hyperparameters aren't tuned.
- **15 epochs sufficient** for convergence with compact ViT on 14K samples under 5-min budget.
- **Batch size and epoch count should be pre-profiled**: Aim for <90 seconds training time to leave margin for inference and submission.
- **Closing the gap from 0.9986→1.0 is intractable via hyperparameter tuning alone**; TTA or ensembling needed for marginal gains.
- **3 candidates is the sweet spot** given agent call budgets of 5–6; 6 candidates leads to budget exhaustion with most unscored.

## Open questions
- What exact architecture/config distinguished 0.9999 (candidate 1) from 0.93 (candidate 2) in run 193628? Design details not logged.
- Would TTA (test-time augmentation) push 0.9999 to 1.0?
- Ensemble of compact ViT + lightweight CNN: does it add meaningful AUC above 0.9999?
- Root cause of 12+ consecutive infrastructure failures (Feb 27–Mar 10): was it `max_agent_calls = -1`, logging pipeline breakage, or manager crash?
- Focal loss (γ=2, α=0.75) vs. weighted BCE: no controlled ablation exists yet.
- Patch size ablation (4×4 vs 8×8) never completed; 8×8 gives only 16 tokens which may be too few.
