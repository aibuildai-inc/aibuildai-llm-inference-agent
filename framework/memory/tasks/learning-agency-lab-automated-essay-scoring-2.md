# Task Memory: learning-agency-lab-automated-essay-scoring-2

## Task profile
- Type: Ordinal regression (essay scores 1–6, treated as regression + threshold optimization)
- Dataset: ~15.5K student essays (Persuade 2.0 corpus); heavy class imbalance — score 3 dominates (36%), score 6 rare (0.87%, ~135 samples); essay lengths range widely (P99 ≈ 972–1024 tokens)
- Metric: Quadratic Weighted Kappa (QWK), higher is better
- Medal thresholds: Bronze ≥ 0.83471; Gold achieved at 0.8389–0.8425

## Best result
- Score: 0.8425 QWK | Medal: Gold | Run: 20260226_061203
- Approach: DeBERTa-v3-large + Huber loss + LLRD + head-tail truncation + SWA + Nelder-Mead threshold optimization. 5-fold stratified CV. Simple regression pipeline with robust loss and careful regularization outperformed complex multi-head designs.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_193431 | ungraded | — | Incomplete; DeBERTa-large MSE baseline, OOF QWK=0.7304 at seq_len=512 |
| 20260213_213009 | 0.83447 | none | SmoothL1 loss, multi-sample dropout, mean pooling; timed out before convergence |
| 20260215_110055 | 0.7919 | none | CORAL ordinal best val (0.8166) but submitted classification head regressed |
| 20260217_114422 | 0.80764 | none | Selective fp32 attention patch; stacking (DeBERTa+LightGBM) hit 0.800 |
| 20260218_112721 | ungraded | — | Pipeline stalled after coder; DeBERTa-large timed out in sanity check |
| 20260218_223313 | 0.83855 | **Gold** | RoBERTa-large (not DeBERTa); extended fold training budgets (7200s); MSE+clip+round |
| 20260220_014558 | 0.8264 | none | Ridge stacking ensemble (DeBERTa-base + LightGBM + features); missed bronze by 0.008 |
| 20260221_234321 | 0.79287 | none | DeBERTa-base completed; large models timed out; blend degraded score |
| 20260222_224441 | 0.82587 | none | Dual-head (MSE+CORAL) + AWP + weighted layer pooling; float32 critical |
| 20260226_061203 | 0.8425 | **Gold** | Huber loss + LLRD + head-tail truncation + SWA; DeBERTa-v3-large |
| 20260227_054052 | 0.82265 | none | Simplest recipe (mean pool + MSE); 3-fold stable but ceiling at 0.80 |
| 20260227_184737 | 0.80719 | none | SmoothL1 + LLRD + layer reinit; base ensemble underperformed single large |
| 20260228_133518 | 0.81168 | none | FP16+grad checkpointing; 5-fold timeout limited convergence |
| 20260301_030240 | 0.8389 | **Gold** | Attention pooling + AWP + LLRD + EMA + multi-sample dropout; bf16 speedup |
| 20260302_031425 | 0.84111 | **Gold** | DeBERTa-large attention pooling + CORAL blend; base→large revision +0.016 |

## Proven strategies
- **DeBERTa-v3-large is table-stakes**: Base models ceiling at 0.80–0.82 QWK; large (304–434M params) consistently reaches 0.82–0.84+. Every Gold run used a large model.
- **Regression + threshold optimization > ordinal classification**: MSE/Huber/SmoothL1 regression with Nelder-Mead threshold optimization on validation QWK outperforms CORAL/CORN ordinal heads. Threshold tuning recovers ~0.01–0.03 QWK.
- **Huber/SmoothL1 loss > MSE**: More robust to label noise from subjective grading; consistent small gains across runs.
- **Attention pooling > mean pooling > CLS**: Learned attention weights over token embeddings capture essay-quality signals; confirmed in Gold runs (20260301, 20260302).
- **Regularization stack (AWP + LLRD + EMA + multi-sample dropout)**: Each contributes ~0.002–0.005 QWK; cumulative ~+0.01. AWP has ~50% per-step overhead — only viable when time budget allows.
- **LLRD is mandatory**: Backbone LR 2e-5, head LR 1e-4; prevents backbone collapse and head divergence on 24-layer models.
- **Head-tail truncation**: Captures essay intro + conclusion; superior to simple left-truncation for scoring tasks.
- **3-fold CV for iteration, 5-fold for final**: 5-fold on DeBERTa-large is ~30h; 3-fold allows 2–3 tuning iterations within budget.
- **bf16/fp16 mixed precision with careful dtype handling**: 2× speedup vs fp32; but DeBERTa-v3 attention layers need explicit fp32 casting or selective patches to avoid NaN.
- **Ensemble of regression + ordinal paradigms**: Blending MSE regression (C1) + CORAL ordinal (C3) predictions achieved best single-run score (0.84111).

## Failure patterns
- **Coder timeout (1200s)**: Killed candidates in 10+ runs. DeBERTa-large at seq_len≥768 with fp32 regularly exceeds sanity-check budget. Mitigation: fp16/bf16, gradient checkpointing, minimal sanity batch.
- **DeBERTa-v3 + bf16/fp16 = NaN**: Recurrent across runs; caused by attention layer numerical instability. Fix: selective fp32 on DisentangledSelfAttention or explicit backbone.float() before training.
- **Ordinal classification (CORAL/CORN) fragile in practice**: CORN produced QWK=0.0 (collapsed predictions) in multiple runs. CORAL more stable but rarely beat regression+thresholds as standalone. Best used as ensemble diversity source.
- **Complex architectures collapse**: Dual-head, sliding window, R-Drop, pseudo-labeling — all added implementation risk and training instability. Simple pipelines consistently outperformed.
- **MSE regression without threshold optimization**: Raw continuous predictions lose ~0.02–0.03 QWK vs optimized thresholds. Always apply Nelder-Mead post-training.
- **DeBERTa-v3-large in full fp32**: Prohibitively slow (~10–15s/step); forces downgrade to base or incomplete training. Mixed precision is mandatory.
- **5-fold CV instability**: Multiple runs saw 2/5 folds diverge (QWK dropping to 0.55–0.62). Fewer folds = more samples/fold = more stable gradients.
- **Ensemble of weak models < single strong model**: DeBERTa-base ensemble (QWK ~0.79) consistently lost to single DeBERTa-large (QWK ~0.81–0.83).

## Feature engineering playbook
- **Handcrafted features (readability, discourse markers, vocabulary richness) are secondary**: LightGBM on frozen DeBERTa embeddings + 26 features achieved ~0.77–0.80 QWK; useful for ensemble diversity but not competitive standalone.
- **Text length correlates strongly with score** (r ≈ 0.6–0.7) but is not a shortcut; transformer representations subsume length signal.
- **Frozen DeBERTa embeddings + LightGBM**: Viable fast-train diversity source for stacking (~0.77–0.80 QWK); complements fine-tuned transformer predictions.
- **Head-tail truncation > left-only truncation**: Captures essay structure (intro + conclusion) critical for holistic scoring.

## Tuning insights
- **Sequence length**: 512 truncates 31% of essays → significant QWK loss. 768 covers ~90–95%. 1024 covers ~98.5%. Best trade-off is 768–1024 depending on compute budget. Diminishing returns beyond 1024 (<1% essays affected).
- **Learning rate**: Backbone 1e-5 to 2e-5; head 1e-4. Higher LR (5e-5+) causes instability; lower (5e-6) undertrained within budget.
- **Batch size**: 2–4 per GPU with gradient accumulation 2–8. Larger effective batch (8–16) more stable but slower per step.
- **Epochs**: 3–5 epochs sufficient for DeBERTa-large on 15.5K samples. Overfitting visible by epoch 2–3 without regularization.
- **SWA (Stochastic Weight Averaging)**: Stabilizes late-stage training; synergistic with LLRD.
- **EMA decay 0.999**: Modest but consistent improvement; low overhead.
- **Multi-sample dropout**: 5 masks at p=0.1 during training, averaged at inference. Lightweight pseudo-ensemble.
- **Stratified K-fold by score label**: Essential given 41× class imbalance ratio.

## Open questions
- **Pseudo-labeling / self-training**: Mentioned in top Kaggle solutions but never successfully implemented in any run due to time budget. Could bridge remaining gap to higher QWK.
- **Per-prompt fine-tuning**: Essays have different prompts; prompt-aware modeling untested.
- **Ensemble weight optimization**: Simple averaging or ridge stacking used; learned weighted blending (Nelder-Mead on ensemble weights) could improve multi-model runs.
- **Focal loss / cost-sensitive sampling for score 6**: Never tried; threshold optimization handles imbalance post-hoc but explicit rare-class handling during training is unexplored.
- **Two-stage training (Persuade 2.0 → Kaggle-only)**: Distribution shift between data sources acknowledged but not systematically exploited.
- **Early stopping on validation QWK (not loss)**: Current runs trained fixed epochs; QWK-aware stopping could prevent overfitting.
- **RoBERTa-large as alternative backbone**: One Gold run (0.83855) used RoBERTa-large instead of DeBERTa; faster training, no NaN issues. Under-explored as a reliable fallback.
