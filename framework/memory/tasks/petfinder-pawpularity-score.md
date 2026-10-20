# Task Memory: petfinder-pawpularity-score

## Task profile
- Type: Bounded image regression (predicting pet photo "pawpularity" score 1–100)
- Dataset: ~8,920 images + 12 binary metadata features; right-skewed target (mean 38, median 33, std 20.6, secondary spike at 100)
- Metric: RMSE, lower is better
- Medal thresholds: Bronze ≈ 17.10, Silver < 17.10, Gold ≈ 16.9 (empirically confirmed)

## Best result
- Score: 16.877 | Medal: Gold | Run: 20260302_110058
- Approach: Cross-family ensemble averaging ConvNeXt-Large-CLIP (10-fold, differential LR, 128-dim SVR bottleneck on pooled embeddings; solo 17.008) with BEiT-Large-384 upscaled from 224px with extended training (solo 17.086). BCE loss on [0,1]-normalized targets, stratified folds, gold unlocked by pretrain-paradigm diversity (CLIP + MIM).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260212_071932 | 17.43 | None | Swin-B 384 + BCE, 5-fold, EMA |
| 20260218_213415 | 17.42 | None | Swin V2 Base + BCE, Mixup/CutMix added |
| 20260219_060351 | 17.35 | None | ConvNeXt-B + SmoothL1+MSE blend, EMA |
| 20260223_225553 | 17.12 | None | Swin-Large 384 + BCE, 5-fold ensemble |
| 20260224_042133 | 17.21 | None | Swin-B + ConvNeXt-B ensemble, gated metadata fusion |
| 20260224_210426 | 16.97 | Silver | Swin-Large 384 + BCE, full tuning |
| 20260225_184436 | 17.07 | Bronze | ConvNeXt-Large 384 + MSE + TTA |
| 20260226_181358 | 17.48 | None | BEiT-Large + BEiT-Base 3-fold 224 ensemble |
| 20260227_211852 | 17.32 | None | Swin-L 384 + ViT-B (MLP+SVR) ensemble |
| 20260301_222700 | 16.97 | Silver | ConvNeXt-Large 384 + BCE |
| 20260301_222804 | 17.43 | None | Swin-L + EfficientNetV2-M cross-candidate ensemble |
| 20260302_110058 | 16.877 | **Gold** | ConvNeXt-L-CLIP+SVR (10-fold) + BEiT-L-384 ensemble |
| 20260302_125245 | 17.081 | Bronze | Swin-L 224 + ConvNeXt-L 384 ensemble, BCE post-revision |

## Proven strategies
- **Cross-pretrain-paradigm ensemble is the gold lever.** Averaging CLIP-pretrained ConvNeXt-L + MIM-pretrained BEiT-L-384 broke 16.9 barrier; pretrain diversity > backbone family diversity.
- **BCEWithLogitsLoss on [0,1]-normalized targets dominates** (16.88–16.97 best); reconfirmed: candidate switching MSE→BCE gained 0.91 RMSE (18.48→17.57). MSE viable only with TTA+large model; SmoothL1/Huber fails.
- **ImageNet-22k pretraining > ImageNet-1k** for Swin/ConvNeXt Large backbones; CLIP pretraining on ConvNeXt adds further edge but ViT-Large-CLIP (304M) times out in 1200s/fold budgets.
- **Two-stage NN→SVR (Ridge) on pooled embeddings with 128-dim bottleneck** boosts small-data regime (ConvNeXt-CLIP solo 17.008). Bottleneck = implicit regularization.
- **ConvNeXt-Large-CLIP + ConvNeXt-Large + Swin-Large + BEiT-Large-384** all reach 17.0–17.1 individually at 384px.
- **Resolution 384px mandatory** for Large models; 224px caps ~17.5 (Swin-L 224 still usable in ensembles, but underperforms 384).
- **10-fold StratifiedKFold beats 5-fold by ~0.2 RMSE** for 200M+ param backbones; 5-fold acceptable floor when time-bound.
- **Differential LR** (backbone 1e-5 to 3e-5, head 1e-4 to 1e-3) + 1–2 epoch warmup; AdamW + CosineAnnealingWarmRestarts.
- **EMA (~0.9975)** ~0.1–0.2 RMSE; **TTA (HFlip+rotation)** 0.2–0.3 RMSE gain.
- **Metadata: concat+linear embed OR gated fusion (FiLM)**; metadata near-noise (|r|<0.025), don't over-invest.
- **Ensemble CNN+Transformer** (ConvNeXt+Swin) outweighs single-family variants; architecture-class diversity > resolution diversity at bronze tier.

## Failure patterns
- **ViT-Large-CLIP (304M) at 224px, 5-fold times out** on 1200s/fold judge limits; CLIP pretrain appeal blocked by compute.
- **BEiT-Large/DeiT3-Large at 384px infeasible** in 150-min budgets; pre-training judged as overfit risk on 8.9k samples.
- **SwinV2-Large + Huber + aggressive Mixup diverges** (RMSE 28.18).
- **Mixup/CutMix hurts Swin-L/ViT on noisy crowd-sourced labels**; prefer flip + color jitter + coarse dropout.
- **BEiT-Large @ 224px caps at 17.48**; MIM does NOT compensate for resolution deficit.
- **EfficientNetV2-M/S judge-killed** at review (<100M params insufficient; also redundant with EffNet-B4).
- **EfficientNet-B4 (19M) caps at ~17.2 RMSE**; param sweet spot is 196–200M.
- **EVA02-Large-448 infeasible** (362 GMACs, OOM).
- **SmoothL1/Huber on raw [0,100]** fails hard (EffNetV2-S: 28.86); match loss to metric.
- **Auxiliary multi-task heads plateau at 19.87**; metadata too sparse to help.
- **LightGBM/second-stage stacking on OOF predictions degrades** (17.94→18.11); SVR works only on embeddings, not prediction stacking.
- **3+ model automated ensembles fail** via GPU contention/fold misalignment; stick to 2-model aggregation.
- **Reviser/coder wall-clock timeouts** recurring; need time-aware training loops.

## Feature engineering playbook
- **Image resolution is the dominant "feature."** 384px > 224px by 0.1–0.5 RMSE; 448+ untested feasibly.
- **Per-fold target z-score normalization** for MSE; [0,1] sigmoid normalization for BCE.
- **Pooled embeddings → 128-dim bottleneck → Ridge/SVR** reproducible +0.1–0.2 RMSE trick on small data.
- **Metadata (12 binary):** gated fusion or linear-embed concat; cross-attention wasteful.

## Tuning insights
- **Loss = metric principle:** BCE-on-[0,1] optimal (reconfirmed: 0.91 RMSE jump on single-model switch); MSE-on-normalized viable runner-up.
- **Folds:** 10-fold for Large 384px when budget allows; 5-fold stratified-on-binned-targets floor; never 3-fold.
- **Param sweet spot:** 196–200M (Swin-L, ConvNeXt-L). <50M underfits; >300M times out or overfits.
- **Epochs:** 6/fold optimal; 8–10 when upgrading resolution mid-training.
- **Augmentation:** start without mixup/cutmix; add only if val confirms.
- **Ensemble sweet spot:** 2 models from different **pretrain paradigms** (CLIP+MIM > CLIP+sup > Swin+ConvNeXt at 384); simple average sufficient.
- **Compute budget:** 10-fold ConvNeXt-L 384 = 60–120 min A100; plan sequential folds; keep single model under 1200s/fold judge limit.
- **SVR bottleneck dim:** 128 works; 1536→128 compression is the active ingredient hypothesis.

## Open questions
- **SVR on raw 1536-dim vs 128-dim bottleneck** — is bottleneck the active ingredient or just compute savings?
- **Why BCE dominates**: sigmoid saturation, implicit regularization, or label rescaling? Ablate normalized targets vs raw.
- **448–512px with gradient checkpointing / LoRA** — could unlock 0.1–0.2 RMSE; EVA02 retry with chunked attention.
- **Triple-paradigm ensemble** (CLIP + MIM + supervised ImageNet-22k) — push below 16.8?
- **OOF-weighted blend vs simple average** for differing-loss pairs (BCE + MSE).
- **Ordinal regression head (K=20 binned BCE)** judge-passed but unquantified.
- **Lightweight regime (<50M params, MobileViT/DistilViT)** — quantify param-vs-accuracy curve below EfficientNet-B4.
- **Knowledge distillation Large→Small** preserving ~17.5 for fast inference.
