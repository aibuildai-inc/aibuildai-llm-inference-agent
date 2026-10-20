# Task Memory: siim-isic-melanoma-classification

## Task profile
- Type: Binary classification (melanoma vs. benign)
- Dataset: 28,984 dermoscopy JPEGs (513 malignant / 28,471 benign, 1.77% positive); multi-image-per-patient with 1,457–2,056 shared train/test patients; metadata: sex, age_approx, anatom_site; original 800–6000px
- Metric: ROC AUC, higher is better
- Medal thresholds: gold ≈ 0.9455, silver ≈ 0.9401, bronze ≈ 0.937

## Best result
- Score: 0.94906 AUC | Medal: gold | Run: 20260421_191208
- Approach: EVA02-Base (MIM-pretrained IN22K) at 448×448, multi-task (melanoma BCE + 9-class diagnosis aux), focal (α=0.25, γ=2.0) + LS 0.05, metadata fusion (sex/age/anatom_site embeddings + GeM), 5-fold GroupKFold by patient_id, sqrt-weighted sampler, EMA (0.999), TTA + rank averaging. 5-fold mean val 0.9365 → test 0.94906.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260302_181942 | 0.921 | none | ConvNeXt-Base IN-22K 3-model ensemble |
| 20260305_193740 | 0.936 | none | EVA02-Base 448px; CutMix/Mixup; metadata fusion |
| 20260418_191318 | 0.886 | none | EfficientNetB4-NS 384px; GPU contention |
| 20260419_004053 | 0.945 | silver | EVA02-Base 448px; EMA; focal+LS 0.05; sqrt sampler |
| 20260419_162409 | 0.918 | none | EVA02-Base BCE(pos_weight=7.6)+1:3; no EMA |
| 20260420_193434 | 0.922 | none | EfficientNetV2-S IN21k 448px; 5-fold; CNN diversity |
| 20260419_004212 | 0.884 | none | EVA02-Base fold-0 fallback; NaN at unfreeze |
| **20260421_191208** | **0.94906** | **gold** | **EVA02-Base MIM anchor reproduced; 5-fold rank avg 0.9365 val** |
| 20260422_163250 | 0.93399 | none | EVA02-Large @560px (C2) submitted 0.936 val; C1 EVA02-Base 0.94477 NOT submitted |
| 20260417_204201 | 0.9212 | none | CAFormer-S36 multi-res + pseudo-label design; 4/5 folds rank-avg |
| 20260216_060348 | 0.89008 | none | Swin-V2-T 256px, 20× sampler, TTA; EffNet candidates starved |

## Proven strategies
- **StratifiedGroupKFold / GroupKFold by patient_id mandatory**: 1,457–2,056 shared patients (~44% overlap); inflates AUC 1–10 pts otherwise.
- **Multi-fold rank averaging = biggest lever**: 5-fold (0.945–0.949) >> 4-fold CAFormer (0.9212) >> single-fold 2-epoch (0.922) >> fold-0 fallback (0.884, 0.890).
- **Focal (α=0.25, γ=2.0) + LS=0.05 + sqrt sampler (7.4x)**: locally optimal; ASL/PolyLoss (0.914), BCE+pos_weight (0.918), plain BCE+class_weight (0.48 sanity) regress.
- **EMA (0.999)** with correct save order: +0.02–0.03 AUC.
- **Multi-task 9-class diagnosis head (weight 0.3)**: +0.02–0.03 AUC; aux supervision via ~5.3k diagnosis-labeled rows regularizes the 513 melanoma positives.
- **Metadata fusion (sex/age/anatom_site → MLP + GeM)**: +1–2% AUC; judges favor metadata candidates.
- **TTA (hflip+vflip+transpose) + rank avg**: +0.5–1% AUC.
- **bf16 autocast mandatory for ViT+focal** (fp16 → NaN on EVA02); grad clip=1.0 + 200-step warmup.
- **448px resolution sweet spot**: 256px capped 0.914 → test 0.890; 560px EVA02-L reached 0.936 but didn't beat 448px Base gold; 456px+ hangs on smaller models.
- **IN-22K MIM ViT dominates**: EVA02-Base (0.945–0.949) > EVA02-Large 560px (0.936) > CAFormer-S36 (0.9212 4-fold) > EffNetV2-S IN21k (0.922) > ConvNeXt-Base/V2 (0.917–0.921) > Swin-V2-T (0.890) > EffNetB4 IN1K (0.884).
- **Progressive unfreeze wait until epoch 5+**; earlier → NaN. LR 2× bump (1e-4→2e-4 head, 1e-5→2e-5 bb) gave +0.06 on Swin-V2-T.
- **Submission selection discipline**: submit the highest-val candidate, not the biggest model (20260422_163250 lost silver by submitting C2 0.936 EVA02-L instead of C1 0.94477 EVA02-Base).

## Failure patterns
- **Submission selection bug (20260422_163250)**: C1 graded 0.94477 (silver) but C2 at 0.93399 was submitted; add final-pick sanity check against val AUC ranking.
- **Judge over-pruning diversity candidates (20260422_163250)**: C3 (10-fold), C5 (preprocessing), C6 (multi-arch ensemble) all filtered early without logged reasoning; may lose untested upside.
- **Tuner/coder agent config-edit refusal** (20260417_204201 cand 1): add "allowed to edit config.py" to system reminder.
- **Tuner orchestration hangs** (20260211_072603, 20260301_200500, 20260216_060348): PL distributed/checkpoint-load leak; test in isolation.
- **External data unavailable in sandbox**: FCMAE SSL fallback 0.745; gate "external ISIC 2018/2019" on existence check.
- **Frozen backbone + GBDT head**: sanity passed but underexploited melanoma texture vs. end-to-end FT.
- **Pseudo-labeling incomplete at deadline / naive catastrophic (0.64)**; schedule as round-2 only.
- **Sandbox/symlink/permission aborts**: add sandbox preflight + judge_output.json guard.
- **GPU contention starves candidates** (EffNetV2-S 2/12 ep; cand 6 1/5 folds; EffNetB4-NS weight-load >30min; 20260422_163250 pivoted 5→2-3 folds). Lock GPU slot per candidate.
- **Swin-V2-T/S consistently last** at 256–384px (0.866, 0.890); 1024-dim bottleneck. Deprioritize.
- **ConvNeXt-V2-Large ceiling ~0.919** (20260422_163250 C4); CNN family caps below bronze 0.937.
- **Overfitting cliff epoch 5–6** (Swin-V2-T train 0.99+ vs val 0.914); patience=2–3.
- **Val-to-test gap 1–2.4%**: lock winning config; don't re-tune anchor.
- **DINOv2-B SSL (0.882) < supervised IN22K**; **ASL+PolyLoss (0.914) regress**.
- **Focal + hard 1:3 sampler + LS≥0.10 → 0.444 collapse**; plain BCE+class_weight → 0.476.
- **Aggressive augs (CLAHE+GridDistortion+ElasticTransform) → 0.720**; **EffNet-B5 @456px hangs**; **CAFormer-S36 NaN under fp16** (use bf16).
- **Sanity AUCs on 50–100 samples zero predictive value**.

## Feature engineering playbook
- **Metadata embeddings**: sex (dim=16, 3-way w/ unknown), anatom_site (dim=16, 8-way; 466 missing), age normalized → MLP (34→128→64) → concat w/ GeM image features.
- **Image aug**: RandomResizedCrop, h/v flips, shift-scale-rotate, color jitter, coarse dropout, CutMix+Mixup. Skip CPU-heavy dermoscopy augs (hair sim, vignetting, microscope-circle) — unvalidated regression risk.
- **Label smoothing 0.05 with focal**; LS≥0.10 collapses.
- **Timm string fragility**: verify `tf_efficientnet_b4.ns_jft_in1k`; prefer Noisy-Student / IN-22k.
- **bf16 autocast + GradScaler + grad accum + EMA** baseline works across B3/V2-S/Swin-T/CAFormer/EVA02.
- **Sampler sweep**: sqrt 7.4× > 20× > 1:1, 1:3; 10–50× unexplored.

## Tuning insights
- **Architecture ranking (test AUC)**: EVA02-Base 448px (0.9491) > EVA02-Large 560px (0.936) > CAFormer-S36 ens (0.932) / 4-fold (0.9212) > EffNetV2-S IN21k (0.922) > ConvNeXt-Base/V2 IN22K (0.917–0.921) > Swin-Base (0.921) > EffNetB4-NS (0.899) > Swin-V2-T 256px (0.890) > DINOv2-B SSL (0.882) > EffNetV2-M SWA (0.881) > Swin-V2-S (0.866).
- **EVA02-Large @560px does NOT beat EVA02-Base @448px**: scale-up ≠ win; anchor recipe is SOTA.
- **Loss**: Focal(α=0.25,γ=2.0)+LS=0.05 > BCE+pos_weight (-0.027) > ASL/PolyLoss (-0.022). γ=2–3 most stable.
- **LR**: 2e-4 head, 2e-5 backbone; AdamW + cosine + 200-step warmup; layer-wise LR decay helps.
- **Epoch budget**: 5–8 converge; patience 2–3; ≤2 ep → 0.907–0.922; ~1 ep/fold → 0.849.
- **Fixed-epoch budgets > step-based time budgets**.
- **Tuning plateau at ~0.93–0.94** (20260422_163250: 5 tuner runs oscillated 0.9403→0.9336→0.9403→0.936); need diversity, not more tuning.

## Open questions
- **Why did 20260422_163250 submission pick C2 (0.936) over C1 (0.94477)?** Add hard rule: submit max-val candidate.
- **Can EVA02-Base anchor recipe be reproduced to gold consistently?** 0.94906 (gold) vs 0.94477 (silver, same architecture) — seeding/OOF quality/per-fold filtering variance.
- **Rank-ensemble EVA02-Base + EVA02-Large + EffNetV2-S** (transformer/CNN diversity, each >0.92 floor) — untested, likely path past 0.9491.
- **Judge over-pruning**: force at least one diversity/CV candidate through if ≥1 anchor alive.
- **Tuner/coder config-edit refusal + PL/checkpoint leak**: instrument.
- **Sandbox preflight + external-data gate + judge_output.json guard + atomic submission write**.
- **GPU serialization**: lock per-candidate slot.
- **Full round-2 pseudo-labeling on test**; **oversampling sweep 10–50×**; **10-fold GroupKFold / stacked OOF**.
- **External ISIC 2018/2019** if sandbox permits; **LibAUC/margin-rank surrogate**; **metadata fusion formal ablation**.
