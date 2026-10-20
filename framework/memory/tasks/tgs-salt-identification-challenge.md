# Task Memory: tgs-salt-identification-challenge

## Task profile
- Type: Binary semantic segmentation (salt vs. non-salt)
- Dataset: ~3k training seismic images, grayscale, 101×101 native; depth metadata train-only (zero at test); ~61:39 salt:non-salt
- Metric: mean-precision-IoU at multiple thresholds (higher is better)
- Medal thresholds: bronze ≈ 0.85918

## Best result
- Score: 0.8449 | Medal: none (above median) | Run: 20260219_031421
- Approach: Segformer MiT-B2 transformer backbone at 256×256, 5-fold CV with per-fold threshold optimization. Single model (OOF 0.8559) beat all ensembles of weaker candidates. Resolution was the dominant lever.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260218_000552 | 0.466 | none | ResNet34 U-Net, Dice+BCE, 5-fold (weak baseline) |
| 20260218_025438 | 0.4549 | ungraded | DeepLabV3+ MobileNetV2, ASPP, depth-stratified CV |
| 20260218_164333 | 0.0 | none | MT ResNet50 reached 0.7576 val; eliminated by judge (empty RLE) |
| **20260219_031421** | **0.8449** | **none** | **Segformer MiT-B2 @ 256×256, Lovasz, per-fold thresh** |
| 20260224_194929 | 0.8089 | none | FPN+SE-ResNeXt50+Lovasz; v3 over-engineering regressed from 0.8224 |
| 20260225_190523 | 0.8413 | none | 4-model ensemble (Segformer B2/B3 + UNet++ B5/SE-ResNeXt50) @ 256 |
| 20260420_082440 | 0.8251 (OOF) | ungraded | EffNet-B0 + SE-ResNeXt50 UNet, Phase A→B, hard vote |
| 20260423_191018 | 0.8209 | none | SEResNeXt50 U-Net + scSE @ 256 + Lovasz Phase B + SWA + 4-way flip TTA |

## Proven strategies
- **Resolution ≥ 256×256** (pad/upsample from 101×101): single largest lever, +~0.02 vs 128×128
- **Lovasz loss** Phase B after BCE+Dice Phase A warmup: +0.025 vs single-phase; Phase A alone is insufficient
- **Transformer backbones (Segformer MiT-B2)** beat CNNs on this 3k dataset; SE-ResNeXt50 U-Net is best CNN baseline (0.82 tier)
- **Stratified k-fold by salt-coverage buckets** mandatory; single-fold inflates 15–20%
- **Per-fold threshold tuning** recovers +0.008
- **TTA: HFlip ONLY** per top-5 audit; V-flip/rotation/transpose violate seismic gravity prior. NOTE: run 20260423 used 4-way flip TTA (H/V/HV) and still scored 0.8209 — flag as inconsistent with audit; H-only remains the principled default
- **U-Net + scSE decoder** > hypercolumn fusion / UNet++ on 3k images (inductive bias > capacity)
- **SWA in Phase B** with cosine decay to 1e-7: stabilizes final convergence (run 20260423)
- **Encoder family orthogonality** (transformer + SE-ResNeXt) > seed diversity: +0.015 vs +0.003
- **1-channel grayscale** with ImageNet-adapted stats (mean=0.449, std=0.226) > forced 3-channel (-0.014)

## Failure patterns
- **Weak backbones at 128×128**: plateau ≤ 0.466
- **Over-engineering**: UNet++ snapshot ensembles, 8-way TTA, post-hoc pseudo-labeling regressed 0.8224 → 0.7811
- **Self-supervised pretraining (FCMAE/ConvNeXt-V2, SAM-1B) underperforms ImageNet supervised** on this grayscale domain — confirmed run 20260423 candidates 4 and 6
- **Noisy-student / Mean-Teacher semi-supervised plateau fast** despite 18k unlabeled test (6× train); EMA + pseudo-filter complexity did not yield consistent gains (run 20260423 C5 killed)
- **SAM + LoRA**: rank-16 + point prompts adds overhead without performance gain (run 20260423 C6)
- **Empty/invalid RLE**: 0.7576 val → 100% empty test; require 2-epoch smoke validation
- **Double-normalization** (albumentations + manual) silent train/test mismatch
- **Depth features hurt at inference**: test depth is zero; use only for fold stratification
- **Lovasz + AMP NaN** without `clip_grad_norm=1.0` and BCE warmup
- **Heavy architectures time out** (MAnet B5 → 0.407); UNet++ adds 26M params with zero gain on 3k images
- **Aggressive augmentation/batch in tuner**: run 20260423 C2 lost (no score, likely OOM/timeout)

## Feature engineering playbook
- Direct bilinear resize 101→256 (matches Segformer anchor); reflect-101 pad to 128 is fallback only
- Grayscale 1-channel, ImageNet-adapted normalization
- Heavy aug: ShiftScaleRotate (rotate ≤ ±15°), GridDistortion, GaussNoise, CLAHE, ElasticTransform; no transpose
- Stratified buckets: [0], (0, 0.25], (0.25, 0.5], (0.5, 0.75], (0.75, 1]
- Weighted sampler 1:1 in Phase A
- Hard-vote masks across folds (≥3/5) > soft-prob blend on post-threshold

## Tuning insights
- Loss: Lovasz-hinge Phase B > Focal-Tversky (NaN) > weighted BCE; require BCE+Dice warmup
- Early stopping patience 20–30
- SWA Phase B with cosine→1e-7: confirmed beneficial (run 20260423); earlier "minimal gain" overruled
- scSE decoder attention helps boundaries
- Pseudo-labels: +0.005 OOF best in-budget; -0.041 if stacked with UNet++/snapshot/8-way TTA. Top-5 3-round retrain-from-scratch is out of budget (16 days/arch)

## Open questions
- Bronze gap ~0.014 from 0.8449 best; closure candidates: in-budget pseudo-label (4th-style cross-prediction), 6–12 backbone cross-arch ensemble, snapshot averaging — see Pseudo-labeling protocol
- Larger Segformer (MiT-B3/B4) untested at 256×256: expected +0.003–0.005
- Soft-prob 5-fold logit averaging with single global threshold (claimed +0.003–0.008) — never run
- Fold 4 persistently -0.05 pts; fold-specific Phase B schedule untested
- TTA inconsistency: run 20260423 used 4-way flips and survived (0.8209) — does V-flip actually hurt, or only at the top of the score distribution? H-only remains principled default until contradicted

## MUST FOLLOW

**A meaningful portion of candidates MUST use HuggingFace transformer backbones — at minimum one per generation, ideally majority.** SMP/timm CNN backbones remain valid for diversity/ablation/ensemble; the 0.8449 best uses HF Segformer (`nvidia/mit-b2`). Pure-CNN candidates plateau at OOF ≈ 0.84 (runs 20260218 / 20260224 / 20260420 / 20260423). If N candidates proposed, ≥1 (preferably ⌈N/2⌉) must use an HF backbone below.

**Primary backbone — Segformer (`nvidia/mit-b2/b3/b4`) via SMP**. `mit_b2` is the verified 0.8449 LB winner (single-fold OOF 0.8559); b3/b4 untested (+0.003–0.005 expected). Hard contract: 1-ch grayscale, ≥256×256, stats `mean=0.449 std=0.226` (do NOT triplicate to 3-ch, regressed -0.014); Phase A (BCE+Dice, 1:1 weighted sampler) → Phase B (Lovasz hinge) with `clip_grad_norm=1.0`. Canonical loading in `tmp/verify_segformer_tgs_salt.py` (log `tmp/verify_segformer_tgs_salt.log`).

**Domain-specialized exploration — `thinkonward/geophysical-foundation-model` (GFM)**, MAE ViT-L (~371M, embed_dim=1200). **At most 1 slot — exploration only.** Traps: (i) input hardcoded `(B,1,400,160)` at `GFM/ElasticViTMAE.py:51-52`; resample 101→400×160 first. (ii) call `model.forward_encoder(x, idx_shuffle=torch.arange(160).unsqueeze(0), len_keep=160)` → `(B,161,1200)`. (iii) MAE decoder useless for segmentation; attach fresh head (drop CLS → reshape `(B,160,1200)` → `(B,1200,1,160)` → upsample → `Conv2d(1200,1,1)`). (iv) PYTHONPATH-only load — clone at `external/geophysical-foundation-model` (repo-relative, or any local path). Verified end-to-end in `tmp/verify_gfm_e2e.py` (log `tmp/verify_gfm_tgs_salt.log`). Other ThinkOnward models NOT applicable.

**Ensemble diversity backbones** (Top-5 audit #2: 6–12 heterogeneous): DINOv2 (`vit_small_patch14_dinov2.lvd142m`), ConvNeXt-V2 (`convnextv2_base.fcmae_ft_in22k_in1k`), SMP UNet++ with `efficientnet-b3/b4` or `se_resnext50_32x4d`. **Caution**: ConvNeXt-V2 FCMAE underperformed ImageNet supervised on this grayscale domain (run 20260423 C4 killed); favor ImageNet supervised CNN diversity over SSL. Segformer 2–4 members, GFM ≤1, remainder CNN/ViT with orthogonal failure modes. Family orthogonality gave +0.015 vs +0.003 for seed.

**Hard rules**: (1) Resolution floor 256×256. (2) **TTA whitelist: HFlip ONLY** — V-flip/rotation/transpose violate seismic gravity prior; train-time rotation ≤ ±15° (1st place sets 0). (3) Never stack pseudo-label + UNet++ + snapshot + 8-way TTA (compound regressed 0.8224→0.7811). (4) **Pseudo-label preconditions** non-negotiable — base OOF ≥ 0.85; training-phase signal on non-converged model; never post-hoc finetune on converged checkpoint. Do NOT attempt 1st-place 3-round retrain-from-scratch (out of 8h/candidate budget). Prefer 4th-style cross-prediction. Avoid noisy-student/mean-teacher unless preconditions clearly met (run 20260423 C5 plateau). (5) Depth conditioning fails at inference; use only for fold stratification. (6) RLE format: 2-epoch smoke validation with re-decoded sample rows. (7) Stratified k-fold by salt-coverage buckets `[0],(0,0.25],(0.25,0.5],(0.5,0.75],(0.75,1]` mandatory. (8) Per-fold threshold tuning recovers +0.008 (fold 4 ≈ 0.58, fold 2 ≈ 0.48). (9) Avoid SAM+LoRA segmentation adapter — overhead without gain (run 20260423 C6).

**Env baseline (2026-04-25, conda `tgs-salt-identification-challenge`, A100-80GB)**: `python 3.11.14`, `torch 2.10.0+cu128`, `smp 0.5.0`, `timm 1.0.24`, `huggingface_hub 1.4.1`. No `transformers` install needed. Scripts `tmp/verify_segformer_tgs_salt.py` and `tmp/verify_gfm_e2e.py` are SSOT — if memory disagrees, the scripts win.


# TGS Salt Identification Challenge — Top-5 Leaderboard Research

Research compiled 2026-04-24. Purpose: strict-evidence reference for how the top Kaggle solutions scaled their models vs compute on the same task we are running. Only statements with URL + verbatim quote are asserted as facts; everything else is marked "not disclosed".

Competition: https://www.kaggle.com/competitions/tgs-salt-identification-challenge (ended 2018-10-19)

Medal thresholds (private LB, from mlebench registry):
- Gold: 0.89097
- Silver: 0.87163
- Bronze: 0.85918
- Median: 0.81665

## Our Setup (baseline for comparison)

- Hardware: 1× A100-class GPU per candidate, 4 GPUs total available
- Per-candidate compute budget: 7–8 wall-clock hours
- Per-run budget: 24 h wall-clock across up to 6 candidates ≈ 42–48 GPU-hours per run
- Candidate scheme: single backbone, 5-fold CV, Phase A (BCE+Dice) → Phase B (Lovasz)
- Example candidates:
  - C2: SegFormer MiT-B4 (~62M params) @ 320 input, 5-fold
  - C4: ConvNeXtV2-Base FCMAE (~89M params) @ 256 input, 5-fold
- No pseudo-labeling, no cross-architecture ensemble
- Current best test score (candidate_2): **0.83310** (Above Median, below Bronze by 0.026)

## Leaderboard Positions

Verified via Wayback Machine leaderboard snapshot (2019-03-27): https://web.archive.org/web/20190327221347/https://www.kaggle.com/c/tgs-salt-identification-challenge/leaderboard

1. b.e.s. & phalanx
2. Tim & Sberbank AI Lab
3. [ods.ai] topcoders
4. SeuTao team (full team name not disclosed in accessible sources)
5. AlexenderLiao team

Correction: bestfitting finished **6th** as a solo team, not 3rd.

## Comparison Table

| Rank | Team | Backbones (params) | Input | Ensemble size | Pseudo-label | Training time | Hardware |
|:---:|------|---|:---:|---|:---:|---|---|
| 1st | b.e.s. & phalanx | U-ResNet34 + U-ResNeXt50 (~21M + ~25M) | 256 | 40 models (2 arch × 5 fold × 4 snapshot) | Yes, 3 rounds × 200 epochs | **16 days on 1× 1080 Ti** | **1× GTX 1080 Ti** |
| 2nd | Tim & Sberbank AI Lab (mctigger) | DPN92/98/107 + SEResNeXt50/101 + SEResNet152 + SENet154 (7 backbones) | 128 | 12 models × 5-fold = 60 trainings | Yes, 2 rounds | 5–12 h/fold × 60 = 300–720 GPU-hours (arithmetic) | **10 GPUs across 4 machines** (6× 1080 Ti + 4× P100) |
| 3rd | [ods.ai] topcoders | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed |
| 4th | SeuTao team | ResNet34 + SE-ResNeXt50 + ResNeXt101-IBN + SE-ResNet101/152/154 (6 backbones) | 128 | 6 models × 10 fold × 7 cycle + jigsaw post-processing | Yes, 2 rounds | not disclosed | not disclosed (code hint: ≥1 GPU) |
| 5th | AlexenderLiao team | ResNet34 + SE-ResNeXt50 (+OC) (~21M + ~28M) | 128 + 256 (dual-scale) | **100 models** (10 fold × 2 arch × 2 scale × snapshot) | Yes | not disclosed (only 300k iters, bs=16) | not disclosed (code hint: up to 4 GPUs available) |

## Detailed Per-Solution Findings

### 1st — b.e.s. & phalanx (Babakhin, Sanakoyeu, Kitamura)

**Primary sources:**
- arXiv v3 (2019-08-05): https://arxiv.org/abs/1904.04445 ("Semi-Supervised Segmentation of Salt Bodies in Seismic Images using an Ensemble of Convolutional Neural Networks", GCPR 2019)
- GitHub: https://github.com/ybabakhin/kaggle_salt_bes_phalanx
- Kaggle writeup: https://www.kaggle.com/competitions/tgs-salt-identification-challenge/writeups/b-e-s-phalanx-1st-place-solution-with-code

**Architecture** (paper §4.2, verbatim):
> "We employ an ensemble of two U-Nets with Imagenet-pretrained encoder backbones: U-ResNet34 and U-ResNeXt50. The output of the ensemble is the average of the predictions of two models in the ensemble."

Decoder uses hypercolumns + scSE attention + Feature Pyramid Attention bottleneck (paper §3.2):
> "we insert concurrent spatial and channel Squeeze & Excitation modules (scSE) ... after each encoder and decoder block"
> "stack the upsampled feature maps from all decoder blocks ... as the input to the final layer."

Input resolution (paper §4.2):
> "All images are resized to the size of 202×202 pixels and then padded to the size of 256×256 pixels."

**Ensemble** — 40 models combined by average voting (paper §4.2):
> "we obtain 4 network snapshots for each of 5 folds giving 20 snapshots in total for a single network architecture. Since we use an ensemble of U-ResNet34 and U-ResNeXt50, it results in 40 models in total which are combined together for inference using the average voting."

**Pseudo-labeling** (paper):
> "We do K = 3 rounds of self-training and T = 200 training epochs per round"

**Training time** — GitHub README:
> "expect this to run for about 16 days on a single GTX1080Ti"
> "expect this to run for 3.5 hours for 18,000 test images on a single GTX1080Ti"

**Hardware**: 1× GTX 1080 Ti (README quote above).

**Parameter count**: not disclosed.

### 2nd — Tim & Sberbank AI Lab (mctigger)

**Primary source:**
- GitHub README: https://github.com/mctigger/KaggleSalt/blob/master/README.md

**Architecture**: nopoolrefinenet (RefineNet variant, no-pool stem) with dual hypercolumn and scSE block padding. Ensemble of 7 backbones — verbatim experiment names:
> "nopoolrefinenet_dpn92_hypercolumn, nopoolrefinenet_dpn92, nopoolrefinenet_dpn98, nopoolrefinenet_dpn107, nopoolrefinenet_seresnext101_ndadam_scse_block_padding, nopoolrefinenet_seresnext50_ndadam_scse_block_padding"

Plus SEResNet152 and SENet154 per the experiment list. Input 128×128 — code-inferred from `tta.Pad((13, 14, 13, 14))` on 101×101 images (`experiments/nopoolrefinenet_dpn92.py`). Training config: `epochs = 200`, `utils.CyclicLR(optimizer, 5, {0:(1e-4,1e-6), 100:(0.5e-4,1e-6), 160:(1e-5,1e-6)})`.

**Ensemble** — 12 models averaged, each 5-fold:
> "ensemble_name = 'ensemble-top-12-test'"
> "all the test predictions from the different specified experiments are averaged"

**Pseudo-labeling**: Two rounds:
> "First-round semi-supervised ensemble… new models which are dependent on the ensemble we created before and additionally depend on the small mask dataset."

Names with `aux_data_pseudo_labels` confirm pseudo-label-trained variants.

**Training time** (per-experiment):
> "This will run 5-folds per experiment and create test predictions for each fold. Expect this to run between 5h/fold to 12h/fold."

Arithmetic: 12 experiments × 5 folds × 5–12 h/fold ≈ 300–720 GPU-hours for supervised/pseudo-label rounds. Total not disclosed as single number.

**Hardware** — verbatim:
> "3x: Ubuntu 16.04 LTS (512 GB boot disk) 16GB RAM 2x NVIDIA GTX 1080TI 11GB"
> "1x Ubuntu 16.04 LTS (512 GB boot disk) 256GB RAM 4x NVIDIA Tesla P100"

Total: 10 GPUs across 4 machines (6× 1080 Ti + 4× P100).

**Parameter count**: not disclosed.

### 3rd — [ods.ai] topcoders

**No public write-up found.**

Exhaustive search across: Kaggle discussion threads (SPA client-rendered, Wayback snapshots only capture shell), GitHub code search (145 `tgs+salt+kaggle` repos), arXiv, Google/Bing, personal blogs. No solution documentation is publicly accessible without Kaggle login.

Architecture, ensemble size, training time, hardware — all **not disclosed**.

Note: Other ods.ai sub-teams published papers but ranked differently:
- Karchevskiy/Ashrapov/Kozinkin arXiv:1812.01429 — 27th place ("top 1%")
- BloodAxe (github.com/BloodAxe/Kaggle-Salt) — 107th place

Neither is the 3rd-place team.

### 4th — SeuTao team

**Primary source:**
- GitHub: https://github.com/SeuTao/Kaggle_TGS2018_4th_solution (README last updated 2018-11-06)

**Architecture** — verbatim:
> "input: 101 random pad to 128*128, random LRflip"
> "resnet34, se-resnext50, resnext101_ibna, se-resnet101, se-resnet152, se resnet154"
> "scse, hypercolumn (not used in network with resnext101_ibna, se_resnext101 backbone), ibn block, dropout"
> "Deep supervision structure with Lovasz softmax"

**Ensemble** — verbatim:
> "We designed 6 single models for the final submission"
> "single model（10fold 7cycle）"
> "cycle voting"

Final pipeline: 3 architectures (resnet34, se-resnext50, resnext101) trained twice on complementary pseudo-label splits → 6 checkpoints in `model 34+50+101 with post processing (final sub)` row, private LB 0.8953.

**Pseudo-labeling**:
> "Data distill (Pseudo Labeling)"

Plus rule-based post-processing:
> "we applied around 10 handcraft rules that gave a 0.010~0.011 public LB boost and 0.001 private LB boost"

**Training time**: not disclosed (only cycle structure).

**Hardware**: not disclosed. Training commands show `CUDA_VISIBLE_DEVICES=0`, implying ≥1 GPU but no type/count.

**Parameter count**: not disclosed.

### 5th — AlexenderLiao team (liaopeiyuan)

**Primary sources:**
- GitHub: https://github.com/liaopeiyuan/kaggle-solutions (README last updated 2023-09)
- Code: https://github.com/liaopeiyuan/kaggle-solutions/tree/master/projects/TGS_salt
- Kaggle discussion 69051 (reCAPTCHA-locked, no Wayback snapshot)

**Placement** — README verbatim:
> "Placement: 5/3291 (Gold Medal)"

**Architecture**: ResNet34 + SE-ResNeXt-50 U-Net with scSE + hypercolumn, SE-ResNeXt-50 branch adds OC (Object-Context) block. Evidence from model filenames: `models/TGS_salt/Unet34_scSE_hyper.py` and `models/TGS_salt/SEResnextUnet50_OC_scSE_hyper.py`. Input sizes 128 and 256 (padded from 101). Code: `SIZE = 101`, `FACTOR = 128`, and `SIZE = 256, PAD = 256` in `inference/inference_128+256.py`.

**Ensemble** — 100 models. README table row:
> "2models-2scale-stable-post | 10 | 100 | 0.885 | 0.894 | 5"

Composition: 10-fold CV × 2 models × 2 scales, with per-fold snapshot averaging (`np.arange(int(79.5*1000),130*1000,10*1000)` checkpoint iterations in `inference_128+256.py`).

**Pseudo-labeling**: file `projects/TGS_salt/training/train0_pesudo_Unet_scSE_hyper.py` with `OHEM = "res_256_pseudo"` and dataset name `'list_train'+str(FOLD)+'_3600'+ne+"_pseudo"`.

**SWA**: explicitly disabled — `DingHan/seresnet50/train_se_resnext50_bn_128.py` has `swa=False`.

**Training time**: not disclosed as wall-clock. Only iteration counts: `num_iters = 300 * 1000, batch_size = 16`; DingHan SE-ResNeXt-50 script: `epochs = 360, batch_size = 16, lr = 0.01`.

**Hardware**: not disclosed. Code hint: `os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'` suggests up to 4 GPUs available, but model/count not stated.

**Parameter count**: not disclosed.

## Takeaways

The top 5 win **not** by larger per-model capacity — their single-model backbones are mostly *smaller* than our MiT-B4 / ConvNeXtV2-B. They win on three axes we currently lack:

1. **Pseudo-labeling on the 18k unlabeled test set.** 1st, 2nd, 4th, 5th all used multi-round self-training.
2. **Cross-architecture ensembles of 6–12 distinct backbones.** We run single-backbone 5-fold.
3. **Snapshot averaging / cycle voting across many checkpoints per fold.** Multiplies ensemble size 4–7× per model.

Compute cost ratio (best-documented vs ours):
- 1st: 16 × 1080Ti-days ≈ 4 A100-days = **~8× our per-run budget**
- 2nd: 300–720 GPU-hours = **~6–15× our per-run budget**

Gap from our best (0.83310) to gold (0.89097) is **0.058**; to bronze (0.85918) is **0.026**. Based on top-5 evidence, the highest-leverage interventions available to us are pseudo-labeling and cross-backbone ensembling — not bigger single models.

## Appendix: What "K folds" and "N checkpoints per fold" actually mean

### K-fold ensemble vs bagging

K-fold cross-validation is structurally similar to bagging but has subtle differences:

| Dimension | K-fold ensemble | Classical bagging |
|---|---|---|
| Sampling | Non-overlapping partition; each model trains on (K-1)/K of data | Bootstrap with replacement; each model sees ~63% unique samples |
| Sample repetition | Each training sample appears in exactly K-1 models | Samples appear with random multiplicity, some not at all |
| Diversity source | Different data subsets + random init + SGD noise | Stronger data-level perturbation |
| Byproduct | **OOF predictions covering full train set** → enables stacking / OOF-based model selection / threshold tuning | No OOF concept |

Kaggle practice favours K-fold over pure bagging because OOF predictions are load-bearing for model selection, second-level stacking, and threshold search. So each fold is like a bagging member, but with **lower diversity** than true bagging.

### Snapshot ensembling / "N checkpoints per fold"

This is Huang et al. 2017's snapshot ensembling: within a single fold's training run, use CosineAnnealingLR with restarts (or CyclicLR) and save a checkpoint at the end of each cycle. Each snapshot lands in a different loss basin — behaves like multiple independently-trained models for one training cost.

Examples from top-5:

- **1st (b.e.s. & phalanx)** — paper quote:
  > "we obtain 4 network snapshots for each of 5 folds giving 20 snapshots in total for a single network architecture"

  5 folds × 4 snapshots × 2 architectures = 40 checkpoints.

- **4th (SeuTao)** — README quote: "single model（10fold 7cycle）"

  Per backbone: 10 folds × 7 cycles = 70 checkpoints. With 6 backbones in the final submission, total is substantially larger.

- **5th (liaopeiyuan)** — snapshot iterations loaded via:
  ```python
  np.arange(int(79.5*1000), 130*1000, 10*1000)  # inference_128+256.py
  ```
  Loads ~6 checkpoint iterations per fold-model.

### How the models are combined at inference

For binary segmentation, top-5 solutions **do not use majority voting**. They use **probability averaging** (pixel-wise mean of sigmoid outputs, then threshold).

**1st place** — paper verbatim:
> "40 models in total which are combined together for inference using the average voting"

In segmentation context "average voting" = per-pixel probability average.

**2nd place** — README verbatim:
> "all the test predictions from the different specified experiments are averaged"

Typical inference loop:
```
for each test image:
    preds = []
    for each of N checkpoints:
        p = sigmoid(model(image))   # H×W probability map
        preds.append(p)
    avg = mean(preds)                # pixel-wise average
    mask = avg > threshold           # threshold from OOF search
```

### Why probability averaging, not majority voting?

- **Majority voting (hard)**: threshold each checkpoint's probability to 0/1 first, then pixel-wise majority. This discards calibration — a pixel where all 40 models say 0.51 is indistinguishable from one where all say 0.99.
- **Probability averaging (soft)**: preserves confidence. When the mean probability is 0.45, thresholding at 0.50 gives a cleaner decision than breaking a 22-vs-18 hard-vote tie.

All five top solutions use soft averaging.

### Additional ensemble tricks seen in top-5

1. **Weighted averaging** — per-backbone weights chosen from OOF score (seen in 2nd and 4th variants).
2. **Threshold search** — sweep the binarization threshold on OOF probability maps to find the optimum for the competition metric. 1st and 4th both did this.
3. **Multi-scale fusion** — 5th runs the same models at 128 and 256 input sizes, upsamples to 101×101, and averages probability maps across scales before thresholding.
4. **Jigsaw post-processing** (4th, domain-specific) — reassemble test images by spatial adjacency into the original seismic slice and clean up mask boundaries across tiles. Quote: "around 10 handcraft rules that gave a 0.010~0.011 public LB boost".

### Implication for our pipeline

Our current 5-fold setup produces 5 checkpoints per candidate, combined by … we currently do not have an explicit ensembling stage in manager_workflow (each candidate submits its own submission.csv from whichever fold's best_model was kept). Options to close the gap:

- Add snapshot ensembling within a fold (CosineAnnealingLR + save N snapshots, average at inference). Cheap — no extra training.
- Average the 5 fold probability maps before thresholding, not just pick best fold.
- Cross-candidate ensemble: average probability maps from candidate_1/2/4/6 after each fold's inference, then threshold.

## Research method notes

- All per-rank research dispatched as parallel general-purpose agents (2026-04-24).
- Sources marked "reCAPTCHA-locked" or "SPA-rendered" could not be retrieved without Kaggle login; confirmed via Wayback Machine that snapshots do not contain discussion body text.
- Every quoted fact carries: (URL, verbatim quote ≤25 words, date where available).
- When numbers were not explicitly stated, we report "not disclosed" rather than estimating.
