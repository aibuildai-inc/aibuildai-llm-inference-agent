# Task Memory: vesuvius-challenge-ink-detection

## Task profile
- Type: Binary semantic segmentation (pixel-level ink detection on volumetric scans)
- Dataset: Small (2 training fragments), 65-slice 3D X-ray volumes, severe cross-fragment distribution shift
- Metric: F0.5 (precision-weighted, direction: higher is better)
- Medal thresholds: median ≈ 0.54 (above-median achieved at 0.54236)

## Best result
- Score: 0.54236 | Medal: none (above median) | Run: 20260227_081642
- Approach: EfficientNet-B5 UNet with 2.5D input using only z-slices 15–49, per-fragment z-score normalization, 256×256 tiles trained on both fragments with strong augmentation, and 4-way flip/rotation TTA at inference.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260218_025451 | 0.10104 | none | Cascaded 2-stage coarse-to-fine; internal F0.5=0.5587 but catastrophic submission encoding gap (5.5×) |
| 20260227_081642 | 0.54236 | none (above median) | EfficientNet-B5 2.5D, z=15–49, per-fragment z-score, 4-way TTA |
| 20260420_082632 | 0.5578 (holdout, ungraded) | ungraded | EfficientNet-B3 16-ch z-window 24–39, spatial holdout, ensemble 60/40 + threshold=0.26, min_size=3200 |

## Proven strategies
- **2.5D mid-range z-slicing beats full 3D**: z=15–49 (run 2) and z=24–39 16-ch window (run 3) both dominate full 65-slice volumes and pure 3D U-Nets (which collapsed to F0.5=0.0 in run 1).
- **Per-fragment z-score normalization**: essential for cross-fragment generalization; intensity distributions differ drastically.
- **EfficientNet backbones dominate**: B5 >> B4 >> ResNet34 >> ConvNeXt-Tiny (run 2); B3 stable with ImageNet 3→16ch first-conv adapt (run 3).
- **TTA in probability space**: 4-way flip/rotate (run 2) and D4 TTA (run 3) consistently improve robustness.
- **Aggressive thresholding for F0.5**: default 0.5 punishes recall; tuned optima 0.7–0.85 (run 2) or 0.26 with min_size=3200px post-processing (run 3, non-intuitive but robust).
- **Spatial holdout > fragment-LOO** (run 3): sub-region holdout with buffer better mirrors test geometry; fragment-LOO plateaus at F0.5 0.47–0.50.
- **Judge gates catch implementation bugs early** (run 1: 4/6 candidates killed pre-training).

## Failure patterns
- **Severe train-test gap**: run 2 validation F0.5=0.929 vs leaderboard 0.54236 (70+ pt gap); run 1 internal 0.5587 vs final 0.10104 (5.5× gap, likely RLE encoding mismatch).
- **Pure 3D U-Net collapse**: F0.5=0.0 (run 1, Candidate 7); 2D-3D fusion diverged.
- **Wider Z-windows dilute signal**: run 3 24-ch F0.5=0.4902 vs 16-ch 0.5375.
- **Late-stage design revisions risky**: run 1 redesign at step 16 introduced label binarization bugs across 4 new candidates (catastrophic).
- **Fragment-LOO plateau unbreakable**: run 3 attempts 2/3/4/8 stuck at 0.47–0.50 regardless of loss/reg/LR tuning.
- **Heavy regularization harmful** (run 3, attempt 8): reviser flagged but not retrained.

## Feature engineering playbook
- Z-slice selection: 2.5D window 15–49 (run 2) or 16-ch 24–39 (run 3); avoid full 65 or pure 3D.
- Per-fragment z-score normalization (not global).
- 256×256 tile size with strong augmentation; train on both fragments jointly.
- First-conv channel-adapt from ImageNet 3-ch to 16-ch is stable.
- Post-processing: threshold + min-size component filtering; optimum varies non-intuitively (0.26 + 3200px in run 3 vs 0.7–0.85 in run 2).

## Tuning insights
- Loss: 0.5*BCE + 0.5*Dice locked in (run 3); Focal/Tversky unexplored.
- Ensemble: correlation-filtered weight sweep (66 combos × 12 threshold/min_size pairs) gives +0.018 F0.5 reliably; ridge is flat (F0.5 [0.552, 0.558]) so robust to noise.
- Low-correlation attempts get zero weight (run 3 attempt 23a2, r=0.789).
- Early-stop ep3 avoids cold-start divergence (run 3).
- Threshold variance ±0.08 F0.5 across attempts — always sweep on holdout.

## Open questions
- Root cause of 5.5× submission gap in run 1 (RLE corruption? metric mismatch? prediction shape?).
- Whether run 3 holdout F0.5=0.5578 generalizes to true test (ungraded).
- Fold-1 (train frag1, val frag2 full) untried due to 20-min budget; expected +0.02–0.05 F0.5.
- Larger backbone (B5) + stochastic depth for wider Z-windows to recover 24-ch signal.
- Multi-fold ensemble within the winning Candidate 1 (run 2) could recoup 1–2 LB points.
- Depthwise-separable or sparse 3D conv to unlock volumetric patterns without overfitting (run 2 Candidate 4 had best held-out 0.516 but was abandoned).

## MUST FOLLOW

**A meaningful portion of candidates MUST use HuggingFace domain-pretrained backbones — at minimum one candidate per generation, and ideally the majority should explore this path.** SMP / timm ImageNet backbones are NOT banned; they remain valid for diversity, ablation, and as a known-working baseline (the current 0.54236 run uses `smp.Unet(efficientnet-b5, encoder_weights='imagenet')` and is allowed to continue as one of the candidates). The reason for the allocation requirement is mechanistic: 5 of 5 Kaggle Top-5 solutions use 3D-aware encoders with domain-aligned pretraining (Kinetics/IG65M video weights or scroll-CT self-supervised weights), and our run-2 demonstrated empirically that a 2D EfficientNet with z-as-channels and a random-init 34→40 stem caps out around F0.5 ≈ 0.54 — roughly 13 points below the Top-5 cluster (~0.67). The two HF checkpoints below are 3D-aware, single-channel, and pretrained on actual carbonized-papyrus X-ray CT volumes; using at least one of them per design round is what gives the candidate pool a chance to break through the 0.55 ceiling that pure-ImageNet candidates have repeatedly hit. If the designer proposes N candidates, at least 1 (preferably ⌈N/2⌉) must use one of the HF backbones below; the remaining candidates may continue using SMP/timm with explicit justification.

**Primary backbone — `scrollprize/timesformer_GP_scroll1`** (38M, TimeSformer-small with divided space-time attention, MIT). The actual end-to-end-trained checkpoint behind the Vesuvius Challenge 2023 Grand Prize — pretrained head `backbone.to_out.1` confirmed at trained-distribution stats. Hard contract not derivable from a smoke test alone: input is **strictly** `(B, 1, 26, 64, 64)` with C=1 (z is the temporal axis, NOT z-as-channels) — `num_frames=26` and `tile=64` are hardcoded; output is `(B, 1, 4, 4)` at 1/16 stride and **must** be bilinear-upsampled by 16 to tile resolution before stitching (skipping this step yields correctly-shaped but spatially-quantized garbage). Preprocessing must match training: `np.clip(image, 0, 200) / 200.0`, **NO per-fragment z-score** — z-score shifts inputs off the trained manifold. Z-slices: pick 26 from the central band (official inference uses 17..43). Inference is sliding-window stride=32. Loading code, pretrained-head check, and real-data forward (sigmoid mean 0.750 on a frag1 ink-rich patch — i.e. genuinely ink-aware) all verified in `tmp/verify_hf_backbones.py` (Test 1) — read that script for the canonical `AutoModel.from_pretrained(..., trust_remote_code=True)` invocation rather than re-deriving.

**Secondary backbone — `scrollprize/resnet50_3um_01122024`** (85.7M, 3D ResNet-50 + 2D UNet decoder). Flexible input `(B, 1, T, H, W)` with T∈[16,32], H,W∈[128,384]; output `(B, 1, H/4, W/4)`. Use case: encoder swap for an SMP-UNet style pipeline — replace EfficientNet-B5 stem with the 3D-CT-domain-pretrained encoder. **Silent decoder-random-init trap (load-bearing):** HF logs `"All model checkpoint weights were used"` with zero missing keys, but the decoder BN `running_mean`/`running_var` are exactly all-zero / all-one and conv weights have Kaiming-init std — the .safetensors contains decoder tensors saved at construction time, untrained. The 1139-class `backbone.fc` head is also dead weight. Empirical confirmation: real-data sigmoid mean 0.504 (random, vs timesformer's 0.750 on the same patch). Therefore: encoder is useful, but **must fine-tune (decoder always, encoder optional) before any submission** — same silent-trap pattern as TGS Salt's GFM. Loading + pretrained-encoder-vs-untrained-decoder distinction is verified in `tmp/verify_hf_backbones.py` (Test 2) — read that script.

**Failure cases not reproducible from a smoke test alone — hard rules:** (1) `timesformer_GP_scroll1` input must be C=1 not z-as-channels: feeding `(B, 26, 64, 64)` will not crash but produces wrong outputs because the model `permute`s internally assuming C=1; `num_frames` and `window_size` are baked into the TimeSformer construction. (2) `scrollprize/dinov2_ps8` is **NOT directly loadable**: empty README, no `auto_map`, no custom modeling `.py`, the `.pt` is a 5 GB raw pickle from an external `ScrollPrize/dinovol_2` training repo. Do not propose this without an explicit >1-day integration approval. (3) `trust_remote_code=True` is required and downloads small `.py` modeling files at first load — runtime must have internet on first load, subsequent runs use `$HF_HOME` cache. (4) If the loading log shows ANY missing/unexpected keys, stop and investigate — both checkpoints are verified to load with zero divergence; any deviation indicates a transformers-version mismatch or corrupted download.

**Env baseline + canonical verify script (2026-04-24, conda env `vesuvius-challenge-ink-detection`)**. `torch==2.6.0+cu124`, `transformers==4.48.1`, `timesformer-pytorch==0.4.1`, `einops==0.8.2`, `huggingface_hub==0.36.2`, `timm==1.0.24`, `segmentation_models_pytorch==0.5.0`, `albumentations==2.0.8`, `scikit-learn==1.8.0`. Note: installing `transformers==4.48.1` downgraded `huggingface_hub` 1.4.1 → 0.36.2 (transformers requires `<1.0`); benign for HF model loading. Single source of truth for both backbones: `tmp/verify_hf_backbones.py` (Test 1 = timesformer, Test 2 = resnet50_3um). If a future change to memory disagrees with this script, the script wins — fix the description, not the test.

# Vesuvius Challenge - Ink Detection — Top-5 Leaderboard Research

Research compiled 2026-04-26. Purpose: strict-evidence reference for how the top Kaggle solutions structured their 3D-aware pipelines vs compute on the same task we are running. Only statements with URL + verbatim quote are asserted as facts; everything else is marked "not disclosed".

Competition: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection (ended 2023-06-14)

Metric: **F0.5** (precision-weighted F-beta, higher is better).

Private leaderboard top-5 scores (Kaggle private LB, original scoring):
- 1st — ryches: 0.682943
- 2nd — RTX23090: 0.681739
- 3rd — traptinblur: 0.679859
- 4th — POSCO DX (Heeyoung Ahn): 0.673941
- 5th — aksell: 0.673795

Note on scoring: the mle-bench harness re-scores submissions on a different held-out split (fragment_a only) so its medal thresholds (median ≈ 0.54) and our local run scores (best 0.5578–0.6183 across recent runs) are NOT directly comparable to the Kaggle private LB numbers above. Top-5 figures throughout this document are **Kaggle private LB** unless stated otherwise.

## Our Setup (baseline for comparison)

- Hardware: 1× A100-class GPU per candidate, 4 GPUs total available
- Per-candidate compute budget: 7–8 wall-clock hours
- Per-run budget: 24 h wall-clock across up to 6 candidates ≈ 42–48 GPU-hours per run
- Candidate scheme (run 20260425_011945, 6 candidates):
  - C1: EfficientNet-B5 UNet 2.5D, z-as-channels (16-ch), tile 256, single backbone 5-fold-style
  - C2: TimeSformer-GP (scrollprize/timesformer_GP_scroll1) at 64-tile/26-frame hard contract
  - C3: ResNet50-3um-pretrained 3D + 2D-UNet decoder
  - C4: 3D backbone trained from scratch
  - C5: TimeSformer multi-stride + multi-z-jitter ensemble (multi-stride at inference)
  - C6: heterogeneous ensemble aggregation (no training)
- No mixup, no cutmix, no channel-shuffle in any candidate; only C5 enabled z-shift ±2 (the rest hard-coded `Z_SHIFT_MAX=0`)
- No multi-resolution training in any candidate
- No fragment-2 sub-splitting beyond a 50/50 vertical 2a/2b in some candidates
- Current best test score (run 20260420 spatial holdout F0.5): **0.5578** (Above Median, below Bronze on Kaggle by ~0.13)

## Leaderboard Positions

Verified via mlebench-stored leaderboard.csv (Kaggle private LB snapshot at competition close 2023-06-14):

1. ryches (Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol)
2. RTX23090 (= tattaka + mipypf / Ron + yukke42)
3. traptinblur
4. POSCO DX (Heeyoung Ahn) [Kaggle handle: ahnheeyoung1]
5. aksell

Score-per-rank cross-check: Bing snippet returned 0.682693 for ryches vs the appendix-recorded 0.682943; both kept in heading. Small discrepancy in the 4th decimal does not affect any analysis below.

## Comparison Table

| Rank | Team | Backbones (params) | Input tile | Z-window of 65 | Ensemble size | Multi-resolution | Pseudo-label | Hardware | Training time |
|:---:|------|---|:---:|:---:|---|:---:|:---:|---|---|
| 1st | ryches | SegFormer mit-b3/b4/b5 + 3D-CNN-stem / UNet3D-stem / UNETR-stem (≈47–82M per Segformer; stem 10K–30M each) | **1024 + 512** dual-tile mix | central **16** (24–39) | **9** models, all SWA, sigmoid-mean | No (single tile/model, mixed at ensemble) | No | 3× NVIDIA A6000 + RTX 4090 personal + vast.ai rented | ~10 h / model |
| 2nd | RTX23090 (tattaka+mipypf+yukke42) | 5 timm 2D backbones × 2 z-config (resnetrs50, convnext_tiny, swinv2_tiny, swin_small, ecaresnet26t, resnext50d) + irCSN-50 + irCSN-152 + yukke r18/r34/MViTv2 (≈16–79M each) | **192 / 224 / 256** per backbone | 24–36 central (per config) | **18 models** (9 fold0 + 9 fold1) sigmoid-mean | No (per-backbone fixed) | No | `--gpus 4` Kaggle Docker; specific GPU model not disclosed | not disclosed (≥55 model-trainings × 30 epochs) |
| 3rd | traptinblur | ir-CSN-50 + ir-CSN-152 (Kinetics, mmaction2) + 2D U-Net decoder (≈25M + 33M) | **224 / 384 / 576** multi-resolution | central **28**, sliding-window 24 of 28 (z-jitter offsets [0,2,4]) | **22 models** (8+6+8 across the 3 resolutions) | **Yes — 224/384/576** | No | 1× RTX 3090 24GB or A100 40GB; 4× of either for 576-res DDP | not disclosed |
| 4th | POSCO DX (ahnheeyoung1) | 3D-ResNet-152 + 3D-ResNet-200 + 3D-ResNeXt-101 (Kinetics) + 2D U-Net decoder | **256** single-resolution | central **22** (21–42) | **12** models (3 backbones × 4 folds) softmax-mean | No (only 256 in public artifacts) | No | not disclosed (CUDA only) | not disclosed (3 dated checkpoint sets) |
| 5th | aksell | 3D-ResNet-18 + 3D-ResNet-34 (Med3D / Kinetics-pretrained) + 2D U-Net decoder (≈33M + 63M) | **256** (train) / **1024** sliding-window (infer) | central **32** (16–47) | **6** models (4× R34 all folds + 2× R18 folds 1/2a) simple mean | No | No | 1× NVIDIA RTX 3090 (single GPU) | not disclosed (50 epochs × 6 models) |

**Single most striking convergence**: 5/5 use a **3D-aware front-end + 2D decoder** (depth-mean-pooled or max-pooled). 0/5 use 2D-with-z-as-channels. **Single most striking divergence**: ensemble size ranges 6 → 22 (3.7×), and only the 3rd-place team trained at multiple resolutions.

## Detailed Per-Solution Findings

### 1st — ryches (Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol) — private LB 0.682943

**Primary sources:**
- Kaggle writeup (URL resolves but Kaggle SPA renders client-side — body inaccessible): https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/writeups/ryches-1st-place-solution
- GitHub: https://github.com/ainatersol/Vesuvius-InkDetection (full source; verbatim README)
- YouTube talk: https://www.youtube.com/watch?v=IWySc8s00P0
- Slides: https://docs.google.com/presentation/d/1YZ_KfjSm9tbzXisOwVBalJ-WVlvOF65iugZa73I8MuI

**Architecture** — README, verbatim:
> "Our team used a two-stage model for this challenge. 1. The first stage involved training 3D models such as 3D CNNs, 3D UNETs, and UNETR. These models outputted a 3D volume with multiple channels, which were then flattened along the depth axis. 2. The second stage involved feeding this flattened output to a robust 2D segmentation model (SegFormer with different backbones) that was invariant to depth."

Stage-1 stems (in `externals/models.py`): 4-layer Conv3d (1→4→8→16→32 ch), or `pytorch-3dunet UNet3D`, or MONAI `UNETR(in_channels=1, out_channels=32, img_size=(16,512,512), conv_block=True)`. Depth collapse via `.max(axis=2)` in all variants. Stage-2: SegFormer mit-b3/b4/b5 from HuggingFace (`SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=1, num_channels=32, ignore_mismatched_sizes=True)`). Two ConvTranspose2d upscalers (×4 total) match output to input tile size. README: > "the selection of a depth-invariant solution".

**Z-window / input** (verbatim from `additional_findings.md` + `dataloading.py`):
> "Our competition solution only included the middle 16 layers of data"

Code: `mid = 65 // 2; idxs = range(mid - 8, mid + 8)` → slices 24–39 of 65. Tile sizes mixed: `tile_size = 1024` for SegFormer-b5 stems and `tile_size = 512` for the UNETR stem. Stride at inference = `tile_size // 4`. No multi-resolution training within a single model — diversity comes from the 1024 + 512 mix at the ensemble level.

**Augmentation** (verbatim from `training.ipynb` `return_augs`):
> `A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=.5), A.RandomBrightnessContrast(p=0.25, brightness_limit=.2, contrast_limit=.2), A.ChannelDropout(channel_drop_range=(1, 2), p=.25), A.ShiftScaleRotate(p=0.25), A.OneOf([GaussNoise, GaussianBlur, MotionBlur], p=0.25), A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25), A.CoarseDropout(max_holes=1, max_width=size*0.05, max_height=size*0.05, mask_fill_value=0, p=0.25)`

`ChannelDropout(channel_drop_range=(1,2), p=0.25)` is the only z-axis-specific augmentation (drops 1–2 of the 16 z-channels). **No mixup, no cutmix, no z-shift, no temporal crop/paste**.

**TTA** (verbatim, `inference_notebook.ipynb`):
> `x=[x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]`

4-way 90° rotation TTA, predictions un-rotated and mean-aggregated. **No flips in TTA path.**

**Loss / optimizer / schedule:**
- Loss: `0.5 * SoftBCEWithLogitsLoss + 0.5 * smp.losses.DiceLoss(mode='binary')`. README: > "AdamW optimizer, dice+bce loss".
- Optimizer: AdamW, lr=1e-4 with `GradualWarmupSchedulerV2(multiplier=10, total_epoch=1)` → `CosineAnnealingLR(eta_min=1e-7)`, weight_decay=1e-6, max_grad_norm=100, AMP enabled.
- Epochs: `epochs = 30`. SWA on most checkpoints with `swa_start = 2`, `SWALR(swa_lr=0.05)`. Best models retrained `_all_train` on all 3 fragments (no held-out) for the final submission.

**Pseudo-labeling / self-distillation:** none.

**Fragment fold strategy:**
- `valid_id = 1` for training (validate on frag1, train on frag2+3); `valid_id = 2` for inference setup.
- **Fragment 2 NOT sub-split** — fragments used whole. Most ensemble members are `_all_train` (no held-out).

**Ensemble** (README verbatim):
> "Our final solution is an ensemble of 9 different models."

The 9 models in submitted `model_tuples`:
1. UNet3D + SegFormer-b5 @ 1024 (per-fold .78)
2. UNet3D + SegFormer-b5 @ 512 (.77)
3. UNet3D + SegFormer-b5 SWA @ 1024 (.78)
4. 3D-CNN + SegFormer-b3 SWA @ 1024 (.78)
5. 3D-CNN + SegFormer-b5 long-train SWA (.77)
6. 3D-CNN-more-filters + SegFormer-b3 64-ch (.78)
7. Jumbo-UNet3D + SegFormer (.79)
8. **UNETR + SegFormer-b5 @ 512 (.82, highest single-model)**
9. UNETR + Multiclass-SegFormer-b5 @ 512

Combination (verbatim, `inference_notebook.ipynb`): per-tile **multiplicative blend with `pow(1/mask_count)` geometric averaging within tiles**, then per-model 4-way rot90 TTA → sigmoid → unweighted mean over 9 models. Diversity sources: stem family × SegFormer depth × stem width × tile size × single-class vs 3-class head × SWA vs non-SWA.

**Post-processing** (verbatim):
- Fixed threshold `TH = 0.5`.
- Multiply by binary papyrus `mask.png`.
- `post_process(mask_pred, TH, 10000)` — components < 10000 px dropped.
- Standard Kaggle RLE (`https://www.kaggle.com/stainsby/fast-tested-rle`).
- **No percentile threshold, no per-fragment threshold sweep.**

**Hardware** (README verbatim):
> "We trained on 3 A6000 GPUs ... each model taking roughly 10 hours to train."
> "We also did several runs on smaller systems, mostly RTX4090."
> "We used vast.ai to rent the compute as well as personal ressources."

**Notable**: Multiclass auxiliary head (`{0:"b", 1:"papyrus", 2:"ink"}`) on one ensemble member as additional diversity. Geometric-mean tile blending replaces additive overlap-add. Post-comp note: the 9-model ensemble did **not transfer** to the 8 µm "monster" scroll fragment — > "All of them seemed to return something that looked mostly like noise" — indicating the 16-z-slice + 4 µm regime is tightly fit.

### 2nd — RTX23090 (tattaka + mipypf + yukke42) — private LB 0.681739

**Primary sources:**
- Kaggle writeup (SPA, title only): https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/writeups/rtx23090-2nd-place-solution
- Speaker Deck (mipypf, primary technical reference): https://speakerdeck.com/mipypf/kaggle-vesuvius-challenge-ink-detection-2nd-solution
- GitHub (`winner_call` branch with full code): https://github.com/mipypf/Vesuvius-Challenge/tree/winner_call
- Mirror: https://github.com/yukke42/kaggle-vesuvius-challenge-ink-detection

**Architecture** — 2.5D encoder (timm 2D backbone applied to z-slice groups) + 3D-CSN block + 1×1 head producing logits at **1/32 resolution**, bilinear-upsampled. Verbatim header of `tattaka_ron/src/exp055/train.py`:
> `2.5D + 3DCSN + no_mixup_epochs + channel shift + stochastic depth + mid_layer option + groups option + manifold mixup + cutmix + normal mixup + shapemarker + ema + label smoothing + fix seed + heavy aug`

9 backbone variants × 5-fold via `train.sh`: `resnetrs50` × 2 z-configs (5×7, 3×9), `convnext_tiny.fb_in1k` × 2, `swinv2_tiny_window8_256.ms_in1k` × 2, `swin_small_patch4_window7_224.ms_in22k_ft_in1k` × 2, `ecaresnet26t.ra2_in1k` × 2 (3×12, 2×15), `resnext50d_32x4d` (15-epoch). Plus 3D-only encoders (exp056): `resnet3d50csnir1x32_mixup_ep30`, `resnet3d152csnir1x24_mixup_ep30`. Yukke pipeline adds 3D-ResNet-18/34 + MViTv2-s.

**Z-window / input:**
- Tile sizes per backbone (verbatim `train.sh`): 256 (resnetrs50/convnext_tiny/swinv2_tiny/ecaresnet26t/resnext50d), 224 (swin_small), 192 (3D-CSN).
- Z-slice ranges (verbatim `--start_z` / `--end_z`):
  - `--start_z 15 --end_z -15` → 35 central slices = 5 groups × 7 channels
  - `--start_z 19 --end_z -19` → 27 central = 3 × 9
  - `--start_z 14 --end_z -15` → 36 central = 3 × 12
  - `--start_z 17 --end_z -18` → 30 central = 2 × 15
  - irCSN-50: 32 central; irCSN-152: 24 central
- **Multi-resolution: NO sweep.** Each backbone trains at one fixed size; cross-backbone variation gives a 192/224/256 mix at the ensemble level.

**Augmentation** (verbatim `train.py`):
- **Manifold mixup CONFIRMED** (lines 750–760): `if np.random.rand() < 0.5: img_feat = self.mixupper.lam * img_feat + (1 - self.mixupper.lam) * img_feat.flip(0)` — mixed 50/50 with input-space mixup. Flags `--mixup_p 0.5 --mixup_alpha 0.2`.
- **Cutmix CONFIRMED** (lines 762–777, `rand_bbox`-based).
- **Channel-shuffle CONFIRMED** (lines 519–526): `ch_arr = [random.sample(ch_arr, len(ch_arr)) if np.random.rand() < 0.2 else ch_arr for _ in range(img.shape[0])]` — z-group axis permuted with **p = 0.2**.
- **Z-shift ±2 CONFIRMED**: flag `--shift_z 2`; code: `if self.train and np.random.rand() < 0.5: shift = np.random.randint(-self.shift_z, self.shift_z + 1)` — applied with **p = 0.5**.
- 2D Albumentations: `Flip(p=0.5)`, `RandomRotate90(p=0.9)`, `ShiftScaleRotate(0.0625/0.2/15, p=0.9)`, `OneOf([ElasticTransform, GaussianBlur, GaussNoise, OpticalDistortion, GridDistortion, PiecewiseAffine], p=0.9)`, `RandomBrightnessContrast(0.3, p=0.3)`.
- Grid cutout `max_height=2, max_width=2, prob=0.5` on 32-pixel grid.

**TTA** (`eval_tta_fp16.py` lines 163–168): **4× TTA** = `["vanilla", "flip_v", "flip_h", "flip_vh"]`. fp16. Stride 32.

**Loss / optimizer / schedule:**
- Loss: `nn.BCEWithLogitsLoss() + FBetaLoss(smooth=1e-6, beta=0.5)` summed (`losses["loss"] = losses["bce"] + losses["fbeta"]`). Label smoothing 0.1: `true = true * (1 - 0.1/0.5) + 0.1`.
- EMA: `ModelEmaV2(decay=0.99)`. Optimizer: AdamW, weight_decay=1e-4. Schedule: cosine + 2% warmup.
- LR per backbone: 1e-3 (resnetrs50/convnext/ecaresnet26t/3D-CSN), 2e-4 (swinv2/swin_small), 0.25e-3 (resnext50d). 3D-CSN uses `--backbone_lr 2e-4` separately.
- Epochs: 30 for all primary; 15 for resnext50d only.

**Pseudo-labeling**: none.

**Fragment fold strategy:**
- Fragment 2 split into **3 vertical strips** named 2/4/5, each ≈1/3 inked-mask pixels (`input/split_fragment2.py` lines 22–35). Final layout: original fragments {1, 3} plus split {2, 4, 5} = **5-fold GroupKFold**. Yukke part used 4 folds (vertical 2-way split of fragment 2).

**Ensemble:**
- **Final: 18 models** (Speaker Deck slide 27 verbatim): "18個のモデル (fold0：9個、fold1：9個)の 平均アンサンブル" = 9 fold0 + 9 fold1. Tested up to 20 (OOM), finalized at 18.
- Combination: probability sigmoid-mean → percentile threshold within papyrus mask.

**Post-processing:**
- **Percentile threshold within mask CONFIRMED** (`eval_tta_fp16.py` lines 235–252; `train.py` lines 593–613): `p_valid *= fragment_mask; threshold = find_threshold_percentile(...); p_valid > np.quantile(p_valid_tmp, threshold)`. Per-fold OOF thresholds via Nelder–Mead optimization on F0.5 land in **0.87–0.93**.
- **Final submission percentile = 0.93** (slide 19 verbatim: "0.9、0.93を使用" with 0.93 chosen on private LB). The seed-appendix recorded "~87th percentile" is per-fold OOF lower bound, not the actual submitted value.
- No min-size filter; standard Kaggle RLE.

**Hardware** (verbatim training flags): `--gpus 4` for almost all configs; `--gpus 1` only for resnext50d-15ep. The team handle "RTX23090" is a pun (RTX 2 × 3090 = 4090) and **not literal hardware**: actual training environment is the Kaggle Docker `gcr.io/kaggle-gpu-images/python:v128` (verbatim `tattaka_ron/README.md` §1) on 4-GPU machines; specific GPU model not disclosed.

**Notable**:
1. **Predict at 1/32 resolution.** Output stride 32 + bilinear upsample; team observed Dice 0.98 vs ground-truth-down-sampled-to-1/32 (Speaker Deck) — buys throughput at almost no Dice loss.
2. **Manifold-vs-input mixup 50/50 mode-switch** instead of always one or the other.
3. **Two complementary z-parameterizations per backbone**: (5×7) vs (3×9) splits of central slices diversify 3D inductive bias inside one backbone family.
4. **Mean+max pooling along the 3D group axis** (`torch.cat([img_feat.mean(2), img_feat.max(2)[0]], 1)` line 537) — doubles channel width to 1024 before head.
5. Ablation negatives (Speaker Deck "What didn't work"): EfficientNet, MaxViT, RepVGG; SGD/Lion; large crop sizes; SWA. **SWA explicitly didn't work for this team** — opposite of rank 1.

### 3rd — traptinblur — private LB 0.679859

**Primary sources:**
- Kaggle writeup (SPA-locked, no Wayback snapshot): https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417536. Body retrieved via third-party mirror: https://kaggle.curtischong.me/competitions/Vesuvius-Challenge---Ink-Detection
- GitHub: https://github.com/traptinblur/VCID_2023_3rd_place_code (README, configs/*, hybrid_unet2.py, reproduce.sh)
- Inference notebook: https://www.kaggle.com/code/traptinblur/3rd-place-ensemble-576-8-384-6-224-8/notebook

**Architecture** — writeup verbatim:
> "only uses ir-CSN as the 3d encoder with a simple mean pooling bridging to 2d decoder"

Backbones: `adaptive_silu` (ResNet50 ir-CSN at 224 and 384) and `adaptive_silu_r152` (ResNet152 ir-CSN at 576). `hybrid_unet2.py` sets `bottleneck_mode='ir'` on `ResNet3dCSN(depth=int(encoder[1:-2]))`. Pretrained weights: mmaction2 Kinetics CSN checkpoints (`download_pretrained_weights.sh`). Bridge: `enc_results = [torch.mean(f, dim=2) for f in enc_results]` collapses depth before 2D U-Net decoder. Decoder filters `[40, 64, 128, 256]`. Writeup: > "their decoder was inspired by https://github.com/selimsef/xview3_solution".

**Z-window / input:**
- Tile sizes (README table): 224/112, 384/128, 576/144.
- Z: `in_chans = 28; mid = 65 // 2; slice_idcs = range(mid-14, mid+14)` → central 28 of 65, sliding window 24 of 28 per forward via `tr_idx = [0, 2, 4]`. Writeup: > "experimented 8*i slices where i=1, 2, 3, 4, 6. 24, 32 or 48 these all worked well. They chose to use 24 slices".
- **Multi-resolution 224/384/576 CONFIRMED** (README table + inference notebook title `3rd_place_ensemble_576-8_384-6_224-8`). Writeup: > "(3rd) Many U-nets with small resolution and Split up fragment 2 to have more training data".
- Best single model writeup verbatim: > "The best single big resolution model is 576x576 patch size r152 type unet on fold 5, scoring 0.77 locally".

**Augmentation:**
- **Mixup + Cutmix CONFIRMED** (writeup verbatim "augmentation used mixup and cutmix We comment out cutout when use cutmix"). README table column `mixup/switch2cutmix`: 5 of 7 model rows use **0.6/0.84** (`p_mixup=0.6, p_switch=0.84`); 2 outliers use 0.1/0.0 and 0.4/0.5. **The seed-appendix "T3 cutmix p=0.6/0.84" is the dominant config but not universal.**
- Z-axis-specific: only the `tr_idx = [0, 2, 4]` adaptive temporal sliding window inside the encoder (implicit z-jitter ensemble for free). No z-shift, no channel-shuffle, no temporal crop+paste.
- 2D (`base_config.py`): `HorizontalFlip p=0.75, VerticalFlip p=0.75, RandomBrightnessContrast p=0.75, ShiftScaleRotate(rotate_limit=180, p=0.75), OneOf{GaussNoise[10,50], GaussianBlur, MotionBlur} p=0.5, GridDistortion(0.3, p=0.5)`. Cutout commented out. Writeup: > "We set the ShiftScaleRotate's rotate_limit to 180".

**TTA** — writeup verbatim: > "no test time augmentation (tta)". **The seed-appendix "T3 dropped TTA entirely" is CONFIRMED.**

**Loss / optimizer / schedule:**
- Loss: BCE + Dice with row-specific weights: 6/1 (224 fold-1, 224 fold-5), 1/0 pure-BCE (224 fold-2/3, 224 fold-4, 384-all, 576 fold-1/4), 3/1 (576 fold-2/3/5). `base_config.py: loss_weights = [1, 0]`.
- Optimizer: AdamW, OneCycle cosine. `max_lr=1.5e-4, init_lr=1.5e-5, warmup_epoch=1, weight_decay=1e-6, max_grad_norm=1000`.
- Epochs: 30 (224/384), 20 (576). Effective batch size 16 throughout (224/384 bs=16 acc=1, 576 bs=2 acc=2 with 4-GPU DDP).
- EMA: enabled all rows, decay 0.997–0.9998 per row.

**Pseudo-labeling**: not disclosed in writeup, README, or "didn't work" list — neither used nor rejected.

**Fragment fold strategy:** Writeup verbatim: > "originally used all 3 scroll fragments as 3 folds but fold 2 scored the worse … so they split it into 3 individual folds (now they have 5 folds)". `fragment2_split.ipynb` "will rename folder 2 to folder 6 then splitting fragment 2 to 3 parts". 5-fold leave-one-out across {1, 3, plus three sub-folders from old frag 2, plus 6 = old frag 3}.

**Ensemble:**
- **22 members** = 8 + 6 + 8 (notebook title `576-8 / 384-6 / 224-8`).
- Combination method: not disclosed in any retrieved source (notebook body SPA-locked).
- Diversity: 3 input resolutions × 2 encoder depths × 7 distinct loss/EMA/mixup combos × per-fold checkpoints.

**Post-processing:**
- **Threshold fixed at 0.5** — writeup verbatim: > "DIDN'T USE THRESHOLD SELECTION - just used 0.5".
- Min-size filter: not disclosed.
- Inference uses smaller stride than training: > "used smaller cropping stride comparing to which we use in training. I guess it's more accurate".
- "Ignore edge of output prediction" listed as competition-wide regularizer; team-specific attribution unclear.

**Hardware** (README "Hardware Requirements"): > "One of: Nvidia RTX 3090 with 24GB VRAM / Nvidia A100 with 40GB VRAM. When TRAIN on big resolution you may need one of: 4x Nvidia RTX 3090 with 24GB VRAM / 4x Nvidia A100 with 40GB VRAM". Team's actual machine not disclosed.

**Notable**:
1. **Adaptive temporal cropping inside encoder forward** — `HybridUnet_adpt_seg_cls.forward` slices the 28-slice z-block at offsets [0, 2, 4] per forward = implicit z-jitter ensemble. The `adaptive_silu` model name refers to this, NOT to an activation function.
2. **Per-row loss tuning** — no single configuration won; team mixed BCE-only, BCE+Dice 6/1, BCE+Dice 3/1 across 7 trained models for ensemble diversity.
3. **Splitting fragment 2 into thirds** treated as both a CV fix and a data-augmentation trick.
4. Verbatim "didn't work" list: > "2d models / pure 3d models / stacked unet which use 2d denoiser / training a 3d denoiser / classification branch for stronger supervision / add maxpooling and conv layers between encoder and decoder". Repo retains `HybridUnet_seg_cls` (with cls branch) as dead code, evidence the cls-branch ablation was actually run.
5. **Threshold deliberately fixed at 0.5** — unusual for a precision-weighted F0.5 metric; team chose simplicity over tuning gains.

### 4th — POSCO DX (Heeyoung Ahn) — private LB 0.673941

**Primary sources:**
- Kaggle writeup (SPA-locked, body unrecoverable, no Wayback snapshot): https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/writeups/posco-dx-heeyoung-ahn-4th-place-solution
- Inference kernel (last run 2023-07-14): https://www.kaggle.com/code/ahnheeyoung1/ink-detection-inference
- Training kernel (last run 2023-06-19): https://www.kaggle.com/code/ahnheeyoung1/ink-detection-training
- Pretrained-weight datasets: `ahnheeyoung1/2023-06-05-12-22-clip-50-200`, `…-resnext101`, `…-resnet200`, `…/inchans16-movineta5-longepochs-21252428`, `…/weights`.
- **No GitHub repo for this competition by this author.**

All quotes below are verbatim from notebook source cells (Kaggle API, retrieved 2026-04-26).

**Architecture** — three 3D-CNN encoders + one 3D video model:
- 3D-ResNet-152: `generate_model(model_depth=152, n_input_channels=1, no_max_pool=True)`, pretrained `r3d152_KM_200ep.pth`.
- 3D-ResNet-200: `SegModel(200)`, pretrained `r3d200_KM_200ep.pth`.
- 3D-ResNeXt-101: `resnext101(sample_size=112, sample_duration=16, shortcut_type='B', cardinality=32)`, pretrained `kinetics_resnext_101_RGB_16_best.pth`.
- MoViNet-A5: `MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=False)` with conv1 replaced. **Disabled in 0.79-LB final submission** (`model_movinet = False`).
- Decoder: `Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)` with final `nn.Conv2d(256, 2, 1, 1, 0)` (2-class CE head). Z collapse: `feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]` before decoder.

**Z-window / input:**
- Tile: `image_size=256` (training arg default + inference). Stride: `tile_size // 3 ≈ 85px` train, `tile_size // 5 ≈ 51px` inference.
- Z-slices: `in_chans=22`; `mid = 65 // 2; idxs = range(mid-11, mid+11)` → central 22 (slices 21–42).
- Notebook header "## summary * use only 6 slices" is **stale** — `in_chans=22` everywhere.
- Pixel clipping: `image = np.clip(image, 50, 200)` (`clipping=True; clip_min=50; clip_max=200`).
- **Multi-resolution: NOT verified.** No 512/1024 references in any public artifact; only `image_size=256`. **The seed-appendix "T4 used multi-resolution 512/1024" is unsupported by published code.**

**Augmentation:**
- **Cutmix CONFIRMED**: `cutmix=True; beta=1; if cutmix and random.random()>0.4: lam = np.random.beta(beta, beta); …` → cutmix probability 0.6 (the `>0.4` condition).
- **Temporal crop+paste+cutout CONFIRMED** verbatim from training kernel `__getitem__`:
  > `cropping_num = random.randint(args.cropping_num_min, args.cropping_num_max)`
  > `start_idx = random.randint(0, args.in_chans - cropping_num)`
  > `image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]`
  > `if random.random() > 0.4: image_tmp[..., temporal_random_cutout_idx] = 0`
  
  Defaults `cropping_num_min=12, cropping_num_max=22`. **The seed-appendix "T4 uses temporal crop+paste+cutout" is CONFIRMED.**
- 2D: `HorizontalFlip(p=0.6), VerticalFlip(p=0.6), RandomGamma((50,150), p=0.6), RandomBrightnessContrast(0.2, p=0.6), ShiftScaleRotate(0.1/0.1/360, interpolation=0, border_mode=0, p=0.6), OneOf([GaussNoise(10,30), GaussianBlur], p=0.6), CoarseDropout(max_holes=4, max_width=size*0.2, p=0.6)`. No mixup.

**TTA** — final 0.79-LB submission verbatim: `flip_test = False`, `rot_test = False`, `tta_temporal = False`. **The seed-appendix "T4 dropped TTA entirely" is CONFIRMED for the recorded best run** (TTA hooks are coded and toggleable but disabled).

**Loss / optimizer / schedule:**
- Loss: 2-class **CrossEntropy + label smoothing 0.3** (verbatim `args.ls=0.3`). Dice/Tversky/Focal-Tversky/F-beta defined but commented out — pragmatic CE-only choice differs from all other top-5 teams.
- Optimizer: `AdamW(lr=0)`, `weight_decay=1e-6`, `max_grad_norm=1000`.
- Schedule: `CosineAnnealingWarmUpRestarts(T_0=9, T_mult=1, eta_max=1e-4, T_up=3, gamma=0.5)` — cyclic with 3-epoch warmup, 9-epoch period, 0.5 amplitude decay per cycle.
- Epochs: `args.epochs=1` (CLI default) but `# 30` comment in source, validation only at `epoch >= 20` → 21–30 epochs. AMP enabled.

**Pseudo-labeling**: none.

**Fragment fold strategy:**
- Fragment 2 split column-wise at x=4300 into pseudo-fragments 2-left (id 2) and 2-right (id 4). Verbatim: `if fragment_id == 2: image, mask = read_image_mask(2); image, mask = image[:, :args.slicing_num, :]; elif fragment_id == 4: image = read_image_mask(2); image = image[:, args.slicing_num:, :]` with `args.slicing_num = 4300`. **4 folds: 1, 2-left, 3, 2-right.**
- Training-tile gate: only fully-in-papyrus tiles kept (`if mask_frag[y1:y2, x1:x2].min() == 1`).

**Ensemble:**
- **0.79-LB final = 12 models** (3 backbones × 4 folds): `model_resnet152=True, model_resnet200=True, model_resnext=True, model_movinet=False`.
- Combination: per-model softmax → channel-1 prob × backbone weight → mean across models. Weights all 1.
- Run-ledger comment block (verbatim): > `# LB Best 0.79 4th -> '/kaggle/input/2023-06-05-12-22-clip-50-200' & '/kaggle/input/2023-06-07-12-22-clip50-200-ls03-resnext101' & '/kaggle/input/2023-06-08-12-22-clip50-200-ls03-resnet200' stride 5 no remove`.

**Post-processing:**
- Threshold fixed at **0.47** (top-of-file constant). Validation sweep `np.array(range(10, 95+1, 5))/100` only used for fold metrics, not submission.
- Min-size: `remove_small_objects` coded with `degree_rate=0.0001` → ≈ `(H*W)*0.0001` px threshold; **the 0.79 ledger note "stride 5 no remove" indicates final submission DISABLED min-size removal**.
- Mask gated by binary papyrus mask before RLE.
- Optional `denoise_image(iter_num=50)` TVL1 denoise coded but `denoise=False` in final.

**Training time / hardware**: not disclosed (CUDA only; no GPU type/count). Saved weight folders dated 2023-06-05/07/08 — three backbone-set training days.

**Notable**:
1. **3D Kinetics video models repurposed as ink-segmentation encoders** — depth/cardinality diversity (152 / 200 / 101-cardinality-32) over the same 2D decoder.
2. **Conv1 surgery** for single-channel z-stack: 3D-ResNet path `conv1_weight = conv1_weight.sum(dim=1, keepdim=True)`; 3D-ResNeXt path replaces conv1 entirely with `Conv3d(1, 64, kernel_size=(7,7,7), stride=(1,2,2), padding=(3,3,3))`.
3. **Temporal random crop+paste+cutout** (rare in this competition's top-5) — randomly take 12–22 contiguous z-slices, paste at random Z offset, then with p=0.6 zero-out a sub-block.
4. **Fragment 2 split at x=4300** to fabricate a 4-fold leave-one-out.
5. **2-class CE + label smoothing 0.3** instead of BCE/Dice — only top-5 team to use this.
6. Cyclic LR `CosineAnnealingWarmUpRestarts(T_0=9, T_mult=1, eta_max=1e-4, T_up=3, gamma=0.5)`.

### 5th — aksell — private LB 0.673795

**Primary sources:**
- GitHub: https://github.com/aksell1981/inkdet_solution (README header "**Vesuvius Challenge - Ink Detection 5th place solution**", full training code)
- Kaggle inference notebook: https://www.kaggle.com/code/aksell7/3dresnet18-3dresnet34-infer
- Kaggle writeup (SPA-locked, no Wayback): https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417642 (5th-place attribution confirmed via https://scrollprize.substack.com/p/12000-ink-detection-followup-prize)

**Architecture** — Two backbones: **3D-ResNet18** (`generate_model(model_depth=18, n_input_channels=1)`) and **3D-ResNet34** (`generate_model(model_depth=34, n_input_channels=1)`), Med3D-style Kinetics-pretrained (`r3d18_K_200ep.pth`, `r3d34_K_200ep.pth`). Decoder verbatim from `SegModel.forward` (lines 357–360):
> `feat_maps = self.encoder(x); feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]; pred_mask = self.decoder(feat_maps_pooled)`

The `Decoder` (lines 325–347) is `nn.Conv2d / nn.BatchNorm2d / F.interpolate` 2× upsample blocks → final `nn.Upsample(scale_factor=4)`. **The seed-appendix "T5 2D-UNet decoder" is CONFIRMED verbatim** from source code.

**Z-window / input:**
- Tile: 256×256 (`make_tiles.py: tile_size = 256`).
- Z: central **32 of 65**, slices 16–47 inclusive (`Z_START=16; Z_DIMS=32; idxs=range(16, 48)`).
- Multi-resolution: training fixed 256×256. Inference upsamples: `size=896, expand=128, tile_size=1024, stride=512` (overlap-add with center-crop accumulation).

**Augmentation:**
- **No mixup, no cutmix.** Only sample-mixing aug is **YOLO-style 4-tile mosaic**: p=0.5 in r34/r18-fold1 scripts, p=0.75 + center-crop in r18-fold2a. Code: `if self.mode=='train' and random.randint(0,1): img,mask,yx = load_mosaic(self,idx)`.
- **No z-axis-specific augs.** All other augs operate in 2D over the 32-slice z-stack treated as image channels.
- 2D: `HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5), RandomBrightnessContrast(p=0.75), ShiftScaleRotate(0.0625/0.2/15, p=0.9), OneOf([GaussNoise(10,50), GaussianBlur, MotionBlur], p=0.4), GridDistortion(0.3, p=0.5)`.

**TTA** (notebook): **6-way** = 4 rotations (`torch.rot90 k=0,1,2,3`) + 2 flips (`torch.flip dims=[3]`, `dims=[4]`).

**Loss / optimizer / schedule:**
- **Plain BCE only** (`criterion = SoftBCEWithLogitsLoss`); Dice/Tversky imported but unused.
- AdamW, lr=1e-4 (warmup base 1e-5), weight_decay=1e-6, max_grad_norm=1000, AMP.
- `GradualWarmupSchedulerV2(multiplier=10, total_epoch=1)` → `CosineAnnealingLR(T_max=epochs, eta_min=1e-7)`.
- `epochs = 50`, `train_batch_size = 16`.

**Pseudo-labeling**: none.

**Fragment fold strategy:**
- **Fragment 2 split horizontally at row 8300** into 2a (rows 0–8299) and 2b (8300–end). `split_fragment.py: h = 8300; tifa=tif[0:h,:]; tifb=tif[h:,:]`.
- 4-fold leave-one-out across `all_ids = ['1', '2a', '2b', '3']`. r34 trains all 4 folds; r18 only fold-1 and fold-2a.

**Ensemble:**
- **6 members**: `chkps = ['Resnet18_3d_fold1', 'Resnet18_3d_fold2a', 'Resnet34_3d_fold2b', 'Resnet34_3d_fold1', 'Resnet34_3d_fold3', 'Resnet34_3d_fold2a']` (4× R34 covering all folds + 2× R18 covering folds 1 and 2a).
- Combination: per-model 6-way TTA mean → simple mean across 6 models.

**Post-processing:**
- Threshold fixed at **0.5**.
- **No min-size filter, no morphological ops.** Instead: **CuPy total-variation L1 denoise on soft probability map** (`mask_pred = denoise_image(mask_pred, iter_num=250)` with split-Bregman params `fidelity=150, sparsity_scale=10, continuity_scale=0.5, mu=1`).
- Mask gated by binary papyrus mask before RLE.

**Hardware** (verbatim README): > "Ubuntu 20.04.5 LTS / CPU AMD Ryzen 7 3700 / 32GB RAM / 1 x NVIDIA GeForce RTX3090 / Python 3.9.7 / CUDA 11.7".

**Notable**:
1. **YOLO-style 4-tile mosaic** as the only sample-mixing augmentation. Two regimes (p=0.5 vs p=0.75 + center-crop). Mosaic preserves arbitrary z-stack channels — unlike mixup which would bleed labels across an unknown z-axis.
2. **Single-channel weight conversion** for Kinetics-pretrained 3D-ResNet: `state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)`.
3. **CuPy TV-L1 denoise on probabilities** (250 iters split-Bregman) replaces conventional morphology. Operates in Fourier domain.
4. **6-way TTA** (4 rot + 2 flip with explicit inverse before averaging) — most aggressive TTA among top-5.
5. **Smallest ensemble (6) with smallest hardware (1× RTX 3090)** — yet only 0.009 F0.5 below rank 1 (which used 9-model SWA ensemble on 3× A6000). Demonstrates ceiling of single-GPU regime is high if architecture is right.

## Takeaways

The top-5 win on **5 architectural / data-handling axes**, none of which are about per-model capacity. Most of the gap to our local 0.5578 is explained by the first three.

1. **3D-aware encoder is non-negotiable.** 5/5 top teams use 3D conv blocks (whether Kinetics-pretrained 3D-ResNet/CSN/ResNeXt or from-scratch 3D-CNN/UNet3D) followed by depth-mean / depth-max pooling to a 2D decoder. **0/5 use a 2D backbone with z-as-channels.** Our run 20260227 uses precisely the 2D-EfficientNet-z-as-channels pattern that the top 5 unanimously avoid.

2. **Tight central z-window: 16–32 slices of 65, never the full volume.**
   - 1st: central 16 (24–39)
   - 2nd: 24–36 central per config
   - 3rd: 28-slice block, 24-of-28 sliding
   - 4th: central 22 (21–42)
   - 5th: central 32 (16–47) No team uses the full 65-slice volume; none use pure-3D end-to-end without depth pooling.

3. **Fragment 2 sub-splitting fixes the fragment-LOO fold imbalance.** 4/5 split fragment 2:
   - 2nd: vertical 3 strips (2/4/5)
   - 3rd: 3 sub-folds via `fragment2_split.ipynb`
   - 4th: column split at x=4300 (2-left, 2-right)
   - 5th: row split at y=8300 (2a, 2b)
   - 1st: NO split; uses `_all_train` (no held-out at all) Our run 20260420 uses a 50/50 vertical 2a/2b which is the 5th-rank pattern; runs before that used the natural 3-fold-LOO that locally plateaued at F0.5 0.47–0.50.

4. **Ensembles are 6–22 members combined by simple sigmoid mean.** No team uses majority voting, weighted Bayesian model averaging, or stacking. Diversity comes from {backbone family, z-config, tile size}, never from seeds. SWA is load-bearing in 1st but explicitly didn't work for 2nd.

5. **Multi-resolution training (224/384/576 or 1024/512) appears in 2/5** (1st mixes 1024+512 at the ensemble level by training each model at one fixed size; 3rd trains at three resolutions and combines 22 models). 2nd/4th/5th use a single fixed training resolution.

**Anti-patterns confirmed by ≥2 teams:**
- Pseudo-labeling on this task: **NONE of the 5 teams used it**, in stark contrast to TGS-Salt where 4/5 did. The Vesuvius dataset has no separate unlabeled corpus comparable to TGS's 18k test images, so pseudo-labeling does not apply.
- Pure 3D-UNet end-to-end: 3rd verbatim "didn't work" list rejects this; 1st explicitly chose depth-mean pooling for "depth-invariance".
- Non-90° rotation TTA: not used by any top-5 team (all rotation TTA is rot90 only).
- Threshold sweeping: 1st, 3rd, 5th use fixed threshold (0.5); 2nd uses percentile-within-mask (0.93); 4th uses fixed 0.47. **No top team does per-fold threshold optimization at submission time.**

**Compute cost (per documented best):**
- 1st: 9 models × ~10h on 3× A6000 ≈ 270 GPU-hours = **~7× our 24h-of-4-A100 per-run budget**
- 5th: 6 models on 1× RTX 3090 (50 epochs each) ≈ not disclosed but **fits in our budget envelope**
- 3rd: 22 models, single-GPU for 224/384 + 4-GPU DDP for 576 ≈ ≥ 100 GPU-hours

**Gap from our 0.5578 to Kaggle bronze (~0.65) is ~0.09**; to gold (~0.68) is ~0.12. **The single most leveraged change available within our budget is the architectural swap (2D z-as-channels → Kinetics-pretrained 3D front-end + 2D decoder)**, expected delta +0.05 to +0.10 F0.5 based on the unanimous top-5 convergence.

## Appendix: 3D-aware encoder, depth-pooling, percentile threshold, fragment-LOO

### 3D-aware encoder vs z-as-channels — what the top-5 actually do

A "3D-aware encoder" feeds the full `(B, 1, Z, H, W)` volume into Conv3d / 3D-CSN / 3D-ResNet / TimeSformer blocks **before** any pooling along Z. This is qualitatively different from a 2D backbone with z-as-channels (`(B, Z, H, W)` reshaped to `(B*Z, 1, H, W)` or fed as a multi-channel image). The two approaches differ on:

| Dimension | 3D-aware encoder | 2D backbone with z-as-channels |
|---|---|---|
| Inductive bias | Translation-equivariant in (Z, H, W); learns z-axis correlations | No inductive bias on z; treats slices as independent feature channels |
| Pretrain | Kinetics video weights (3D-ResNet, ir-CSN, MViTv2) directly transferable | ImageNet 3-channel; 1st-conv stem must be re-initialized → underfit risk |
| Depth handling | Depth-mean / depth-max pool **after** encoder, into a 2D decoder | Depth implicit in channel concat; no explicit pooling |
| Cost | ~1.3–1.5× FLOPS at the same spatial resolution (Z extra dim) | Cheaper (single 2D forward) but caps performance on 3D-structured signal |

All 5 top teams chose the first column. Our local 2D-EfficientNet-with-z-as-channels approach converges at F0.5 ≈ 0.54 — roughly 0.13 below the Kaggle top-5 cluster (~0.67).

### Depth-mean vs depth-max pooling

After the 3D encoder produces feature volume `(B, C, D', H', W')`, all 5 top-5 teams collapse the depth axis before the 2D decoder:

- 1st: `.max(axis=2)` (max-pool over depth)
- 2nd: mean+max concatenation (`torch.cat([img_feat.mean(2), img_feat.max(2)[0]], 1)`) — doubles channel width
- 3rd: `[torch.mean(f, dim=2) for f in enc_results]` (mean per scale)
- 4th: `[torch.mean(f, dim=2) for f in feat_maps]` (mean per scale)
- 5th: `[torch.mean(f, dim=2) for f in feat_maps]` (mean per scale)

Mean-pooling is the modal choice (4 of 5). 1st uses max, 2nd uses mean+max concat. This pre-decoder collapse is what makes the model "depth-invariant" — it learns z-aware features but discards exact-z-location at the decoder boundary, mirroring the way ink signal can appear at slightly different depths across fragments.

### Percentile threshold within papyrus mask

2nd-place team's submission threshold: > "0.9、0.93を使用" — the threshold is the **0.93-quantile of the predicted probability distribution restricted to the papyrus mask**, not a fixed sigmoid value. Code:
```python
p_valid *= fragment_mask        # restrict to papyrus pixels
threshold = np.quantile(p_valid_tmp, percentile)   # percentile in [0.87, 0.93] across folds
mask = p_valid > threshold
```

Why it helps F0.5 (precision-weighted): F0.5 punishes recall, so a calibration-drift-robust thresholder beats a fixed 0.5. Per-fragment intensity distributions differ; a fixed value of 0.5 shifts effective recall up or down by 10–20% across fragments. A percentile-based threshold (e.g. 87th–93rd within mask) adapts automatically: it keeps the top X% of pixels per fragment regardless of absolute calibration.

The other 4 top-5 teams use fixed thresholds (0.5 in 1st/3rd/5th; 0.47 in 4th). The percentile approach is therefore not unanimous, but it is the unique innovation of the 2nd-place team and is mechanistically attractive specifically for F0.5 + cross-fragment generalization.

### Fragment-LOO sub-splitting — why and how

The official competition data has 3 training fragments with very different sizes and ink distributions. Naïve fragment-LOO 3-fold creates an imbalance: fold-2 (≈largest fragment) dominates and folds 1/3 plateau early. 4 of 5 top teams sub-split fragment 2:

| Team | Sub-split direction | Pieces |
|---|---|---|
| 2nd | vertical 3 strips | 2/4/5 — each ≈1/3 of inked-mask pixels |
| 3rd | 3 sub-folders | 5-fold LOO including the 3 sub-pieces + frags 1 + 3 |
| 4th | column at x=4300 | 2-left, 2-right → 4 folds with frags 1, 3 |
| 5th | row at y=8300 | 2a, 2b → 4 folds with frags 1, 3 |
| 1st | no split | uses `_all_train` (no held-out at all) |

Our run 20260420 uses a 50/50 vertical split similar to 5th. Earlier runs (20260218, 20260227) used natural 3-fold-LOO and plateaued at F0.5 0.47–0.50.

### Ensemble combination — soft averaging, not voting

For binary segmentation, all 5 top-5 teams use **per-pixel sigmoid averaging** (then thresholding) rather than majority voting:

- 1st: per-tile geometric blend → 4-rot TTA mean → sigmoid → unweighted mean over 9 models
- 2nd: 4-flip TTA → sigmoid mean over 18 models → percentile threshold
- 3rd: combination not disclosed (notebook SPA-locked) but ensemble of 22 with no TTA
- 4th: per-model softmax → channel-1 prob × backbone weight → mean over 12 models → fixed threshold 0.47
- 5th: 6-way TTA → simple mean over 6 models → CuPy TV-L1 denoise → fixed threshold 0.5

**No top-5 team uses majority voting, weighted-by-OOF stacking, or rank averaging.** Soft probability averaging preserves calibration, which a 22-vs-18 hard-vote tie would discard.

### Implications for our pipeline

Our current run 20260425_011945 has 6 candidates. If we want to close the gap to the top-5 cluster within the 24h/4-GPU budget, the priority order from the convergence above is:

1. **Make 2 of 6 candidates use Kinetics-pretrained 3D-ResNet or ir-CSN** with depth-mean pool → 2D decoder. (We already have HF 3D backbones available — `scrollprize/timesformer_GP_scroll1` is used in C2, `scrollprize/resnet50_3um_01122024` in C3, but neither is Kinetics-pretrained — they are scroll-CT-domain-pretrained.)
2. **All 6 candidates must operate on a tight central z-window of 16–32 slices** (currently C1 uses 35-channel `z=15–49`, C5 uses 26-frame TimeSformer hard contract).
3. **Fragment 2 sub-split** must be the default fold strategy. Two patterns worth running side-by-side: vertical 3-strip (2nd-rank pattern) and horizontal 50/50 (5th-rank pattern).
4. **Mixup α=0.2 p=0.5 + manifold mixup at 50/50 mode-switch + z-shift ±2 + channel-shuffle p=0.2** should appear in at least 1 candidate (mirrors 2nd-rank exactly; verifies whether its load-bearing reputation transfers to our scoring regime).
5. **Cross-resolution ensemble 224/384/576** in at least 1 candidate (3rd-rank pattern). 1024 is too expensive for our 4h cap — start with 224/384.
6. **Percentile-within-mask threshold** in at least 1 candidate (2nd-rank pattern), to characterize whether the 0.13 calibration gap explanation holds locally.

Anti-patterns to actively avoid (all confirmed by ≥1 top-5 team's explicit "didn't work" list):
- Pure 3D-UNet end-to-end (3rd-rank rejected)
- 2D z-as-channels (5/5 avoid)
- Non-90° rotation TTA (none use)
- Per-fold threshold sweep at submission (none do)
- Pseudo-labeling (none use; no unlabeled corpus exists)

## Research method notes

- All per-rank research dispatched as 5 parallel general-purpose agents (2026-04-26).
- Sources marked "SPA-locked" or "no Wayback snapshot" could not be retrieved without Kaggle login; some writeups were reconstructed from third-party Quartz mirrors (Curtis Chong's `kaggle.curtischong.me` for 3rd place) or from Speaker Deck (mipypf for 2nd).
- Every quoted fact carries: (URL, verbatim quote ≤25 words, date where available).
- When numbers were not explicitly stated, we report "not disclosed" rather than estimating.
- Seed-claim audit summary (vs the appendix paragraph in `memory/tasks/vesuvius-challenge-ink-detection.md` written 2026-04-24):

| Seed claim (T-anonymized) | Real rank | Status |
|---|---|---|
| T1 SegFormer 2D decoder + 3D-CNN front-end | 1 ryches | Confirmed |
| T1 9-model SWA ensemble | 1 | Confirmed |
| T2 manifold-mixup load-bearing | 2 RTX23090 | Confirmed |
| T2 z-shift ±2 + channel-shuffle | 2 | Confirmed (z-shift ±2 p=0.5; channel-shuffle p=0.2) |
| T2 percentile **~87th** within mask | 2 | **Partially confirmed** — per-fold OOF range 0.87–0.93; **submission used 0.93** |
| T3 multi-resolution 224/384/576 | 3 traptinblur | Confirmed |
| T3 cutmix p=0.6/0.84 | 3 | **Partially confirmed** — dominant config (5/7 models); 2 outliers use 0.1/0.0 and 0.4/0.5 |
| T3 dropped TTA entirely | 3 | Confirmed verbatim |
| T4 multi-resolution 512/1024 | 4 POSCO DX | **NOT verified** — only `image_size=256` in public artifacts; if used, lived only in writeup body which is SPA-locked |
| T4 temporal crop+paste+cutout | 4 | Confirmed verbatim |
| T4 dropped TTA entirely | 4 | Confirmed for 0.79-LB best run |
| T5 2D-UNet decoder + 3D front-end with depth pooling | 5 aksell | Confirmed verbatim |
