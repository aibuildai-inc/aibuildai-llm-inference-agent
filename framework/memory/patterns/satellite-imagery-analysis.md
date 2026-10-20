# Pattern: satellite-imagery-analysis

## When this applies
- Small/medium image datasets (1k–17k samples) of overhead/aerial/satellite imagery
- Native low resolution (32×32 to 75×75), often multi-band or non-RGB (SAR, aerial RGB)
- Binary classification with class imbalance or auxiliary metadata
- Tight compute budgets (5-min training, single GPU)

## Proven patterns
- **Keep native resolution, avoid ImageNet upsampling**: aerial-cactus 0.9999 with 32×32 compact ViT vs EfficientNet-B0 32→224 killed by judge
- **Convolutional stem before transformer/patch embedding**: aerial-cactus 0.9986 val AUC with conv stem + 4×4 patches; pure ViT lacks inductive bias on tiny inputs
- **Domain-specific channel engineering**: HH–VV band-math as 3rd channel (statoil 0.18114); flips + 90°/180°/270° rotations because no canonical orientation exists
- **Locked k-fold indices across all base models**: statoil stack OOF 0.17033 — prevents meta-learner leakage on small data
- **Diverse ensemble (DL + GBDT) beats single model**: statoil stacked 0.17 vs solo 0.20+; even a weak LightGBM (OOF 0.64) lifts blend via calibration
- **3–4 candidates is the sweet spot**: aerial-cactus run 193628 (3 cands) hit 0.9999; 6-candidate runs exhausted agent budgets

## Common pitfalls
- **Pseudo-labeling on satellite test sets**: statoil C5 killed — machine-generated distractors poison labels
- **Upsampling tiny images to 224×224**: wastes compute, blurs features (aerial-cactus EfficientNet-B0 failure)
- **Pure ViT without conv stem on 32×32**: only 64 tokens, insufficient diversity
- **Wide/deep CNNs on <2k samples**: statoil OOM + overfit on 1.3k
- **Misconfigured focal loss α**: aerial-cactus needs α=0.75 (majority-weighted, 3:1 imbalance); α=0.25 hurts
- **Single bad candidate craters score**: aerial-cactus run 193628 candidate 2 hit 0.93 vs 0.9999 siblings (7pp gap)
- **HP tuning plateaus**: statoil stuck at 0.17–0.18 after ~3 iters; further gains require code changes (calibration), not search

## Model selection guide
- Dataset <2k samples → stacked ensemble (small CNN + ResNet18 + LightGBM + multi-task CNN) with locked folds (statoil)
- Dataset 10k–20k, tiny images → compact ViT (~110–156K params) with conv stem (aerial-cactus 0.9986+)
- Multi-band/SAR → add band-math channel + tabular GBDT branch
- Imbalanced binary → focal loss with α matching majority direction, OR weighted BCE as safe fallback
- Always reject: ImageNet-pretrained backbones requiring upsampling, pure ViT on <40×40 inputs

## Feature engineering patterns
- Native-resolution input, dataset-specific normalization (not ImageNet stats)
- Multi-band: include band difference/ratio as extra channel
- Auxiliary metadata (e.g., inc_angle): masked-regression aux head + missing-indicator binary feature beats imputation
- Aggressive aug: Mixup/CutMix (α=0.2–0.4), random erasing, 90° rotations, flips, color jitter
- Stratified splits preserving class ratio (3:1 in aerial-cactus)

## Tuning patterns
- AdamW + cosine annealing + warmup + grad clipping; 15 epochs sufficient under 5-min budget
- Mixup α=0.2–0.4 + TTA (flips + 90° rotations) stabilizes training on small data
- Stochastic depth for ViT on <20k samples
- Meta-learner: L2 logistic regression (C≈40) on OOFs from identical folds
- Post-hoc calibration (isotonic/Platt) is the likely next-gain lever once stack plateaus
- Profile epoch time to <90s, leave margin for inference/submission
