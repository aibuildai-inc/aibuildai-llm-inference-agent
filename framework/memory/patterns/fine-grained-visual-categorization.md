# Pattern: fine-grained-visual-categorization

## When this applies
- Image classification with many classes (100s to 60K+) where intra-class differences are subtle
- Severe long-tail distributions (singletons, <10 samples/class common)
- Domain-shift between train/test (disjoint locations, distribution drift)
- Metric learning / re-identification settings (whale-categorization)
- Tasks include: dog-breed-identification, herbarium-2020/2021/2022, imet-2020, inaturalist-2019, iwildcam-2019/2020, leaf-classification, whale-categorization

## Proven patterns
- **IN-22K/CLIP/MIM pretraining is mandatory**: Confirmed in herbarium-2020 (ConvNeXt IN-22K reached 0.451), herbarium-2022 (EVA02 MIM-IN22K hit 0.7352), imet-2020 (EVA02 CLIP-MIM 0.6765), inaturalist-2019 (gold 0.108). IN-1K-only candidates consistently eliminated
- **ConvNeXt-Tiny/Small dominates as default backbone**: Won herbarium-2020 (0.451), iwildcam-2019 (0.474 gold), herbarium-2021 (0.377). Best speed/capacity tradeoff under time budgets
- **Class-balanced loss (effective number, β=0.999) + sqrt sampling**: Consistently beat raw inverse-frequency weighting (herbarium-2020/2022, iwildcam-2019/2020)
- **High resolution (336–448px)**: herbarium-2022 used 336px for 0.7352; inaturalist-2019 needed 448px for gold; imet-2020 mandatory 384px
- **Checkpoint resumption + extended training**: herbarium-2022 progressed 0.6583→0.7256 across resumed epochs; herbarium-2020 confirmed 8–10 epochs is sweet spot
- **Aggressive candidate pruning (1–2 survivors)**: herbarium-2020/2022 both won by concentrating budget vs splitting across 3+ candidates
- **Metric learning (triplet+aux-CE, CosFace) for re-ID**: whale-categorization gold 0.5792 vs failed plain CE

## Common pitfalls
- **Large backbones timeout under 1200s coder limits**: ConvNeXt-Base/Large, Swin-Base/Large, EfficientNet-B5 repeatedly killed (herbarium-2020/2021, imet-2020, iwildcam-2020)
- **DataParallel + AMP on huge classifier heads → NaN**: herbarium-2020 (32K classes), herbarium-2021 (64K classes)
- **ArcFace is unstable on long-tail**: whale-categorization collapsed at 0.30–0.38; CosFace with progressive margin (0.25→0.55) is the stable replacement
- **MixUp/CutMix incompatible with metric-learning losses** (whale-categorization)
- **Val-to-test gap from validation overfitting**: herbarium-2020 best val (0.592) gave only 0.395 test; herbarium-2021 LDAM+DRW had 0.49 val but 0.31 test
- **Aggregator/submission bugs cost medals**: whale-categorization lost gold→none from wrong candidate selection
- **Tar/data extraction inline kills runs**: inaturalist-2019 run 1 DNF; pre-extract before agent loop
- **Skipping detector cropping in camera-trap tasks**: iwildcam-2019/2020 — full-image models drop 30+ accuracy points

## Model selection guide
- **<1000 samples, tabular features available** → Logistic Regression stacking (leaf-classification 0.0116)
- **Re-identification / open-set** → ResNet50 + triplet + aux-CE (weight 2.0–2.5) OR CosFace progressive margin
- **Camera-trap / domain-shift** → ConvNeXt-Tiny or Swin-Base on MegaDetector crops
- **Multi-label, 1000s of classes** → EVA02-Base CLIP-MIM + ASL (γ_neg=4, clip=0.05)
- **10K–60K classes long-tail** → ConvNeXt-Tiny/Small IN-22K + class-balanced focal/effective-number loss
- **Tight time budget** → DINOv2/EVA02 frozen features + MLP head (inaturalist-2019: 0.133 in 30min)
- **Avoid**: ViT-Small without rich pretraining, EfficientNet-B0/B3 IN-1K, Swin-Large unless 8h+ budget

## Feature engineering patterns
- **Detector-based cropping** for camera traps (iwildcam-2019/2020): single biggest preprocessing win
- **Progressive resolution** (224→288→384): herbarium-2020, imet-2020, iwildcam-2019
- **Sequence/group aggregation at inference** (iwildcam-2020 seq_id averaging): free 0.4–2% gain
- **Horizontal-flip TTA**: consistent +1–3% across CNN-based runs
- **Per-class threshold optimization** (multi-label): imet-2020 gained +5% F1
- **Contiguous label remapping** required when class IDs have gaps (herbarium-2022)

## Tuning patterns
- **3–6 tuner runs** typical; gains continue past apparent plateaus (herbarium-2020)
- **Cosine LR + warmup + layer-wise decay (0.75–0.9)**: stable across ConvNeXt/Swin/ViT fine-tuning
- **EMA over final 25% of training**: stabilizes late epochs (herbarium-2020, imet-2020)
- **Single-GPU per model**: DataParallel causes 5–8× slowdown on huge classifier heads (herbarium-2021)
- **Size schedule to actual budget**: imet-2020 Swin-V2 Large 12-epoch plan completed only 1.4 epochs and lost to 4-epoch EVA02-Base
- **Don't change architecture mid-training**: stochastic depth changes on resume caused collapse (herbarium-2020)
- **Pre-flight credential + data-access checks**: dog-breed-identification lost 2 runs to Kaggle 401 + dispatch stalls
