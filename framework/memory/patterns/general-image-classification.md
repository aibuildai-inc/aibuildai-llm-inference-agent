# Pattern: general-image-classification

## When this applies
- Binary or multi-class image classification on natural images (e.g., dogs-vs-cats-redux-kernels-edition)
- Medium-sized datasets (~25K images), variable-resolution JPEGs
- LogLoss or similar calibration-sensitive metrics
- Constrained compute budgets (e.g., 20-min CPU training windows)

## Proven patterns
- **Pretrained CNN fine-tuning** is the dominant approach: ResNet50 with frozen conv1→layer2 and fine-tuned layer3→layer4 was the only architecture to consistently survive judge filtering in dogs-vs-cats run 20260207_182214
- **EfficientNet-B0 / EfficientNetV2-S** offer the best speed-accuracy tradeoff for CPU-constrained runs (~5.3M params); designed across multiple dogs-vs-cats runs
- **ImageNet normalization + 224×224 input** is universal and never the source of failures
- **80/20 stratified train/val split** is the consensus baseline
- **Quick-test verification** (import, forward pass, mini-batch loss) before full training catches bugs early
- **BCEWithLogitsLoss with raw logits** (no sigmoid in model head) is the correct binary-classification setup

## Common pitfalls
- **Budget exhaustion before training**: 4 of 5 dogs-vs-cats runs (20260207_003651, _180239, _181310, _182214) ran out of agent calls/tokens during design phase, producing zero scored submissions
- **Designing too many candidates (5)**: Consumes 20–25K tokens on design, leaving nothing for training
- **Double-sigmoid bug**: Model with sigmoid + BCELoss expecting probabilities → incorrect gradients (killed EfficientNetV2-S in run 20260207_182214)
- **scheduler.step() in batch loop instead of epoch loop**: Decays LR 625× too fast; recurring in Candidates 3 and 4
- **Multi-model ensembles under tight CPU budgets**: Consistently killed for exceeding 20-min training cap (estimated 45–300 min)
- **ViT-B/16 on CPU**: Warmup + grad-clip overhead makes it impractical
- **Declaring features without implementing them** (e.g., label smoothing ε=0.1 in config but standard BCE in code)

## Model selection guide
- Tight CPU budget (≤20 min) → **EfficientNet-B0** (smallest, fastest)
- Moderate CPU budget, want robustness → **ResNet50 with partial freezing** (layer3+ trainable)
- GPU available → ConvNeXt-Tiny or EfficientNetV2-S
- Avoid ViT-family on CPU
- Default to **1–2 candidates max**, not 5, to preserve training budget

## Feature engineering patterns
- **Augmentation stack**: RandomResizedCrop + HFlip + ColorJitter + GaussianBlur + RandomGrayscale (validated across dogs-vs-cats runs)
- **Prediction clipping** away from 0/1 is essential for LogLoss metrics to prevent infinite penalties
- **Label smoothing (ε=0.1)** expected to yield 0.01–0.02 LogLoss improvement on calibration metrics (untested but consensus)
- **TTA (horizontal flip averaging)** is cheap and recommended for final submission
- Aspect-preserving resize/pad to 224×224, not naive squash

## Tuning patterns
- **Optimizer**: SGD+momentum is the safe default; AdamW worth trying for transformer heads
- **Scheduler**: ReduceLROnPlateau is robust; CosineAnnealing for fixed-epoch budgets — always call `scheduler.step()` **per epoch**, never per batch
- **Freezing depth**: For ResNet50 on ~25K images, freeze through layer2; fine-tune layer3+ and head
- **Epochs**: 5–10 epochs feasible on CPU for EfficientNet-B0/ResNet50; profile per-epoch wall time before committing
- **Budget allocation rule**: Reserve ≥60% of agent calls for train→eval→submit; cap design phase at ≤40%
