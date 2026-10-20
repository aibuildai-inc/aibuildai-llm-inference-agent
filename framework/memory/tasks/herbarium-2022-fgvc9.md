# Task Memory: herbarium-2022-fgvc9

## Task profile
- Type: Fine-grained image classification (15,501 classes)
- Dataset: ~665K training images of herbarium specimens; extreme long-tail distribution (min 4 imgs/class, max 64; 7.8% of classes have ≤10 images, 84 classes have ≤5); non-contiguous category IDs (0–15504 with 4 missing, requires contiguous label mapping)
- Metric: Macro F1-score (higher is better)
- Medal thresholds: Bronze ≥ ~0.70 (both runs achieved bronze)

## Best result
- Score: 0.7352 | Medal: Bronze | Run: 20260212_132750
- Approach: EVA02-Small (patch-14, 336px) with MIM ImageNet-22K pretraining, class-balanced weighting via effective number of samples, trained for 10 epochs with checkpoint resumption. Progressive training from epoch 3 (0.6583) to epoch 10 (0.7256 val → 0.7352 test).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260227_105752 | 0.70109 | Bronze | EVA02-Small patch-14, 3 tuner runs, ~5 epochs effective |
| 20260212_132750 | 0.7352 | Bronze | EVA02-Small 336px, 10 full epochs, checkpoint continuation |

## Proven strategies
- **EVA02-Small with MIM pretraining is the dominant architecture**: Won both runs convincingly; MIM IN22K pretraining >> supervised-only IN21K/IN1K for fine-grained botanical features
- **Class-balanced loss (effective number of samples, Cui et al. 2019)**: Essential for extreme long-tail; confirmed across both runs
- **Checkpoint resumption across tuner runs**: Validation F1 improved consistently (0.6583→0.7226→0.7256 in run 1; similar asymptotic gains in run 2); resume-friendly architectures are critical under wall-clock budgets
- **IN21K/IN22K pretraining is mandatory**: IN1K-only pretraining is insufficient for 15.5K-class fine-grained task (Candidate 5 in run 2 auto-failed)
- **Longer training pays off**: 10 epochs (run 1, 0.7352) > ~5 epochs (run 2, 0.7011); model had not plateaued at epoch 10

## Failure patterns
- **Slow models get killed**: Swin-Small (~50 min/epoch), CAFormer-S18 (~46 min/epoch) fail to converge in budget; need ≤~30 min/epoch to get 2.5+ epochs in 90-min sanity checks
- **IN1K-only pretraining fails**: EfficientNet-B3 with IN1K flagged/eliminated in both runs
- **Multi-stage or auxiliary-loss designs add untunable complexity**: ConvNeXt two-stage training and CAFormer genus-auxiliary loss both failed due to hyperparameter sensitivity under tight budgets
- **Focal loss adds complexity without clear gain** over effective number weighting on this task
- **EfficientNet-B3, ConvNeXt-Small, Swin-Small all filtered/killed** before completing training in both runs — never validated as competitive

## Feature engineering playbook
- **Resolution matters**: 336px captures botanical detail better than 224px at modest GPU cost (40GB GPUs); patch-14 (256 tokens at 224px) > patch-16 (196 tokens) for spatial detail
- **Contiguous label remapping required**: Category IDs have gaps (15,505 IDs for 15,501 classes); classifier head needs contiguous 0–15500 mapping
- **Inverse-frequency sampling + effective number weighting**: Both needed for the long-tail; neither alone is sufficient

## Tuning insights
- **Epoch budget is the binding constraint**: Wall-clock limits (~24h total, ~90 min per sanity run) mean only fast-converging models survive; prioritize architectures that show gains by epoch 3–5
- **Checkpoint reuse is the key lever**: Save checkpoints aggressively; regenerate predictions from best rather than retraining
- **Candidate filtering (6–7 → 1–3) early is correct**: Eliminating weak candidates preserves agent-call and tuner budget for the winner
- **Val F1 trajectory 0.66→0.72→0.73 across epochs 3→7→10**: Diminishing but real gains; worth training beyond epoch 10 if budget allows

## Open questions
- **Would 448px or progressive resizing (192→224→336) squeeze more F1?** Higher resolution untested; progressive resizing could fit more epochs early
- **Test-time augmentation (multi-crop, horizontal flip)** not attempted; could push past 0.74 without retraining
- **Ensembling killed candidates**: Predictions from ConvNeXt/Swin/CAFormer were discarded, not ensembled; even weak models may contribute via voting
- **Were ConvNeXt-Small/Swin-Small genuinely worse or just under-trained?** One full training run would validate the filtering decision
- **Auxiliary taxonomy loss (genus/family)**: Failed under short training; may help if model trains >10 epochs
- **EVA02-Base or EVA02-Large**: Only Small variant tested; larger variants may improve if epoch budget is sufficient
