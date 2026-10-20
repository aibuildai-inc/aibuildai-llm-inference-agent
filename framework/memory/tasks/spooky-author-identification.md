# Task Memory: spooky-author-identification

## Task profile
- Type: Multi-class text classification (3-author attribution: EAP/HPL/MWS)
- Dataset: Short prose excerpts; character-level stylistic signals dominate (punctuation, vocabulary, syntax)
- Metric: Multi-class log loss (lower is better)
- Medal thresholds: Bronze ~0.29381; Silver achieved at ≤0.225

## Best result
- Score: 0.22188 | Medal: Silver | Run: 20260301_162627
- Approach: 10-model stacking ensemble combining TF-IDF+LR, 6 RoBERTa-large variants (different seeds/folds), and 2 BERT-large-cased models, with meta-learner trained on out-of-fold predictions via HuggingFace Trainer.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260227_224311 | 0.22503 | Silver | 10-model stack: TF-IDF+LR, 4× RoBERTa-large, 2× DeBERTa-base, 2× DeBERTa-v3-large, 1× DeBERTa-large; 3-way meta-blend + temperature scaling |
| 20260301_162627 | 0.22188 | Silver | Dropped DeBERTa-v3 entirely (unstable); standardized on RoBERTa-large + BERT-large via HF Trainer |

## Proven strategies
- **10-model stacking ensemble** consistently beats best solo model by ~0.11 log loss (0.354 → 0.22x); this is the only reliable path past bronze
- **RoBERTa-large is the workhorse**: best individual architecture, ~0.354 solo; 4–6 seed/fold variants drive stacking diversity
- **TF-IDF + Logistic Regression baseline** (~0.348–0.362) is surprisingly competitive and valuable as an ensemble member (adds non-transformer signal)
- **HuggingFace Trainer** for transformer training: production-grade stability vs. custom loops which repeatedly failed to converge
- **Out-of-fold stacking with meta-learner** (LR stacking or 3-way blend) > simple averaging; temperature scaling helps
- **Seed/fold diversity of the same architecture** provides meaningful independent signal (6 RoBERTa-large variants all contribute)
- **GPU pinning (1 candidate per GPU)** eliminates 2.5–7× epoch slowdowns from contention
- **Byte-level BPE tokenizers** handle punctuation/special chars natively — advantage for stylistic authorship cues

## Failure patterns
- **LightGBM is catastrophically bad** here (0.47–1.08 log loss); kill immediately, don't revisit — stylistic features need deep models
- **Small transformers underperform**: DistilBERT 0.402, DeBERTa-v3-small 0.441 — insufficient capacity regardless of regularization
- **DeBERTa-v3 numerical instability**: crashed/diverged across 15+ attempts in run 2 (v3-small/base/large); worked in run 1 only with heavy babysitting. **Latest guidance: avoid DeBERTa-v3 entirely** unless root-causing precision/dataloader issue
- **Custom training loops** fail to converge reliably — pivot to HF Trainer by default
- **GPU contention** (6 candidates on 4 GPUs) caused 2.5–7× slowdowns and LR-decay-to-zero bugs in round 1
- **CUDA OOM + DataLoader deadlock** on RoBERTa-large 5-fold attempts despite single-fold working

## Feature engineering playbook
- Use byte-level BPE tokenization (RoBERTa/BERT native) — preserves punctuation signal
- No heavy hand-engineered features needed; TF-IDF on raw text is sufficient as classical baseline
- Stack heterogeneous architectures (TF-IDF+LR + RoBERTa-large + BERT-large-cased) for OOF diversity

## Tuning insights
- Model capacity dominates: 300M+ parameter transformers required; small models plateau early
- Diminishing returns beyond 6–7 ensemble members
- Meta-learner choice: LR stacking works; 3-way blend (LR + LR-core + NN) + temperature scaling was best in run 1
- Pin each candidate to a dedicated GPU *before* round 1 full training, not after

## Open questions
- Root cause of DeBERTa-v3 numerical instability (precision? dataloader? env) — could unlock additional ensemble diversity if fixed
- Whether more sophisticated stackers (XGB meta-learner, L2-weighted blending) beat LR stacking
- Optimal ensemble size: 5 vs 10 models — run 2 notes diminishing returns past 6–7
- Path to gold (<0.22188): likely needs either DeBERTa-v3 revival, pseudo-labeling, or external pretrained LM checkpoints
