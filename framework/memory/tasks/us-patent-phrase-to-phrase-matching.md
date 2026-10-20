# Task Memory: us-patent-phrase-to-phrase-matching

## Task profile
- Type: Regression (semantic similarity, continuous target in [0,1] with 0.25 granularity)
- Dataset: Short phrase pairs (anchor, target) with CPC patent codes; ~733 unique anchors each repeated ~45x — anchor-level leakage is the dominant pitfall
- Metric: Pearson correlation coefficient (higher = better)
- Medal thresholds: Bronze ~0.8616, Silver ~0.867, Gold ~0.875 (inferred from runs)

## Best result
- Score: 0.8753 | Medal: Gold | Run: 20260224_225105
- Ensemble of DeBERTa-v3-large (C1 5-fold OOF 0.8554, C2 4-fold OOF 0.8517) + bert-for-patents (C4 5-fold OOF 0.8485), MSE loss, multi-fold OOF averaging. Ensemble lifted single-best 0.8566 → 0.8753 (+0.02).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260219_044013 | 0.85995 | none | DeBERTa-v3-large ensemble (regression + ordinal + multi-task), AWP/LLRD/EMA, GroupKFold |
| 20260219_082403 | 0.5854 | ungraded | ELECTRA-large 5-fold BCE; contaminated by StratifiedKFold anchor leakage + stale-checkpoint resumption |
| 20260224_225105 | 0.8753 | Gold | DeBERTa-v3-large dominant, MSE loss, multi-fold OOF ensemble with bert-for-patents |
| 20260301_163011 | 0.86759 | Silver | 10-model ensemble across 4 families; DeBERTa-v3-Base + BF16 + MSE + CPC natural-language context hit 0.9045 single |

## Proven strategies
- **DeBERTa-v3 (large or base) is the default backbone.** Consistently dominant across all 4 runs; beats RoBERTa-large, ELECTRA-large, DeBERTa-v3-base (as C1), bi-encoders, domain bert-for-patents as standalones.
- **MSE regression loss** beats ordinal/CORAL (0.8566+ vs 0.8240) and BCE ordinal (capped at 0.73 due to 0.25 granularity).
- **Multi-fold OOF ensembling lifts ~+0.02** over best single model (0.8566 → 0.8753). Budget folds from design phase.
- **GroupKFold on anchor_id is mandatory**; StratifiedKFold inflates CV and leaks (anchors repeat ~45x).
- **Ensemble diversity across model families** (DeBERTa + ELECTRA + bert-for-patents) — weak standalone candidates (C4 at 0.2412 alone) still add lift.
- **BF16 mixed precision + simple head** outperforms fp32 + exotic pooling.
- **CPC taxonomy injection as natural language context** helps (ELECTRA-large 0.8601 single-fold).

## Failure patterns
- **Anchor leakage via StratifiedKFold**: run 20260219_082403 collapsed to 0.5854 from reported 0.88–0.93 val scores due to fold contamination + stale checkpoint resumption inside fold loops.
- **Custom CLS/ContextPooler manipulation on pretrained DeBERTa**: never learns (val 0.05), or gives divergent val/test (0.851 val vs 0.231 test). Prefer stock pooling.
- **Ordinal/CORAL framing**: structural mismatch with Pearson metric; caps at 0.73–0.82.
- **Bi-encoder/Siamese (C4 in run 1)**: -3 to -4 pts vs cross-encoder; killed early.
- **DeBERTa-v2-xlarge**: OOM; not a practical scale-up path.
- **Config drift**: gradient accumulation declared but not implemented in training loop (run 4) — wasted compute.
- **R-Drop overhead not justified** (0.8324 test on RoBERTa-Large).
- **Single-model plateau at ~0.845** when architecture is suboptimal; exhaustive tuning of ELECTRA/BERT-small could not close 0.012–0.017 gap to bronze.

## Feature engineering playbook
- Concatenate `anchor [SEP] target [SEP] CPC_text` with CPC code expanded to its natural-language description (not raw code).
- Cross-encoder input format (single sequence with [SEP]) consistently beats bi-encoder/Siamese.
- Keep target as continuous float; do NOT bin into ordinal classes.
- GroupKFold(n_splits=5, groups=anchor_id) — non-negotiable.

## Tuning insights
- Backbone choice dominates hyperparameter tuning; DeBERTa-v3-large + standard settings > hand-tuned smaller models with exotic losses.
- Regularization cocktail (AWP + LLRD + multi-sample dropout + EMA) gives diminishing returns past 0.86; won't close architectural gaps.
- Published DeBERTa-v3-large benchmark is 0.8512 (arxiv 2401.06782); reproduction is margin-tight — watch lr schedule and epoch count.
- Weighted ensembles sometimes underperform simple averaging (run 1: 0.85995 ensemble < 0.86188 best single C1).
- Enforce explicit fold-clean checks; one-liner diff review is insufficient for catching checkpoint contamination.

## Open questions
- Does deberta-v2-xlarge (900M) or XLM-RoBERTa close the ceiling past 0.88 if OOM is solved via gradient checkpointing?
- Why did weighted ensemble underperform best single in run 1 — weighting scheme or correlation (0.98–0.99) between members?
- Can DeBERTa-v3-Base's 0.9045 single-fold (run 4) be replicated with proper multi-fold CV? Time budget exhausted before verification.
- Root cause of DeBERTa-v3-large "eval-mode collapse" in run 2 step 1 — preprocessing divergence or dropout/BN handling?
