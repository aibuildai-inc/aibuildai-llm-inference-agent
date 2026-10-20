# Pattern: text-similarity-matching

## When this applies
- Short text-pair regression/classification with a similarity target (continuous or graded)
- Dataset has repeated entities (anchors, queries, products) that create grouping structure
- Domain-specific vocabulary (patents, medical, legal) where taxonomy/context can be injected
- Metric rewards calibrated continuous output (Pearson, MSE, MAP) over hard classification

## Proven patterns
- **DeBERTa-v3-large as default backbone**: dominant across all 4 us-patent runs; beats RoBERTa-large, ELECTRA-large, bert-for-patents, and bi-encoders as standalones (best single 0.8566 in run 20260224_225105)
- **Cross-encoder format `[anchor SEP target SEP context]`**: beats bi-encoder/Siamese by 3–4 pts (us-patent C4 bi-encoder killed early)
- **Multi-fold OOF ensembling across model families** lifts ~+0.02 reliably (us-patent: 0.8566 single → 0.8753 Gold ensemble in run 20260224_225105); even weak standalones (C4 at 0.2412) contribute when diverse
- **MSE regression loss for continuous similarity**: beats ordinal/CORAL by ~0.03 (us-patent 0.8566 vs 0.8240) even when target has discrete granularity
- **Inject domain taxonomy as natural language**, not raw codes: CPC text expansion gave ELECTRA-large 0.8601 single-fold (us-patent run 20260301_163011)
- **BF16 + stock pooling** outperforms fp32 + custom CLS/ContextPooler heads

## Common pitfalls
- **StratifiedKFold on repeated entities causes catastrophic leakage**: us-patent run 20260219_082403 collapsed from 0.88–0.93 reported val to 0.5854 test — anchors repeating ~45x must use GroupKFold
- **Custom pooling on pretrained DeBERTa**: either fails to learn (val 0.05) or shows huge val/test divergence (0.851 val → 0.231 test)
- **Ordinal/CORAL framing on continuous metrics**: structural mismatch caps Pearson at 0.73–0.82
- **Stale checkpoint resumption inside fold loops**: silent contamination; one-liner diff review insufficient
- **Config drift**: gradient accumulation declared but unimplemented (us-patent run 4) — verify training loop
- **Weighted ensembles can underperform simple averaging** when members are highly correlated (0.98–0.99); us-patent run 1: 0.85995 weighted < 0.86188 best single
- **DeBERTa-v2-xlarge OOM** without gradient checkpointing; not a free scale-up

## Model selection guide
- Default → DeBERTa-v3-large cross-encoder + MSE
- Need diversity for ensemble → add DeBERTa-v3-base, bert-for-patents (or domain BERT), ELECTRA-large
- Avoid first → bi-encoder/Siamese, ordinal heads, custom pooling, RoBERTa-large alone
- Last resort scale-up → DeBERTa-v2-xlarge + gradient checkpointing (unverified)

## Feature engineering patterns
- Concatenate pair + domain context in single sequence: `text_a [SEP] text_b [SEP] context_natural_language`
- Expand domain codes to descriptions before tokenization (CPC code → CPC text)
- Keep continuous targets as float; never bin into ordinal classes for Pearson metrics
- **GroupKFold on the repeating entity** (anchor_id, query_id) is non-negotiable — set up before any modeling

## Tuning patterns
- Backbone choice dominates hyperparameter tuning; switching DeBERTa-v3-large → smaller model cannot be recovered by exotic losses (us-patent ELECTRA/BERT-small plateau ~0.845, never closed 0.012+ gap to bronze)
- Regularization cocktail (AWP + LLRD + multi-sample dropout + EMA) gives diminishing returns past ~0.86
- Budget multi-fold OOF from design phase — single-fold scores (us-patent 0.9045 DeBERTa-v3-Base) are unreliable signal
- R-Drop overhead not justified (RoBERTa-Large 0.8324)
- Verify fold cleanliness with explicit overlap assertions, not visual diff
