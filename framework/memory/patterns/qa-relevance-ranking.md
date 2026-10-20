# Pattern: qa-relevance-ranking

## When this applies
- Extractive QA tasks with span prediction over long contexts (Wikipedia, multi-paragraph docs)
- Multilingual or domain-specific QA where pretrained QA heads matter
- Tasks with strict compute budgets (~1,200s training/inference timeouts)
- Datasets where most candidate windows/spans contain no answer (severe negative imbalance)

## Proven patterns
- **Two-stage transfer (SQuAD2 → task) is essential**: chaii random QA head scored 0.5566 vs. SQuAD2-pretrained 0.7204; tensorflow2-QA also relied on SQuAD2 initialization for revisions
- **Sliding window with conservative stride**: max_seq_length=384, doc_stride=128 worked for chaii (0.74274 Silver); long-document QA cannot skip this
- **Logit-averaging ensembles at inference**: 5-fold start/end logit averaging gave chaii +0.01–0.02 Jaccard with no retraining cost
- **Backbone size matched to compute budget**: BERT-Base finished in tensorflow2-QA (0.476) where DeBERTa-Large/BigBird timed out; MuRIL-large fit chaii's smaller dataset budget
- **Same-language/same-domain augmentation has highest ROI**: XQuAD Hindi (+1,190) helped chaii reach Silver; cross-lingual proxies (TyDiQA) failed

## Common pitfalls
- **Aggressive positive-only span filtering**: chaii dropped 0.6794 → 0.5529; null-answer windows are needed for SQuAD2-head calibration
- **Per-example TF-IDF at inference**: tensorflow2-QA primary bottleneck; O(n × vocab) on CPU killed inference at 30K test scale
- **Validation-test gap from partial-data training**: tensorflow2-QA fell 0.629 → 0.476 (≈15%); training on 80K/276K with uncalibrated threshold doesn't generalize
- **Heavy backbones on long contexts**: DeBERTa-v3-Large and BigBird never finished 1 epoch in tensorflow2-QA
- **Cross-lingual proxy data without target language**: TyDiQA (no Hindi/Tamil) added zero value to chaii

## Model selection guide
- Small dataset (<10K), multilingual → language-specific large backbone (MuRIL-large for Indic; XLM-R-Large otherwise) with SQuAD2 pretraining
- Large dataset (>100K), long contexts, tight inference budget → BERT-Base with SQuAD2 init
- Avoid DeBERTa-Large / BigBird unless timeout >2× standard budget and sparse attention is verified active
- Always prefer SQuAD2-pretrained checkpoints over vanilla MLM checkpoints for QA heads

## Feature engineering patterns
- Balanced negative subsampling (2–3 negatives per positive) when ~90% of windows are null-answer (chaii proven)
- Oversample minority language/class (4× Tamil in chaii) rather than passive reweighting on small sets
- Precompute retrieval indices (BM25/dense) offline; never refit TF-IDF per inference example
- Stratified k-fold CV by language/domain to prevent split skew with imbalanced subgroups

## Tuning patterns
- LR=2e-5 beats 1e-5 on small QA datasets with pretrained heads (chaii: 0.6794 vs. 0.6656)
- 3–5 epochs optimal for small datasets; 8 epochs gave no gain in chaii
- Layer-wise LR decay 0.9, weight decay 0.01, warmup 10%, batch size 16 are robust defaults
- Calibrate answer/no-answer threshold on held-out split; do NOT reuse train-val threshold (tensorflow2-QA lesson)
- Budget full-data passes (≥4–6 epochs on full set) before ensembling; partial-data training inflates validation metrics
