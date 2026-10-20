# Task Memory: tensorflow2-question-answering

## Task profile
- Type: Extractive question answering (span prediction) on long documents
- Dataset: Natural Questions-style; ~276K training examples, ~30K test examples; long Wikipedia-context documents
- Metric: micro-F1 (higher is better)
- Medal thresholds: bronze ≈ 0.61913 (validation exceeded, test did not)

## Best result
- Score: 0.47602 | Medal: none | Run: 20260304_061536
- Approach: BERT-Base backbone with TF-IDF-based span candidate retrieval. Trained on 80K/276K examples for 2 epochs under timeout pressure. Validation F1=0.6293 but dropped to 0.476 on test—large generalization gap.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260304_061536 | 0.47602 | none | BERT-Base + TF-IDF span filtering; partial-data training |

## Proven strategies
- BERT-Base backbone fits within training + inference budget where heavier backbones do not
- TF-IDF span candidate filtering is faster than dense retrieval for shortlisting
- SQuAD2 pretraining as initialization helps revision attempts

## Failure patterns
- DeBERTa-v3-Large: sliding-window over full docs too slow; never finished 1 epoch
- BigBird: training timeout; sparse attention fell back to dense, erasing its advantage
- Test-time TF-IDF vectorization is O(n_examples × vocabulary) on CPU—primary inference bottleneck; even revised predict.py timed out
- Revision candidates (DeBERTa-v3-Base SQuAD2, RoBERTa-Base SQuAD2) both timed out during training
- Large validation→test gap (0.629 → 0.476, ~15% drop): overfitting on 80K subset and/or uncalibrated confidence threshold

## Feature engineering playbook
- Use TF-IDF shortlist only if indices are precomputed; avoid per-example refitting at inference
- Prefer static retrieval (BM25 or precomputed dense embeddings) over per-example TF-IDF for 30K-test scale
- Keep span candidate count small to survive inference timeout

## Tuning insights
- Inference speed—not model capacity—is the binding constraint; prefer BERT-Base over DeBERTa-Large/BigBird
- 2 epochs on 80K is underfitting; budget must allow full-data passes (≥4–6 epochs on 276K) to close the val/test gap
- Calibrate the answer/no-answer confidence threshold on a held-out split; do not reuse the training-val threshold
- Validation F1 from partial-data training is an unreliable proxy for test F1

## Open questions
- Is the 0.629→0.476 gap driven by threshold miscalibration, distribution shift, or subset overfitting?
- Can precomputed BM25 / dense embeddings eliminate the TF-IDF inference bottleneck without hurting recall?
- With a longer timeout, does full 276K × 4–6 epoch training recover test F1 toward validation levels?
- Can BigBird's sparse attention be forced on (not silently downgraded) to make long-context training viable?
