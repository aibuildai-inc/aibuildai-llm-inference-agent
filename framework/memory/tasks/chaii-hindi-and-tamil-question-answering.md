# Task Memory: chaii-hindi-and-tamil-question-answering

## Task profile
- Type: Extractive question answering (multilingual)
- Dataset: 1,002 training samples (662 Hindi, 340 Tamil); long contexts (~11K chars mean); Devanagari and Tamil scripts; known data quality issues (Hindi ~93.8% clean per docs)
- Metric: Word-level Jaccard similarity (higher is better)
- Medal thresholds: Bronze ≥ 0.73725; Silver achieved at 0.74274

## Best result
- Score: 0.74274 | Medal: Silver | Run: 20260226_181358
- Approach: MuRIL-large-cased with two-stage transfer (SQuAD v2 → chaii), XQuAD Hindi augmentation (+1,190 samples), 5-fold ensemble at inference via start/end logit averaging. Sliding window tokenization (max_seq_length=384, doc_stride=128).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260224_042133 | 0.6656 | None | XLM-R-Large-SQuAD2 single model, XQuAD augmentation, 4× Tamil oversampling, 5-fold CV |
| 20260225_184425 | 0.71176 | None | XLM-R-Large-SQuAD2 + MuRIL ensemble (best OOF 0.73393), balanced negative subsampling; timed out before convergence |
| 20260226_181358 | 0.74274 | Silver | MuRIL-large-cased backbone, two-stage transfer, XQuAD Hindi augmentation, 5-fold logit-averaging ensemble |

## Proven strategies
- **Two-stage transfer learning is non-negotiable**: SQuAD v2 pre-training → task fine-tuning; direct fine-tuning on 1,002 samples cannot recover QA format knowledge (random QA head scored 0.5566 vs. SQuAD2-pretrained 0.7204)
- **Language-specific pre-training wins**: MuRIL-large (17 Indian languages) outperformed generic XLM-R-Large on final scores; language specialization is the core differentiator for Hindi/Tamil
- **5-fold logit-averaging ensemble at inference**: Consistent +0.01–0.02 Jaccard boost, minimal inference overhead, no retraining cost
- **XQuAD Hindi augmentation**: 1,190 aligned bilingual QA pairs; high ROI because same domain and target language
- **Balanced negative subsampling for sliding windows**: ~90% of windows contain no answer; 2–3 negatives per positive essential (not optional)
- **Sliding window with conservative stride**: doc_stride=128 for max_seq_length=384 handles long contexts; smaller strides unreliable due to script complexity
- **Stratified 5-fold CV by language**: Prevents lopsided splits inflating/deflating metrics given 2:1 Hindi/Tamil imbalance

## Failure patterns
- **Aggressive span filtering on noisy data**: Positive-span-only training dropped score from 0.6794 → 0.5529 (run 1); competition data quality issues make filtering counterproductive—regularize via dropout/early stopping instead
- **Removing all negative windows**: Hurt validation by eliminating null-answer calibration signal needed for SQuAD2 transfer heads
- **Smaller model capacity**: XLM-R-Base (~270M params) consistently killed by judge; cannot bridge gap to medal threshold vs. Large (~560M) variants
- **Cross-lingual proxy data without target languages**: TyDiQA (Telugu/Bengali, no Hindi/Tamil) provided no value; killed in run 3
- **Post-processing-only candidates**: Word-boundary snapping on Devanagari/Tamil scripts is unreliable; insufficient design diversity
- **Compute timeouts**: Ensemble training (especially dual-backbone) repeatedly hit 1,200s budget; candidate 6 in run 2 achieved 0.73393 OOF but timed out before final convergence

## Feature engineering playbook
- **4× Tamil oversampling**: Compensates for 662:340 Hindi/Tamil imbalance; passive reweighting insufficient on small sets
- **XQuAD Hindi augmentation (+1,190 samples)**: Highest-ROI external data source tested; same domain, same language, aligned bilingual pairs
- **Sliding window offset mapping**: Accurate character-to-token span alignment is foundational; must handle multi-byte Devanagari/Tamil characters correctly
- **External data source ranking**: XQuAD Hindi (proven) > TyDiQA multilingual (failed); prioritize same-language, same-task sources

## Tuning insights
- **Learning rate sensitivity**: LR=2e-5 outperformed 1e-5 (run 1: v1 at 2e-5 scored 0.6794 vs. v3 at 1e-5 scored 0.6656); higher LR for small datasets with pretrained heads
- **Layer-wise LR decay**: 0.9 decay factor used successfully across runs
- **Epochs**: 3–5 epochs optimal; 8 epochs (run 1) showed no benefit over shorter schedules
- **Weight decay 0.01, warmup 10%**: Standard regularization; no ablation showed improvement from deviating
- **Batch size 16**: Stable across runs given GPU memory constraints
- **Discriminative LR for QA head**: MuRIL may benefit from higher QA-head LR (5e-5) vs. base layers (1.5e-5) when QA head is randomly initialized; not fully tested
- **GPU contention**: Sequential execution preferred over parallel cuda:0/cuda:1 runs

## Open questions
- **Dual-backbone ensemble ceiling**: XLM-R-SQuAD2 + MuRIL ensemble (candidate 5, run 3) hit 0.74643 OOF but was not fully tuned; aggressive hyperparameter search could push >0.76
- **XQuAD Tamil availability**: XQuAD Hindi helped significantly; XQuAD Tamil (if accessible) would likely provide comparable boost for the 340 Tamil minority samples
- **MuRIL QA-head learning rate**: MuRIL used same discriminative LR schedule as XLM-R-SQuAD2; try 5e-5 for randomly-initialized QA head to compensate for lack of QA pre-training
- **Epoch-by-epoch convergence diagnostics**: Run 1 showed score regression when moving from 3-fold to 5-fold with lower LR; need loss curve analysis to disentangle fold count vs. LR effects
- **Answer_start fixing logic**: Positive-span filtering catastrophically failed (0.5529); unclear if root cause was data corruption from answer_start alignment code or structural overfitting—needs ablation
