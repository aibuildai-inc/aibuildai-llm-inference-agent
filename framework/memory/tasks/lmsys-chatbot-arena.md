# Task Memory: lmsys-chatbot-arena

## Task profile
- Type: 3-class classification (winner_a, winner_b, tie)
- Dataset: ~51K human preference comparisons (prompt + response_a + response_b); class balance ~35/34/31; ~60% of samples exceed 512 tokens (p90 ~1500 tokens)
- Metric: multi-class log loss (lower is better)
- Medal thresholds: bronze ≤ 1.00283; silver ≤ ~0.988; gold unknown

## Best result
- Score: 0.98784 | Medal: silver | Run: 20260225_224850
- Approach: DeBERTa-v3-large single-sequence classifier (512 tokens, 25%/75% prompt/response split), swap augmentation, multi-sample dropout (5 masks, p=0.3), label smoothing (0.15), temperature scaling, and test-time augmentation (response swap + average). Virtually tied with run 20260211 (0.98785).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260211_164139 | 0.98785 | silver | DeBERTa-v3-large 1-epoch, label smoothing 0.02, TTA swap, proportional truncation 20/40/40 |
| 20260222_231056 | 1.03046 | none | Ensemble of dual-encoder LightGBM + DeBERTa base/large; many candidates timed out or killed |
| 20260225_224850 | 0.98784 | silver | DeBERTa-v3-large + multi-sample dropout, label smoothing 0.15, temperature scaling, TTA |

## Proven strategies
- **DeBERTa-v3-large is the dominant backbone**: Consistently best across all runs; outperforms DeBERTa-v2-xlarge (too slow), Longformer (weaker despite 4K context), and frozen-embedding approaches
- **Swap augmentation is mandatory**: Flipping response_a/response_b and mirroring labels doubles effective data at zero cost; every competitive candidate used it
- **Test-time augmentation**: Average predictions from original + swapped inputs exploits label symmetry; free accuracy boost at inference
- **Temperature scaling**: Post-hoc calibration reduced overconfident predictions; 15%+ log loss improvement over raw logits (1.32 → 1.11 in one run)
- **Label smoothing**: Both 0.02 and 0.15 improved calibration; 0.15 worked well in latest run with multi-sample dropout
- **Multi-sample dropout (5 masks, p=0.3)**: Directly improved calibration and log loss vs. single dropout (proven in run 3)
- **bf16 mixed precision on A100**: Stable, fast, enables larger batch sizes; fp16 caused OOM on model loading

## Failure patterns
- **DeBERTa-v2-xlarge**: Killed in both runs 2 and 3; 885M params too slow (30–50 min/epoch), 17 params/sample overfitting risk, timeout failures even after restructuring
- **Llama-2-13B / large LLMs with LoRA**: peft/bitsandbytes not installed in environment; 1–3 samples/sec too slow for automated pipeline
- **Extended context (1024–1536 tokens)**: Hits compute wall every time; 1536 reached only 18% of epoch before timeout. 512 tokens is the practical ceiling given budget
- **Multi-model ensembles exceeding budget**: Multi-Window DeBERTa (3×434M), complex stacking pipelines consistently consumed entire 24h budget
- **Checkpoint management**: Best checkpoint overwritten by worse subsequent run (run 3); model didn't support resume → forced full retrain
- **Coder sanity check timeouts (1200s)**: Candidates 1/2/3 in run 2 timed out; required lightweight reruns with skipped test prediction

## Feature engineering playbook
- **Token budget allocation**: 25% prompt / 75% split between responses (latest); earlier 20/40/40 also worked. Proportional truncation preserves semantic content
- **LightGBM on hand-crafted features**: Length diffs, token ratios, verbosity metrics provide orthogonal signal; trains in minutes, useful as ensemble diversity component (best solo: ~1.01–1.03)
- **Dual-encoder frozen DeBERTa + LightGBM**: Fast iteration (~4–5h for 5-fold CV), reached 1.0313; useful for rapid exploration but ceiling is lower than fine-tuning

## Tuning insights
- **1 epoch is often sufficient**: DeBERTa-v3-large overfits after ~1 epoch (val loss degrades); best checkpoints found mid-to-late first epoch
- **LR schedule**: Cosine annealing with warmup (10% of steps) works; cosine + checkpoint resume caused instability in run 3 → linear decay more robust for recovery
- **Batch size**: bf16 enables larger batches; gradual warmup stabilizes early training
- **No class weighting needed**: ~35/34/31 balance is close enough; label smoothing handles calibration
- **MAX_LENGTH=512**: Truncates ~50–60% of samples but is the speed-accuracy sweet spot; 1024 may help but never successfully completed training within budget

## Open questions
- **Stacking ensemble**: OOF from DeBERTa-v3-large (best ~1.004) + LightGBM features with meta-learner could push to 0.99–0.995 but never successfully implemented; requires careful fold alignment
- **Position interpolation for DeBERTa-v3**: Extend to 1024 tokens via interpolation of position embeddings; untested but could unlock gains from longer context without Longformer's weaker backbone
- **Knowledge distillation**: Distill DeBERTa-v3-large into Longformer-base to get full 4K context with better representations
- **LoRA on 7B+ models**: Requires peft pre-installed; if environment supports it, Llama-2-7B or Mixtral could break the silver ceiling
- **2 epochs at shorter max_seq_len (768)**: Untested whether additional pass compensates for truncation; may help if overfitting is controlled
