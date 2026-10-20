# Task Memory: AI4Code

## Task profile
- Type: Ranking/regression (predict relative ordering of markdown cells among code cells in Jupyter notebooks)
- Dataset: 119K training notebooks; median 35 cells per notebook (26 code, 11 markdown), max 1005 cells; 93% of notebooks appear in isolation (only 5091 ancestor groups have >1 notebook)
- Metric: Kendall tau correlation (↑ higher is better)
- Medal thresholds: Bronze ≈ 0.8533 tau; top solutions reach 0.86–0.88 tau

## Best result
- Score: None (no candidate completed) | Medal: ungraded | Run: 20260224_193316
- Approach: Attempted DeBERTa-v3-base pointwise regression, GraphCodeBERT, and Pairwise CodeBERT ranking. DeBERTa candidate progressed furthest (model loaded, 83K training samples built, epoch 1 started) but timed out at 1200s.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260224_193316 | None | ungraded | All 3 candidates timed out; DeBERTa-v3 pointwise got furthest (began training) |

## Proven strategies
- *None validated with scores yet.* However, Kaggle leaderboard references confirm:
  - DeBERTa-v3-base pointwise regression is the backbone of top-1 solution (0.8637 tau)
  - Pairwise CodeBERT classifier + greedy insertion reaches bronze (~0.8533 tau) per public notebooks
  - Bi-encoder cross-attention architectures appear in 2nd-place solution (~0.88 tau)

## Failure patterns
- **Coder timeout is the dominant failure mode**: All 3 candidates (GraphCodeBERT, DeBERTa-v3, Pairwise CodeBERT) killed at 1200s. Agents spent >95% of wall time exploring APIs/data instead of executing quick sanity runs.
- **Exploration-first agent behavior**: Coders investigated library details and data formats exhaustively before attempting any training. Must force run-first, iterate-second workflow.
- **No pre-constrained configs**: Without hard limits on sample fraction and epoch count in config templates, coders default to full-data runs that blow the time budget.

## Feature engineering playbook
- **Context window is the critical bottleneck**: BERT-family models cap at 512 tokens; median notebook has ~37 cells. Code context selection strategy (sampling, truncation, TF-IDF ranking of code cells) is likely the highest-leverage design choice.
- **Ancestor group augmentation is limited**: 93% of notebooks are singletons—cross-notebook data augmentation via ancestor groups will have minimal impact.
- **Notebook length segmentation**: Consider separate strategies for short (<30 cells) vs. long (>100 cells) notebooks, as context compression difficulty scales nonlinearly.

## Tuning insights
- **Force tiny sanity runs first**: Use sample_frac=0.005–0.01, 1–2 epochs, to validate pipeline end-to-end before scaling up. This is non-negotiable given 1200s coder timeouts.
- **Pre-set config constraints for tight budgets**: Bake sample_frac and max_epochs into config templates so coders cannot default to full-data training.
- **DeBERTa-v3-base over CodeBERT**: Expected performance ordering from leaderboard evidence is DeBERTa-v3 > CodeBERT for pointwise regression. Prioritize DeBERTa.

## Open questions
- Will DeBERTa-v3 pointwise regression reach bronze if trained to completion? (It began converging before timeout—likely yes with sample_frac=0.01, 1–2 epochs.)
- What is the actual tau gain from pairwise ranking consistency vs. loose pointwise position predictions?
- Does notebook length correlate with markdown position difficulty, and should model/strategy be conditioned on it?
- What is the optimal code-context selection strategy (random sample vs. TF-IDF vs. positional neighbors) for the 512-token window?
- Can a lightweight LightGBM baseline (no transformers) serve as a fast sanity-check score anchor?
