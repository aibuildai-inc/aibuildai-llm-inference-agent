# Pattern: reasoning-and-problem-solving

## When this applies
- Tasks requiring ordering/ranking of structured items (e.g., notebook cells, document segments)
- Long-context inputs that exceed transformer token limits (median 37 cells in AI4Code vs 512-token cap)
- Datasets with limited cross-instance augmentation (93% singletons in AI4Code)
- Metric is rank correlation (Kendall tau) rather than pointwise accuracy

## Proven patterns
- *No validated runs yet*; AI4Code (run 20260224_193316) produced None score. Leaderboard evidence (not yet reproduced) suggests:
  - DeBERTa-v3-base pointwise regression is the top-1 backbone (~0.8637 tau on AI4Code)
  - Pairwise CodeBERT + greedy insertion reaches bronze (~0.8533 tau)
  - Bi-encoder cross-attention reaches ~0.88 tau (2nd place)

## Common pitfalls
- **Coder timeouts dominate failures**: All 3 AI4Code candidates (GraphCodeBERT, DeBERTa-v3, Pairwise CodeBERT) hit 1200s wall and produced no score
- **Exploration-first behavior**: Coders spent >95% of wall time on API/data investigation before running training (AI4Code run 20260224_193316)
- **Unconstrained configs**: Without baked-in sample_frac/epoch caps, coders default to full-data runs that exceed budget
- **Skipping sanity runs**: No tiny end-to-end validation before scaling = wasted budget on broken pipelines

## Model selection guide
- Long-context ranking with code → DeBERTa-v3-base pointwise (highest leaderboard evidence)
- Tight time budget / sanity check → LightGBM positional baseline before transformers
- Pairwise consistency needed → CodeBERT pairwise + greedy insertion (proven bronze-tier)
- Avoid: GraphCodeBERT as first attempt (AI4Code: timed out before producing signal)

## Feature engineering patterns
- **Context window selection is highest-leverage**: 512-token cap vs median 37 cells means code-cell sampling/TF-IDF ranking matters more than model choice
- **Skip cross-instance augmentation when singletons dominate**: 93% singleton rate in AI4Code makes ancestor-group augmentation low-ROI
- **Segment by input length**: Separate strategies for short (<30 cells) vs long (>100 cells) inputs since compression difficulty scales nonlinearly

## Tuning patterns
- **Force tiny sanity runs first**: sample_frac=0.005–0.01, 1–2 epochs, validate pipeline end-to-end (non-negotiable under 1200s coder limits per AI4Code evidence)
- **Pre-constrain config templates**: Hard-code sample_frac and max_epochs so coders cannot default to full-data training
- **Run-first, iterate-second**: Mandate a working tiny run before any data/API exploration
- **Anchor with non-transformer baseline**: LightGBM positional features give a fast score floor even when transformer runs OOM/timeout
