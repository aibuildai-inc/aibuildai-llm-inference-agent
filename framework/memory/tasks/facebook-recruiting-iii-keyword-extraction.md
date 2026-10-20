# Task Memory: facebook-recruiting-iii-keyword-extraction

## Task profile
- Type: Multi-label classification (keyword/tag extraction)
- Dataset: ~6M Stack Overflow posts (6.1GB), title + body text → tags; ~23.6K–42K unique tags
- Metric: Micro F1 (higher is better)
- Medal thresholds: Bronze ≈ 0.71345 (estimated); gold achieved at 0.8026

## Best result
- Score: 0.8026 | Medal: gold | Run: 20260422_234718
- Approach: Candidate 4 = DistilBERT mean-pooled embeddings + LightXML sampled head (400 negatives/step) + Asymmetric Loss (γ_neg=4.0), LLRD 0.95, bf16 + gradient checkpointing. Clean-body-hash duplicate override covered ~53% of test at inference; literal tag-in-text override + per-tag threshold calibration on 200K-row holdout applied post-hoc. Validation micro-F1 0.6154 → test 0.8026 (duplicate override dominates).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260419_174442 | — | — | Setup failed: Kaggle API HTTP 401 |
| 20260419_175901 | 0.56098 | none | Title-hash dup lookup + HashVec+SGD 0.432 |
| 20260419_235913 | 0.6668 | none | Streaming FTRL + hashed n-grams + 3-round per-tag thresholding |
| 20260421_191250 | 0.22929 | none | ASL+SparseAdam+FTRL ensemble (internal 0.6850, submission regressed) |
| 20260422_183340 | 0.00748 | none | FTRL++ per-tag gates + literal override (val 0.6445 → test collapse) |
| 20260422_234718 | **0.8026** | **gold** | DistilBERT+LightXML+ASL + clean-body-hash override (~53% test cov) |
| 20260424_001702 | 0.4930 | none | RealTFIDF + per-tag liblinear OvR + dedup + literal override (TOP_K=200) |
| 20260424_173120 | 0.69057 | none | DistilBERT+LightXML+ASL TOP_K=12K + dup-override (53.26% cov); grid [0,0.60] saturated |
| 20260424_174159 | 0.69247 | none | BERT-base+LightXML+ASL+dup-override+literal-fusion; tuner stopped early at step=5 |

## Proven strategies
- **Clean-body-hash duplicate override at inference**: Dominant lever — ~53% test coverage; orthogonal to backbone (DistilBERT, BERT-base, CodeBERT, FTRL all benefit); THE bronze/gold-breaking move; day-1.
- **DistilBERT (6-layer, 768d) + LightXML sampled head**: Scales to 12K–13K labels with 400 random negatives/step + positives; bf16 + gradient checkpointing at batch 256+. Generic pretrain outperforms code-aware backbones (CodeBERT/SOBERT underperformed in Run 9). Late-breakthrough: budget 8–12+ rounds.
- **Asymmetric Loss (γ_neg=4.0, γ_pos=0.0)**: Downweights easy negatives; broke 0.586 balanced-loss plateau; essential for LightXML-style sampled training.
- **LLRD 0.95 + linear warmup**: Prevents cold-start transformer divergence at extreme label space.
- **Per-tag OvR with real TF-IDF (word 1-2 + char_wb 3-5, sublinear_tf, min_df=3) via joblib.Parallel**: Beats hashing-streaming and four-head fusion; sparse CSR avoids dense weight overhead.
- **Per-tag threshold calibration on 200K holdout**: Per-quantile (head/mid/tail) thresholds ~+0.08 F1; **grid MUST span [0.0, 1.0]** — Run 8 saturated at 0.60, likely cost +0.05–0.08.
- **Literal tag-in-text OR-override**: +2–3 F1 for code/identifier tags; orthogonal to learned features, day-1.
- **Streaming FTRL single-pass fallback**: 0.6111→0.6522→0.6668 monotonic; ceiling ~0.6668; diversity value only.
- **Strict submission format validator + artifact persistence guards** (`os.makedirs(exist_ok=True)`, `assert Path(results_json).exists()`): non-negotiable kill-gate; Run 9 lost C1 gold-repro to missing artifacts.

## Failure patterns
- **Validation-to-test gap works BOTH directions**: Run 6 val 0.6154 → test 0.8026 (dup boost); Runs 4–5 val 0.68/0.64 → test 0.23/0.007. Always estimate dup-coverage pre-submission.
- **Threshold grid truncation [0.0, 0.60]**: Run 8 saturated upper boundary; full [0.0, 1.0] mandatory.
- **Artifact persistence failures kill candidates**: Run 9 C1 (gold-repro) and C4 killed by judge despite complete code; sanity-run results.json/best_model.pth/submission.csv must be written explicitly.
- **Tuner wall-time exhaustion**: Run 9 stopped at step=5 with only 2 tuner runs vs budget 40 calls; profile per-candidate wall-time + cap max_train_steps.
- **Multi-model ensemble fusion fragility**: Run 8 C5 fusion crashed; Run 9 C4 (TF-IDF+SGD⊗DistilBERT) killed; per-tag independent logistic over multiple bases is high-risk.
- **Domain-specific backbones underperform**: Run 9 CodeBERT (C3) and SOBERT (C6 fallback) both lagged generic DistilBERT/BERT-base; HF Hub timeouts (Run 8 1200s) compound cost.
- **CascadeXML hierarchical head**: Run 8 C3 collapsed 0.1566; flat sampled-softmax dominant.
- **Single-paradigm OvR beats ensemble fusion**: Run 7 C6 dense 4-head OOM; clean OvR won.
- **Streaming FTRL trap for TOP_K expansion**: Run 7 C1 timed out 1200s; prefer batch TfidfVectorizer.
- **OvR scaling cliff**: Run 7/8 hit TOP_K=200 vs 10K target → F1 0.49.
- **BM25/kNN tag transfer / semantic kNN on MiniLM / FastText hashed / Aho-Corasick / FTRL-L1 / multi-seed hashing**: all regressed (0.099–0.43 range).
- **TOP_K expansion without holdout scaling**: 10K→13K with 30K val broke calibration.

## Feature engineering playbook
- Clean-body-hash duplicate pre-assignment + train-dedup (~4.2M rows) — THE highest-ROI feature; day-1.
- Literal tag-vocab-in-text OR-fusion (NOT post-override): +2–3 F1; day-1; tune gate threshold to avoid false-positive inflation.
- Real TF-IDF (word 1-2 + char_wb 3-5, sublinear_tf, min_df=3, max_df=0.9) for non-transformer; reservoir IDF on 400K–1M.
- TOP_K tag vocab 10K–13K sweet spot for transformer+sampled-head; OvR may cap at 200–1K.
- Title weighted 3:1 over body. Preserve code blocks case-sensitive in separate namespace.
- Pre-submission diagnostic: estimate clean-body-hash test coverage to predict val→test gap.

## Tuning insights
- DistilBERT/BERT-base+LightXML+ASL + dup-override is medal-tier anchor; TF-IDF+OvR strongest non-transformer fallback.
- Tuner-only headroom exhausts ~step 12–14 (~0.56–0.61 ceiling on non-transformer); architectural moves required.
- Per-quantile thresholds > global; **grid range [0.0, 1.0] mandatory**; holdout 200K stable, 30K unstable.
- Sparse/sampled optimizers (SparseAdam, FTRL, LightXML negatives) mandatory at 10K+ labels.
- Streaming chunked reads at 20K-row batches stable; never load full 6M CSV.
- bf16 + gradient checkpointing enables batch 256+ at 13K labels.
- Single-paradigm + bronze levers (dedup + literal + thresholds) > multi-head fusion.
- Generic transformer pretrain > code-aware backbones for SO tag prediction (Run 9 confirmed).
- Late-breakthrough: early val (0.2–0.3) ≠ final; don't kill C1-style transformers prematurely.

## Open questions
- **Threshold grid [0.0, 1.0] retry on Run 8/9 design**: expected +0.05–0.08, would push 0.69→bronze 0.71345.
- **Artifact persistence harness fix**: explicit `os.makedirs+assert` guards in coder template to prevent Run 9-style C1 kills.
- **Tuner wall-time budgeting**: Run 9 used only 2/40 calls — instrument per-candidate timing and cap max_train_steps.
- **Inference-time ensemble TF-IDF+OvR + DistilBERT+LightXML** (avoid Run 8 C5 fusion crash): lexical+semantic diversity.
- **Sparse C6-style four-head fusion** (Run 7 dense OOM'd): checkpointed sparse updates.
- **Tag co-occurrence Apriori rules** (java→spring): lightweight fusion, +1–2 F1; Run 9 C5 partial attempt — needs higher gate thresholds.
- **FTRL with decaying L1/L2** as fallback when transformer budget tight.
