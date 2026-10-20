# The recipe

What to build, in the order that the evidence says matters. Every claim here was
measured on Qwen2.5-14B-Instruct; the figures are from `RESULTS.md`.

## The proposer

One EAGLE-style decoder layer over the target's post-final-norm hidden state,
reusing the target's own embedding and LM head (never shipped with the draft --
the framework ties them at load time).

```
layers        1
inter         13824          <- see THE DATA PRECONDITION
hidden        5120           (fixed: the target's)
n_heads       40             (fixed)
kv_heads      8              (fixed: GQA)
head_dim      128            (fixed)
fc            Linear(2*hidden -> hidden)   fuses input embedding + target hidden
norm          RMSNorm
params        ~328M
tree          4 steps / 24 topk / 64 draft tokens
```

## 1. THE DATA PRECONDITION -- read before choosing `inter`

**inter 13824 requires at least ~5M self-distilled training tokens. Below ~3M
tokens, use inter 8192 instead.** This is a precondition, not a companion fact.

It is phrased that way because the mistake has been made. Given "inter 13824
trained on 5.9M tokens" as a single recommendation, a run adopted the geometry
and kept a 2.37M-token corpus -- which is strictly *worse* than a narrow head at
that budget. It scored 1.6961 where the narrow head had reached 1.9760.

Measured in one step, geometry and tree held fixed, only the corpus changed:

| corpus | epochs | internal score | tau |
|---|---|---|---|
| 2.37M tokens | 2 | 1.6876 | 2.732 |
| 4.37M tokens | 2 | 1.8461 | 3.018 |
| 4.37M tokens | 3 | 1.8837 | 3.061 |
| 4.37M tokens | 4 | **1.9470** | **3.163** |

**+15.4% from data volume and epochs alone**, with no architecture change.

## 2. Corpus composition: weight toward the weak domain

Per-domain single-step accuracy of a trained head:

```
chat       0.58      <- the weak domain; over-represent it
code       0.78
reasoning  0.86
```

The step that produced the +15.4% moved its mix from chat 2900 / reasoning 2300
/ code 2700 to **chat 8000 / reasoning 3500 / code 3500**.

Do not substitute epochs for tokens. On another target, ~11 epochs over 0.57M
tokens drove training loss to 0.054 and still reached only 0.54 single-step
accuracy on held-out positions. Passes over a small corpus buy acceptance that
does not generalise.

## 3. Width is a null. Twice.

inter 13824 -> 16384 on the same 4.37M corpus: **1.8416 vs 1.8461** --
indistinguishable, against a ~0.30% repeat spread. An earlier null at 2.6M
tokens was dismissed as data-limited; it was not only that.

**Scale the corpus first. Widen last, if at all.** If you widen, state the data
scale you widened at.

## 4. Deep supervision over several future positions

Training the head against multiple future positions, not just the next token,
was the single largest late-stage gain: tau 3.163 -> 3.302 -> 3.359 -> 3.701
across steps 6-8, carrying the internal score from 2.0333 to 2.1342.

This targets acceptance (tau) rather than per-cycle cost, and it kept paying
after data and epoch tuning had flattened.

## 5. The tree matters more than earlier evidence suggested

The final head uses `4 steps / 24 topk / 64 draft tokens` and beats a
13824-geometry head on the same corpus running `4/16/64` by **5.7% on MT-Bench
(2.0089x vs 1.9001x) while having slightly LOWER tau (3.2061 vs 3.2486)**.

Lower acceptance with higher throughput means the gain is per-cycle cost, not
acceptance -- a tree effect. An earlier 2x2 on this project had found tree
changes worth only +2.7%/-0.8%, so treat that as refuted at this geometry and
re-tune the tree once the head is trained.

## 6. Decompose before you optimise

`speedup = tau / per-cycle cost`. Tuning cost is fast, safe and runs out;
improving tau means training, which is slow and may not pay back inside one
step. Both are legitimate. Measure which factor a change moved, and by how
much, before committing to it -- a step that leaves the score flat while
measurably raising tau is progress, provided the number is recorded.

## What not to bother with

- **Widening `inter` beyond 13824** without first getting past ~5M tokens.
- **Chasing differences under ~1.5%.** That is inside the instrument's
  cross-process resolution; see RESULTS.md.
- **Believing a single sweep.** `statistics.median` of two values is their mean,
  so two sweeps reject no outlier. Use three, and prefer max over median:
  throughput noise is one-sided, since a stall can only slow a sweep.
