---
name: aibuildai-specdec-recipe
description: >-
  What eight prior 10-step searches learned about training exact speculative-decoding
  draft heads, filtered for the fact that SGLang now owns decoding. Says which findings
  still apply to the head (architecture family, regression-dominant loss, data volume
  and composition, negatives), which have moved out of scope entirely (CUDA graphs,
  single-sync builds, tree-node sweeps, per-cycle profiling), and the measurement traps
  that cost earlier runs real time. Read this before planning the first change.
---

# Training a draft head when the framework owns decoding

Eight complete 10-step searches preceded this task, all on the same target and host, all
scored differently: speedup over an eager HuggingFace `generate()` loop. That harness is
gone. **You submit a trained head; SGLang runs it.** This document carries forward the
part of those eight runs that still applies and is explicit about the part that does not.

Evidence tags: **[8x]/[6x]/[5x]** = independently reached by that many runs.
**[1x]** = one run only. **[neg]** = tried and failed. **[moved]** = no longer yours.

---

## Read this first: the eager harness rewarded things that do not deploy

The best eager submission scored **4.33x** there and is **slower than stock SGLang**.
Two high-scoring heads cannot be loaded by any serving framework at all -- one carried an
extra residual MLP plus a final norm, another had no attention whatsoever. The search
found excellent answers to a question nobody asked.

So: the architecture is fixed by `reference_head.py`, and the score is a ratio
inside SGLang. Your levers are the head's weights, its size, its training, and three
speculative tree parameters.

**Expected spread.** Two pairs of eager-harness runs with byte-identical inputs produced
3.2084 vs 2.6775 and 4.3345 vs 3.1384 -- **17% and 28%** apart on scores, **22.6% and
32.7%** in absolute tok/s. The search is high-variance. Treat any single number,
including every number below, as one sample.

## What the first run of THIS task reached, and where it stopped **[1x]**

`specdec-sglang-v1` scored **1.971x** on this task's own 40-prompt harness (103.26 tok/s
against a 52.38 baseline), reached at **step 4 of 10**.

> **Calibration you need before you trust your own score.** That same head measures
> **1.6637x on MT-Bench** (80 questions x 2 turns, 1024 tokens, greedy) and **1.5430x**
> at temperature 1. A second host measured four of its own searches on **Spec-Bench**
> (480 instances, 6 domains) and found the same thing: its best head scored 2.1726 on
> this harness and **1.8641x** official, and all four heads inflated by **11.2-16.5%**.
> Two independent benchmarks, two hosts, same conclusion. The baseline barely moves between the two protocols (52.38 vs 52.66),
> so the whole ~16% gap is acceptance length: **this task's prompt set is easier to
> predict than MT-Bench.** It has no math and no code and is mostly explanatory.
>
> Your score here is still the right thing to optimise — it is the task's metric and it
> is measured consistently. But do not report it as a deployment number, and expect a
> real workload to land lower. Per-category MT-Bench speedups span 1.412x (writing) to
> 2.296x (math).
>
> Quality is unaffected as far as 664 paired items can show: GSM8K 92.2% vs 91.2%,
> HumanEval 80.5% vs 79.9%, neither significant (combined McNemar p = 0.109), though the
> sign is consistently negative so a sub-1.5% degradation is not excluded. Steps 5-10 ran real experiments, none beat it, and four of the nine
scores were the incumbent head resubmitted. If you are starting fresh, the useful
information is what is already eliminated -- all measured, not reasoned:

| lever tried | result |
|---|---|
| head width 4096 / 8192 / 12288 (178M / 241M / 304M) | all inside noise **at ~2.6M training tokens** -- see the warning below |
| 40 KV heads instead of GQA's 8 (283M) | did not beat the incumbent |
| CE loss weight, two settings | did not beat it |
| flat unroll weights | did not beat it |
| K=4 unroll depth instead of K=3 | did not beat it |
| tree 64 -> 72 draft tokens | tau +0.9%, **throughput -7.6%** |
| tree 3/32/80 | **would not start** -- see the workspace note below |

The winning configuration is modest: **1 layer, hidden 5120, inter 8192, 40 heads,
8 kv_heads, head_dim 128, tree 3 / 24 / 64**, with tau saturating at **3.13-3.17**.

**START FROM THIS CONFIG [measured head-to-head].** The best head the project has
produced: **1 layer, inter 13824, 327.7M params, 8 kv_heads, tree 4/16/64, trained on
5.9M tokens**. Measured on one host against one plain arm it reached **1.9001x on
MT-Bench (tau 3.2486)**, versus 1.6637x (tau 2.6579) for a 241.2M head at inter 8192 on
~2.6M tokens -- **+14.2% throughput, +22% tau, winning all eight categories**. It also
measures 1.8641x on Spec-Bench, so the two suites agree to 1.9%.

Do not start from the narrower config just because an earlier table called width a null.

**Width is NOT settled -- our null was data-limited [cross-host].** Another host's best
head used **inter 13824, 327.7M params, 5.9M training tokens** and reached **tau 3.737**
against this task's 3.149 at inter 8192 / 241M / ~2.6M tokens. It also changed the tree
(4/16/64), so width is not isolated -- but if you scale the data, re-test width. Do not
read the table above as "width is exhausted"; read it as "width did not pay at 2.6M
tokens".

**The tree is smaller than you will expect, and growing it loses.** Do not spend a step
re-discovering 72 nodes. If you want to explore tree size, go DOWN from 64 first.

**A 512 MiB wall sits just above the usable range.** `3/32/80` dies during CUDA-graph
capture with `Buffer overflow ... batch_prefill_tmp_v with size 660275200 ... only
536870912 bytes available in AlignedAllocator` -- flashinfer's default workspace, with
~40 GB of the card free. SGLang's suggested fixes (lower `mem-fraction-static`, shrink
the decode batch, disable decode graphs) all miss the cause, and the third would wreck
the measurement. It is not worth raising: the 64 -> 72 result says bigger trees are
slower anyway.

---

## Still yours, and still true

### 1. The head family **[8x]**
An autoregressive head in the target's own feature space, predicting the next
post-final-norm hidden state, with candidate logits from the target's **frozen**
`lm_head`. Every run converged here.

- **A separate draft model loses [3x-neg].** At batch 1 a forward costs a largely fixed
  dispatch; one run measured **0.67x**, a slowdown.
- **Medusa-style parallel heads lose [1x-neg].** Every offset predicted from the same
  frozen state caps depth-k accuracy at head k alone: offset-5 accuracy 0.209 versus
  0.432 for the autoregressive form.

This is already what `reference_head.py` implements. You are choosing its size and
training it, not redesigning it.

### 2. Regression-dominant loss **[5x]**
```
loss = SmoothL1(predicted_feature, true_feature) + ~0.1 * CE(token_logits)
```
Five runs landed on a CE weight of 0.1-0.2 independently. The feature term is the point.
Isolating it on matched data gave **+0.026 at offset-5** -- it helps most at depth, which
is where acceptance comes from.

**Hold total update steps fixed when you scale data**, or you are measuring duration
rather than diversity.

### 3. Data volume, then composition **[8x]**
Generate with the target itself, teacher-force to collect hidden states.

- **Start around 500k tokens, not 200k.** The run that opened at 512k had a usable head
  after one step; those opening near 200k spent an extra step climbing.
- **Draw each new slice from a disjoint prompt offset.** One run's first attempt
  overlapped its previous pool and contaminated its own held-out split.
- **Volume saturates near 1.5-2.5M tokens [2x-neg].** Two independent runs lost score on
  the step that pushed past it.
- **Then fix composition, which beats more volume [2x].** Measure **per-domain** tau,
  rebalance toward the weakest domains, and spend new data where tau is worst rather
  than uniformly. One run gained **+35% in a single step** this way (1.4976 -> 2.0181);
  a later run repeated it for **+12.7%**, and a third found a targeted math slice worth
  **+7.6%**. Rebalance, do not just scale.
- Memory is the wall: ~51 KB per example of hidden states at hidden 5120. Keep large
  tensors **fp16 on CPU and cast per batch**.

### 4. Capacity: cheap pays, expensive does not **[6x-neg / 2x]**
- **An extra full transformer layer never paid in six runs.** One tested 1-5 layers
  (layers 4-5 tracked 3 within ±0.008); another shipped a single layer for ten steps; a
  third's 2-layer head lost on the benchmark despite training fine; a fourth's was a
  null step with tau identical.
- **Cheap capacity can pay [2x].** An extra MLP rather than a full block gave **+3.6%**
  once, and a three-MLP stack won another run -- but that run also re-swept its tree
  narrower at the same time, so capacity and tree shape moved together. The distinction
  that holds is **cost per cycle, not parameter count**.
- Under this task the honest version of that lever is `n_layers` and `inter` in the
  required architecture. Try wider before deeper.

### 5. Depth beyond the trained horizon **[3x-neg]**
A tree level the head was never trained to predict well cannot pay for the vocabulary
projection it costs. One run trained a head specifically for a deeper horizon and still
lost. Train the unroll depth you intend to speculate at.

---

## No longer yours **[moved]**

These were large wins in the eager harness and are **SGLang's job now**. Attempting them
wastes steps, and in some cases you cannot reach the code at all.

| finding | why it no longer applies |
|---|---|
| CUDA-graph the target verify forward (+75% in one step, the single biggest eager win) | SGLang captures its own graphs. You do not own the decode loop. |
| CUDA-graph the tree builder (+4.3%) | same |
| single-sync tree descent, static topology/mask precompute (+5-11%) | same |
| sweeping tree node count to the bend in the verify-cost curve (+15.4%) | SGLang's tree is controlled by three integers, not a width schedule -- see below |
| "go wider, 250-300 nodes" | that advice came from a launch-bound eager loop. Under CUDA graphs the verify forward is **compute**-bound and the optimum moved to ~61 nodes. Do not port the number. |
| per-cycle profiling of build vs verify vs accept | not reachable from a weight file |
| proving a rewrite preserves tau structurally | you cannot change tau's definition; SGLang verifies |

**What replaces the tree sweep.** `head_config.json` may set `sglang_overrides` with
`speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`.
Those three are the whole tree. The defaults in `models.json` are 4 / 8 / 32 and are
**untuned** -- two trained heads measured 1.74x and 1.90x against stock SGLang at those
defaults, so there is likely headroom. SGLang requires
`num_steps * eagle_topk >= num_draft_tokens`.

---

## Exactness: you cannot break the accept rule -- but output is not bit-identical

Under greedy decoding SGLang's verify kernel accepts a draft token only when it equals
the target's own token, and stores the target's token either way. Every emitted token is
the target's argmax. Exactness is a property of the verify/accept logic, and that logic
is SGLang's.

What this means in practice:
- **Do not spend steps proving exactness.** Earlier runs needed an fp32 gate because the
  submission owned verification. You do not.
- **bf16 token streams may still differ between arms, and usually do.** A near-tie
  rounds differently between a cached single-token decode and a batched tree verify. The
  harness reports `identical_prompts` and does not require it to match.
  **Quantified 2026-10-14:** 10 of 30 prompts produce identical 128-token streams; the
  per-token divergence rate is ~0.86%. Confirmed numerical, not a bug, three ways: the
  plain arm is bit-deterministic (30/30 across cold processes); the divergence fits a
  uniform per-token flip model on all four distributional statistics with no early
  clustering; and it is flat across tau 2.2-3.18, whereas a logic error would scale with
  acceptance. Corroborated by the eager task, where one run saw 14 of 30 bf16 mismatches
  **while passing a strict fp32 gate 8/8**.
  So do not chase `identical_prompts`. But do not tell anyone the output is unchanged
  either -- it is a valid greedy decode, not the *same* greedy decode.
- SGLang cannot run fp32 at all (no Float dispatch in its attention or fused rmsnorm), so
  an fp32 gate inside it is not available even if you wanted one.

---

## How to measure, which every prior run learned the hard way

1. **The 30-prompt harness is authoritative. Dev numbers rank, they do not report.** One
   run swept to a dev speedup of 2.636 and the harness returned **2.119** -- a 20%
   overstatement. Smaller ±2-3% drift is routine. Use dev to choose, never to conclude.
2. **tau is the counting statistic.** The harness derives it from
   `meta_info["spec_verify_ct"]`: tokens emitted per target forward. It is far less noisy
   than tok/s and it is what your head actually controls. Steer by tau; report tok/s.
3. **The method arm is only repeatable to about +-1.2%. Do not trust a 1% win.**
   Measured: two steps scored a BYTE-IDENTICAL head (same md5, same config, tau equal to
   17 significant figures, so provably the same tokens) and returned **101.964 vs 104.340
   tok/s -- 2.33% apart**. An earlier set of four measurements of another head spanned
   only 0.52%, and that tight run was twice mistaken for the instrument's precision.
   A tight run of same-input measurements is NOT a noise estimate without a second
   artifact to cross-check it.
   Consequences you should act on:
     - Two configurations within ~2% are indistinguishable. Measure each twice before
       preferring one, or you will chase noise for a whole step.
     - Do not report four decimals. 1.9908 and 1.9760 are the same result.
     - The no-speculation arm is NOT the problem: nine measurements gave mean 52.380,
       sd 0.0348 (**0.07%**). Graph capture makes it ~15x more repeatable than the
       speculative arm. Whatever adds the variance is specific to speculation.
4. **Score a fixed reference head first, if you have the budget.** One known head
   measured at the start gives you a per-run noise estimate for free, and would have
   caught the +-1.2% above on day one instead of step 10.
5. **Scoring costs two engine loads (~40 min).** Two 14B engines cannot be resident at
   once, so the arms cannot be interleaved; both run back to back in one call. Plan on
   2-3 authoritative measurements per step, not ten.
6. **Run `check_draft.py` first.** Minutes, not 40. It catches an unloadable head, a
   missing patch, and a silently disabled speculator before you spend a full benchmark.
7. **A silently disabled speculator scores ~1.0 and looks like an honest null.** The gate
   checks `spec_verify_ct > 0` for exactly this reason. If you ever see a suspiciously
   clean 1.00, check that first.
8. **The first engine start JIT-compiles kernels.** That inflated a baseline by 13% once
   and nearly produced a false null result. The warmup absorbs it; never compare a cold
   arm with a warm one.
9. **Read the WARNING lines in both arm logs.** A vLLM comparison was void twice -- once
   from a quiet model-runner fallback, once from async scheduling disabled in only one
   arm. Both were visible only as warnings. Any fallback zeroes the gate here.
10. **Validation accuracy is frequently anti-correlated with the harness.** One run's head
   won every held-out offset and lost on tau; another gained 35% on the harness while its
   dev chain accuracy got worse. Rank candidates by tau, decide by the harness.
11. **Never compare validation numbers across different corpora or domain mixtures.**
   Every prior run was burned by this.
12. **Verify checkpoint identity by hash before measuring.** One run copied a `.pt`
    mid-save; another loaded a head with tensors silently missing, leaving a layer
    randomly initialised.

---

## Order of work, by measured return per hour

```
1. Get a trained head through check_draft.py and banked        [8x]
2. Data volume to ~1.5M tokens, then STOP                      [8x]
3. Per-domain tau -> rebalance composition                     [2x, +35%/+12.7%/+7.6%]
4. Regression-dominant loss, CE weight 0.1-0.2                 [5x]
5. Sweep the three sglang_overrides (defaults are untuned)     [0x -- untested, cheap]
6. Wider MLP / more layers, re-measuring each                  [6x-neg on full layers]
```

Items 1-2 are where every prior run spent its early steps productively. Item 5 has never
been tried by any run and is the cheapest unexplored lever in this task.

---

## Operational notes that cost prior runs whole steps

- A worker polling a long job every ~2 minutes exhausts its **150-turn ceiling** before
  the job finishes. **6 of 38 steps died this way** across the runs that lacked this
  note; the runs that had it lost **0 of 30**. This is the single most reliably
  reproduced effect in any of these documents. Pass an explicit long timeout and sleep in
  long blocks -- the Bash default is 2 minutes.
- **Export and bank a scoreable fallback early.** A worker that hits the ceiling loses
  everything not already exported. Several runs shipped a previous step's config after a
  failed experiment: a null step beats a lost step.
- **Stopping early is legitimate when the budget is nearly spent** -- but it is not a
  reason to decline work with hours left. One worker cited "stopping early is
  legitimate" to skip the biggest available lever with five hours remaining; a worker in
  the next run reasoned the opposite way from the same words and landed a +75% step.
  Judge by time remaining, not by having something banked.
- **There is a WALL-CLOCK deadline as well as the turn ceiling, and it has already cost
  this task a step [1x].** `specdec-sglang-v1` lost step 6: the worker trained three
  candidates sequentially (~1.5h each), launched a fourth benchmark, noted its own hard
  deadline, then spent its final minutes in a `sleep 540` poll and was killed mid-call.
  Its benchmark log was 0 bytes, nothing was submitted, and no score program was created
  -- so step numbers silently stopped matching scoring-program numbers for the rest of
  the run. Its best measured candidate (1.9607) was on disk and would have scored.
    - **Budget the step, not just the turns.** Three trainings plus three ~40-minute
      benchmarks does not fit one step. Pick the candidate count from the time left.
    - **Never sleep away the margin.** If a result will not be back in time, submit the
      best candidate already measured.
    - **Re-export best-so-far after EVERY benchmark.** The lost worker's staged artifact
      was three hours old and predated all of its own measurements.
- **If you resubmit the incumbent, say so plainly in your writeup.** Four of
  `specdec-sglang-v1`'s nine scores were the same head resubmitted (md5 `b6fdeb95f828`,
  four times). That is the correct action when your experiment measures worse -- a null
  step beats a lost step -- but it reads to a later reader as a converged search rather
  than one measurement repeated. State which artifact you submitted and whether it is
  new.
- A detached training job that survives teardown can abort the **whole run** on a cgroup
  assertion. Kill children by PID, and check for duplicates first -- two workers killed a
  bash wrapper instead of the trainer.
