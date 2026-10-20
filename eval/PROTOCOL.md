# Evaluation protocol — aligning with the speculative-decoding literature

Written 2026-10-14, after an audit found our evaluation was internally sound but **not
comparable to any published result**.

## The problem with what we measured before

| | ours (runs 1-9) | literature standard |
|---|---|---|
| prompt set | **40 hand-written prompts**, no declared source, mean 98 chars | **MT-Bench**, 80 questions, 8 categories |
| turns | single | **two** (MT-Bench is multi-turn; context carried) |
| max_new_tokens | 128 | **1024** |
| temperature | 0 only | **0 and 1** both reported |
| batch size | 1 | 1 (we match) |
| metrics | speedup, tau | speedup, tau (we match) |
| quality check | **none** | task benchmark, or argued from the accept rule |

Our internal numbers are self-consistent and correctly measured — the audit confirmed
that — but a reviewer cannot place 1.971x next to a published figure, because the prompt
set, the generation length and the turn structure are all different. Shorter generations
in particular flatter speculative decoding less (prefill amortises differently) or more
(less chance to diverge), and nobody can tell which without matching the protocol.

## What the literature actually does

**"Lossless" is a theorem, not a measurement.** Leviathan et al. (2023) and Chen et al.
(2023) prove their rejection-sampling acceptance rule leaves the target model's output
distribution unchanged. EAGLE and successors inherit that framing. For greedy decoding it
reduces to our condition: accept only on argmax match. Nobody proves it bitwise, because
the claim is about the algorithm, not about floating-point.

**Validation is near-universally: speedup + mean acceptance length on MT-Bench.**
Usually with per-category breakdown, at temperature 0 and 1, batch 1.

**Quality benchmarks appear when a method is deliberately lossy.** Medusa's "typical
acceptance" relaxes the criterion with a threshold, so it cannot claim distribution
equality — and there task scores carry the argument. Our acceptance rule is exact, so we
are in the first camp, not the second.

**Byte-level agreement is essentially never reported.** Our measurement of it (10/30
streams identical, ~0.86% per-token bf16 divergence) is unusual, and is a property of
running both arms in the same framework with identical kernels. It does not indicate a
problem the literature has solved differently; it indicates most work never looks.

## The protocol we adopt

### Primary: MT-Bench speedup and acceptance length

- **Questions**: all 80 from `mt_bench_question.jsonl` (FastChat, the file the papers use)
- **Turns**: both, with turn 1's answer in context for turn 2
- **max_new_tokens**: 1024
- **Temperature**: 0 (primary) and 1 (secondary)
- **Batch size**: 1
- **Arms**: same SGLang build, same engine settings, only `speculative_*` differing —
  run as separate processes, as in `harness/benchmark.py`
- **Report**: overall and per-category wall-clock speedup, plus
  tau = total output tokens / total target forwards

### Secondary: quality parity

Our accept rule is exact, so this is confirmatory rather than load-bearing. Run one task
benchmark with and without the head and show scores match within its own noise. GSM8K
(exact-match) or HumanEval (pass@1) are the usual choices.

### State the denominator, always

Published "speedup" figures differ in what they divide by — vanilla HuggingFace for some,
the same serving framework for others. A 3x against HuggingFace and a 2x against SGLang
can describe the identical system. **We report against stock SGLang as primary and name
the denominator every time.** Our own history is the cautionary tale: the eager harness
produced a 4.33x that is 1.39x against a real server, and one head that scored 2.68x
there is 0.85x — slower than stock.

### Carry the measurement caveats

- The speculative arm resolves to about **±1.2%**; two configurations within 2% are tied.
- The non-speculative arm is far steadier (**0.07%** across 9 measurements).
- A first process after idle can be slow (observed 46.2 vs 52.4 tok/s) from JIT and graph
  capture. Warm the non-speculative path and use `--repeats 3`, never 2 — with two
  sweeps the median *is* the mean and cannot reject an outlier.

## Status

| step | state |
|---|---|
| MT-Bench questions fetched | done, 80 questions |
| harness written | `bench_mtbench.py` |
| re-evaluate champion head under protocol | **done** — T=0 **1.6637x** (tau 2.658), T=1 **1.5430x** (tau 2.441) |
| quality parity benchmark | **done** — GSM8K 92.2 vs 91.2, HumanEval 80.5 vs 79.9, combined McNemar p=0.109 |
| eager baseline under this protocol | **done** — 17.295 tok/s; token counts within 0.8% of SGLang's, so denominators are fair |
