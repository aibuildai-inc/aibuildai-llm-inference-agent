---
name: specdec-measurement
description: >-
  How to measure decoding throughput at batch size 1 so the number means
  something. Covers GPU pinning, CUDA synchronisation, warm-up, what the
  baseline must be, the exact definitions of alpha and tau, why a rate measured
  at one draft length does not predict another, and how a test harness can
  report a false failure. Read this before timing anything or believing a
  speedup.
---

# Measuring decoding speed honestly

A speedup is a ratio of two measurements. Most wrong speedups come from the
denominator, the clock, or the bookkeeping — not from the decoding algorithm.

## Pin exactly one GPU

Load every model with an explicit single device. Never `device_map="auto"`.
Sharding a model across cards changes where time goes and invalidates latency
measurement entirely. One card, one process, batch size 1.

Within a comparison, the baseline and the candidate must run **on the same
card**, ideally back to back in the same process. A ratio between two different
cards, or between a quiet machine and a busy one, measures the machine.

## Synchronise around every timed region

CUDA work is asynchronous. Without `torch.cuda.synchronize()` immediately
before you start the clock and immediately before you stop it, you are timing
how fast Python queued the work, not how long the GPU took. This single mistake
can make a method look several times faster than it is.

Always run one short generation first and throw the result away. The first call
pays for kernel autotuning, lazy module initialisation and allocator warm-up.

## The baseline must be the same algorithm every time

Fix one baseline implementation and reuse it for every candidate. If each
candidate brings its own baseline, you are comparing candidates against their
own optimism. Record, next to every ratio, *which* baseline produced the
denominator — a baseline that itself runs through a patched or instrumented
model can be much slower than a stock one, which silently inflates every
speedup computed against it.

## Interleave the two sides, do not run them in phases

Timing every baseline prompt and then every method prompt looks equivalent to
interleaving them. It is not, on any machine you share. Load that arrives
between the two phases lands entirely on one side of the ratio: a baseline
measured while the host was busy inflates the speedup, and a method measured
while busy deflates it.

Instead, for each input, time both sides back to back, and **alternate which one
goes first** so any within-pair ordering effect cancels. Repeat the whole sweep
and report the **median with its spread**, not a single pass.

This is not hypothetical. In one run the phase-based protocol measured a
baseline at half the machine's normal speed during one candidate's scoring
window. That candidate's ratio was inflated by about 14%, which was enough to
beat a genuinely better candidate, and the search then selected the worse one.
Re-measured interleaved, the ranking changed.

A useful cross-check: a load-independent statistic (tokens committed per
expensive pass) should reproduce EXACTLY between runs, because it is
deterministic for fixed inputs. If it reproduces and the ratio moves, the
machine moved, not the method.

## Definitions that are easy to get subtly wrong

- **alpha** = accepted draft tokens / **evaluated** draft tokens.
  Not accepted / *drafted*. Tokens that were proposed but never verified —
  because an earlier token in the same cycle was rejected, or because a branch
  was pruned — are not in the denominator.
- **tau** = emitted tokens / decoding cycles. The average number of tokens the
  method commits per target forward pass. This is the load-independent measure
  of a method's quality; throughput also moves with the machine.
- Assert that **every cycle emits at least one token**. A cycle that emits zero
  is either a bug or a silent infinite loop.

Report alpha, tau and tokens/second together. Two of them cannot be inferred
from the third, and a change in throughput with tau unchanged means the machine
moved, not the method.

## Never extrapolate a rate

Per-token verification cost measured with a handful of draft tokens does not
predict the cost of verifying a large candidate tree. The cost per token can
grow by an order of magnitude between a short chain and a wide tree, because a
different kernel regime takes over. Measure the configuration you are about to
rely on, at the size you will use it.

The same applies to fitting a simple linear model to prefill timings and using
it to reason about the cached decode loop. They are different shapes.

## A test's exit code can lie

A crash in summary or reporting code can exit non-zero while every case it
measured actually passed — and the reverse is also possible. Count the evidence
rows before believing a verdict: how many prompts were compared, how many
matched. Treat the printed evidence as the result and the exit code as a hint.

Related: never pattern-match a process list against a string that appears in
your own command. A search for a running job will match the search itself.
Use process ids, or break the literal.
