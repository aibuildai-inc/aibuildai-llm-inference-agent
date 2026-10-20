---
name: specdec-performance-model
description: >-
  Where the time actually goes in single-stream decoding, and what that implies
  for which speculative-decoding ideas can pay off. Covers host-dispatch-bound
  execution at batch size 1, why drafting cost rather than verification cost
  caps the speedup, how target size decides whether any method wins, and why
  ranking draft/target pairs by speedup can invert. Read this before choosing
  what to optimise.
---

# Where the time goes at batch size 1

Before optimising, find out what the machine is actually doing. At batch size 1
the answer is usually *not* what a FLOP or memory-bandwidth argument predicts.

## Single-stream decoding is often host-dispatch-bound

A small model's forward pass can cost many times its memory-bandwidth floor,
because a single decode step issues a very large number of small GPU
operations and the host cannot launch them fast enough to keep the device busy.
The device is idle between kernels.

Two consequences:

1. **A small draft model is not cheap in the way parameter count suggests.**
   Its cost is closer to a fixed per-pass overhead set by its layer count than
   to a fraction of the target's cost. The ratio of draft cost to target cost
   can be an order of magnitude worse than a FLOP ratio implies.
2. **Verification is nearly free per extra token, up to a point.** Verifying a
   few extra positions in one forward pass adds very little, because that pass
   is dominated by the same fixed overhead. This is the premise the whole method
   rests on, and it is usually true — but only in the regime you measured it in.

So the cap on speedup is normally the **proposer's cost**, not verification.
That is the argument for proposers that ride on a forward pass the target
already makes.

## Measure the decomposition first

Establish, for your machine and models: the cost of one target decode step, the
cost of one proposer step, and the marginal cost of verifying n extra positions
for the n values you intend to use. From these you can predict a method's
speedup from its tau before implementing it — and you can rule out ideas that
cannot pay off.

Do this with the same discipline as any other timing: pinned card, warm-up,
synchronised clocks. See `specdec-measurement`.

## Target size decides whether anything wins at all

The same proposer, with the same acceptance rate, can lose on a small target and
win on a large one. Acceptance is a property of the model pair's agreement and
barely moves with target size; what moves is whether the proposer's fixed cost
amortises against a slower target step. Below some target size, every
speculative method is slower than plain decoding.

Therefore: **do not rank proposer/target pairs by speedup while dispatch-bound.**
Dispatch cost scales with layer count while compute cost scales with weights, so
the ranking you measure on one target can invert on another. Rank by acceptance
rate and tau, which are properties of the pair, and treat speedup as a property
of the pair *and* the machine.

## What this means for effort allocation

- A better proposer that costs another full model pass often buys nothing.
- A slightly worse proposer that costs nothing extra often buys a lot.
- Widening the candidate set (trees) exploits cheap verification directly, and
  is usually the highest-leverage change once the proposer is cheap.
- Reducing per-pass dispatch overhead (fewer, larger operations; capturing the
  decode step; avoiding per-step Python work) attacks the actual bottleneck and
  helps the baseline and the method alike — which means it can *reduce* the
  reported speedup while making the system faster. Report both absolute
  throughput and the ratio.
