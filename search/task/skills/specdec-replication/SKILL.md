---
name: specdec-replication
description: >-
  How to compare your own results against a published method or a public
  repository without fooling yourself. Covers the three-stage replication
  protocol, why a repository's own baseline number is often not comparable,
  recomputing speedups from raw output files, the decoding-rule confound, and
  the failure modes of running published research code. Read this before
  adopting a published technique or citing a published speedup.
---

# Comparing against published work

A published speedup is a ratio produced by a particular harness, on particular
hardware, under a particular decoding rule. Reproducing the number is a
research task in itself, and skipping it produces months of chasing a gap that
was never real.

## The three-stage protocol

When your own implementation of a published method underperforms its claim,
resolve it in this order. Do not skip a stage.

1. **Their checkpoint, their pipeline.** Run the authors' released weights in
   the authors' own harness. This establishes that the claim reproduces at all
   on your hardware, and gives you the number to aim at.
2. **Their pipeline, your training.** Train the component yourself with their
   data-generation script, their entry point and their hyper-parameters, then
   run it in their harness. A gap here is a training or data gap, isolated from
   everything else.
3. **Your setting.** Transfer to your own models, your own harness, your own
   decoding rule.

A gap at stage 3 is uninterpretable unless 1 and 2 passed. If you find yourself
explaining a stage-3 gap with a theory about data quality, stop and run stages
1 and 2.

## Their baseline is probably not your baseline

The most common reason a published speedup looks unreachable is the
**denominator**. A repository's baseline script often generates through its own
patched or instrumented model rather than a stock one, and is markedly slower
than stock generation on the same hardware. Every speedup computed against it is
inflated by that factor.

Always: measure stock generation yourself, on your card, and restate any
published ratio against your own baseline before comparing it to your work.
Record which baseline produced every number you report.

## The decoding rule is a confound, not a detail

Methods are often reported under a different acceptance rule, temperature, or
candidate shape than the one you are running. The same trained component can
appear far weaker simply because it is being evaluated under a stricter rule —
a decoding-rule difference can easily outweigh a data-quality difference.

When comparing, hold these constant and vary one at a time: the acceptance rule,
the temperature, the candidate shape (chain versus tree, and its size), and the
component's training data. Attributing a gap to the last of these before the
first three have been equalised is the classic mistake.

## Recompute the ratio from raw outputs

Evaluation scripts frequently report token counts that are zero, or wall times
that include model loading, or both. Where the harness writes per-answer records
(generated ids, step indices, timings), recompute tokens/second yourself from
those records, and sanity-check the token count against a re-tokenisation of the
produced text. A ratio you did not compute is a ratio you cannot defend.

## Published repositories do not run as shipped

Budget time for this and check the artifact path end to end *before* spending
GPU hours. Failure modes seen repeatedly:

- A default flag selects a newer model variant, so the released checkpoint for
  the published version will not load.
- The trainer writes one checkpoint format and the loader reads another.
- The training entry point passes raw data to a preprocessing step that
  requires an already-formatted dataset.
- An unconditional import inside a post-training save block fails and destroys
  the output of a completed multi-hour run — while every load-side check you ran
  beforehand still passed.

The lesson from the last one: smoke-test against the **production output path**,
not just the load path, and make the smoke test write and then re-read an
artifact the same way the real run will.

## Record what you patched

When you must modify a published repository to run at all, keep the patches
minimal, mark each one in the source, and list them in the write-up: model
name, local data paths, and precision changes are the usual set. A result from
a patched repository with unlisted patches is not a replication.
