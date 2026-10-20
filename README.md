# AIBuildAI-SpecDec

A transferable recipe for training speculative-decoding draft heads, with the
harness that measures them and the evidence for each design choice.

The reference result below is for one target model; the recipe itself is not
specific to it (see **Scope**). Model and method names appear in the results and
the recipe where they carry meaning, not in the project's structure.

```
target: Qwen/Qwen2.5-14B-Instruct      host: 1x A100-SXM4-80GB
MT-Bench, T=0, 80 questions x 2 turns, 1024 new tokens, batch 1

  HF generate(), eager          16.783 tok/s
  stock SGLang, no speculation  52.214 tok/s
  SGLang + this head           104.892 tok/s     2.0089x      tau 3.2061

  quality (paired, GSM8K 500 + HumanEval 164):
    7 : 7 discordant, McNemar exact p = 1.000
```

Full numbers and their caveats: **[RESULTS.md](RESULTS.md)**.
How the head was built: **[RECIPE.md](RECIPE.md)**.

## What is and is not claimed

**Claimed.** On this target, this host and these benchmarks, the head roughly
doubles SGLang's batch-1 decoding throughput, and we could not detect a quality
difference at n=664 paired items.

**Not claimed.** Not bit-identical output. Greedy verification guarantees the
verifier emits the target's own argmax, so a proposer cannot change which token
is chosen -- but bf16 ties round differently between a cached single-token
decode and a batched tree verify, and token streams do diverge (10-15 of 30
streams matched exactly on a probe; ~0.86% of tokens differ). The honest phrasing
is "output distribution preserved under the verifier's own logits; token streams
not bit-identical".

**Not claimed.** That quality is provably unchanged. 7:7 discordant means *no
detectable difference at this sample size*, not zero difference. Resolving a
1-point difference at p<0.05 needs roughly 800 items per arm.

## Requirements

- 1 GPU with >= 60 GB free (measured on A100-SXM4-80GB)
- `Qwen/Qwen2.5-14B-Instruct` (~28 GB) reachable by `huggingface_hub`
- The exact versions in `requirements-frozen.txt`. The ones that matter:
  `torch 2.13.0+cu130`, `transformers 5.12.1`, `sglang 0.5.21`,
  `flashinfer 0.6.18`, `accelerate 1.15.0`
- **A one-line SGLang patch** (see below). Without it the head computes a
  different function and the measurement is meaningless.

## Quickstart

```bash
# 1. patch SGLang -- NOT optional
./tools/patch_sglang.sh

# 2. fetch the head weights (see heads/README.md) to heads/draft_head.pt

# 3. check prerequisites only (~2 min)
export HF_HOME="$HOME/hf-cache"          # needs ~28 GB for the target
export SPECDEC_HEAD="$PWD/heads/draft_head.pt"
./eval/run_eval.sh --check

# 4. run it
./eval/run_eval.sh --phase mtbench   # ~70 min, the headline number
./eval/run_eval.sh --phase quality   # ~90 min, GSM8K + HumanEval
./eval/run_eval.sh                   # all phases, ~4 h
```

Compare your output with `eval/REFERENCE.md`. Anything within ~2% is a match --
that is the instrument's resolution, not slack.

### The SGLang patch

`tools/patch_sglang.sh` edits `sglang/srt/models/qwen2_eagle.py` in your
installed SGLang. The stock file deletes the draft layer's `input_layernorm`,
which this head needs: with the layernorm removed the head computes a different
function (relative error 1.84 against the reference implementation, versus
7.7e-17 with the patch). The script matches on structure rather than line
numbers, is idempotent, and refuses to apply twice.

`eval/run_eval.sh --check` verifies the patch is present and will not run
without it.

## Layout

```
train/       build a head of your own (the code that produced this one)
  recipe.yaml      the config that drives prepare/train/eval/export
  prepare.py       harvest teacher features from the target
  head_model.py    the trainable head
  train.py         multi-step (unroll) feature regression
  eval.py          advisory dev acceptance
  export.py        -> head.pt + head_config.json
  corpus/          the incremental corpus augmenters (read train/README.md)
task/        the measuring instrument -- do not edit
  models.json      target, engine settings, the draft tree, tunable bounds
  prompts.json     the 30 fixed prompts, pre-tokenised
  harness/         make_draft.py, benchmark.py, check_draft.py, _engine_arm.py
  reference_head.py    reference head implementation, validated against the target
eval/        literature-protocol evaluation
  run_eval.sh      MT-Bench (T=0/T=1) + HF eager + GSM8K + HumanEval
  PROTOCOL.md      what is measured and why it is measured that way
  REFERENCE.md     our numbers, to diff yours against
docs/        the method knowledge, as standalone notes
  recipe.md measurement.md exactness.md methods.md performance-model.md
tools/       patch_sglang.sh
heads/       how to obtain the weights
search/      re-run the autonomous search that found the head
  task/            the task folder the framework reads as data_root
  task/skills/     the seven knowledge documents the agents read
  run_config.yaml  the configuration that produced this head
  score.reference.py   the scorer the winning run wrote (a re-run writes its own)
```

`task/` is the instrument: `models.json`, `prompts.json` and `harness/` decide
whether a number means anything. Treat them as read-only. If you change them you
are measuring something else, and `eval/REFERENCE.md` no longer applies.

## Four things you can do with this

1. **Verify the published number.** `heads/` + `eval/` + `task/`. No training.
2. **Train your own head.** `train/` has the code that produced this one, driven
   by `train/recipe.yaml`. See `train/README.md`.
3. **Reapply the method elsewhere.** `RECIPE.md` states what moved the number and
   what did not, with the measurements, so the recipe can be taken to another
   target.

None of the three needs the agent framework. `RECIPE.md` and
`train/recipe.yaml` capture what the search converged on.

4. **Re-run the search itself.** `search/` holds the task folder, the seven
   knowledge documents the agents read, the run configuration that produced this
   head, and the reference scorer; `framework/` is the agent framework that runs
   them, as a source snapshot at the revision that produced the head. See
   `search/README.md` to launch and `framework/EXCLUDED.md` for what the
   snapshot leaves out.

   A re-run will not retrace the published trajectory -- each step is an agent
   forming its own hypothesis, and a run writes its own scorer from the task
   contract. What should reproduce is the baseline and the end point.

## Scope

Measured on one target, one GPU model, one SGLang version. The recipe has been
applied to a second, much larger target (a 123B dense model at tensor-parallel
4, reaching 2.12x on the same internal harness), which suggests it is not
specific to Qwen2.5-14B -- but that result is not part of this repository and
was not evaluated on MT-Bench.

## License

This repository's code and documentation are MIT licensed (`LICENSE`).

The three bundled evaluation datasets and the target model keep their own terms,
and the draft head weights are derived from self-distilled outputs of the target
model -- see `DATA_AND_LICENSES.md` before redistributing any of those.
