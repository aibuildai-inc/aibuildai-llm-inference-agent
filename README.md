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
- a C++ host compiler `nvcc` accepts -- flashinfer JIT-compiles kernels on first
  use. A system `g++` works; if the machine has none,
  `conda install -c conda-forge gxx_linux-64=12.4 gcc_linux-64=12.4` into the
  serving environment; the harness finds the prefixed compiler and points nvcc
  at it, so no symlinks are needed. Without a host compiler the first arm fails
  with `nvcc fatal : Failed to preprocess host compiler properties`
- The exact versions in `requirements-frozen.txt`. The ones that matter:
  `torch 2.13.0+cu130`, `transformers 5.12.1`, `sglang 0.5.21`,
  `flashinfer 0.6.18`, `accelerate 1.15.0`
- **A one-line SGLang patch** (see below). Without it the head computes a
  different function and the measurement is meaningless.


## Three ways in

Pick the one you want; they are independent.

| | needs | gives you |
|---|---|---|
| **A** verify the published number | the head weights, no training | the 2.0089x reproduced, or a number to compare |
| **B** train your own head | a GPU and hours | a head of your own from this recipe |
| **C** re-run the end-to-end search | a GPU, 48 h, **an AIBuildAI account** | the agent builds a head from the task contract |

## Quickstart A — verify the published number

```bash
git clone https://github.com/importZL/AIBuildAI-SpecDec.git
cd AIBuildAI-SpecDec

# 1. patch SGLang -- NOT optional
./tools/patch_sglang.sh

# 2. fetch the head weights from release v1.0.0 -> heads/draft_head.pt
#    (see heads/README.md; needed only for eval/, not for search/ or train/)

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

## Quickstart B — train your own head

    cd train
    python prepare.py          # harvest teacher features (hours; writes data/)
    python train.py            # -> runs/<output_dir>/head.pt
    python eval.py             # advisory dev acceptance
    python export.py ../out    # -> ../out/full/{head.pt, head_config.json}

Everything is driven by `train/recipe.yaml`. Read `train/README.md` first: the
published corpus was grown incrementally across search steps rather than built
by one `prepare.py` run, and the file explains which increments did what.

## Quickstart C — re-run the end-to-end search

This is the agent that *found* the head, not the head. It needs no trained head
and no weights to start; it builds one from the task contract.

    # 0. `aibuildai run` is gated on an AIBuildAI account: the framework fetches a
    #    signed lease and fails closed without one. A and B above need no account;
    #    only C does. See search/README.md, "If you have no AIBuildAI account".

    # 1. install the framework (the order matters -- packages/ first)
    python3 -m venv ~/aibuildai-venv && . ~/aibuildai-venv/bin/activate
    export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=21c3972447b7b51b0c86bb53f4e2552bac2526a4
    cd framework
    pip install -c requirements.lock -e ./packages/execution
    pip install -c requirements.lock -e ./packages/conversation
    pip install -c requirements.lock -e .
    cd ..

    # 2. confirm it installed as the revision that produced the result
    aibuildai -V                 # AIBuildAI science-2026-08-04+439.gd4c0692e7.dirty
                                 # 0.0.0+unknown here means _version.py did not survive the clone

    # 3. patch SGLang -- NOT optional
    ./tools/patch_sglang.sh

    # 4. sign in to the product (separate from the Claude login that supplies
    #    model access), then launch (10 steps, 48 h budget; ours used 36 h)
    aibuildai login
    ./search/launch.sh [playground-dir]      # default ~/aibuildai-playground

**Read `search/README.md` before launching.** It covers what the repository
cannot supply -- a Claude Code login (no API key needed), a conda environment
holding torch/SGLang/flashinfer, one GPU with >= 60 GB free, ~200 GB of disk --
and two things that will otherwise surprise you:

* a re-run writes its **own** scorer from the task contract, so two runs of the
  same search can be scored differently. `search/score.reference.py` is the one
  the published run used, to diff against or to pin;
* a re-run will **not** retrace the published trajectory. Each step is an agent
  forming its own hypothesis. What should reproduce is the stock baseline
  (52.30--52.43 tok/s across all ten scored steps) and roughly the end point.


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

## Reapplying the method elsewhere

`RECIPE.md` states what moved the number and what did not, with the
measurements, so the recipe can be taken to another target. `docs/` holds the
same knowledge as standalone notes.

Quickstarts A and B need no agent framework at all -- `RECIPE.md` and
`train/recipe.yaml` capture what the search converged on, so the recipe is
usable without ever running an agent. Only C needs `framework/`.

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
