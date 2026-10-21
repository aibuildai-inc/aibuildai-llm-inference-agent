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
- `conda` (miniforge is fine). A bare GPU image usually has neither conda nor
  a Python the framework accepts (`>=3.11,<3.14`); Quickstart A step 1 installs
  it in three commands
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
| **C** re-run the end-to-end search | a GPU, and either an AIBuildAI login or the source-build skip flag; 5--6 h per step (17 h to verify, 36 h for all ten) | the agent builds a head from the task contract |

## Quickstart A — verify the published number

These are the commands as run on a second machine, in this order.

```bash
git clone https://github.com/importZL/AIBuildAI-SpecDec.git
cd AIBuildAI-SpecDec

# 1. conda, if the machine has none. Everything below lives in conda
#    environments; a bare GPU image usually ships neither conda nor a
#    Python new enough for the framework (Ubuntu 22.04 ships 3.10).
curl -L -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash && exec bash
#    `exec bash` replaces the shell, so paste the rest AFTER it returns

# 2. the serving environment: torch + SGLang + flashinfer for YOUR CUDA, plus a
#    host compiler for flashinfer's JIT. The harness reads CXX/CC from this
#    interpreter's own bin/, so install the compiler INTO this env.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
#    the CUDA build below must match that driver; cu130 needs driver >= 580
conda create -y -n sglang-baseline python=3.12
conda activate sglang-baseline
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
pip install sglang==0.5.21 flashinfer-python==0.6.18 transformers==5.12.1 \
  accelerate==1.15.0 numpy safetensors pyyaml huggingface-hub
conda install -y -c conda-forge gxx_linux-64=12.4 gcc_linux-64=12.4

# CHECK the stack before going further
python -c "import torch, sglang, flashinfer, transformers; \
print(torch.__version__, sglang.__version__, flashinfer.__version__, transformers.__version__)"
python -c "import torch; print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
x86_64-conda-linux-gnu-g++ --version | head -1
#    nvcc must be told WHICH host compiler: it looks for a plain `g++` on PATH,
#    conda-forge ships only the prefixed name, and a GPU image often has no
#    system g++ -- that combination fails with `cannot execute 'cc1plus'`.
#    The harness passes -ccbin for every real run; pass it here too.
echo 'int main(){return 0;}' > /tmp/t.cu
nvcc -ccbin "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" /tmp/t.cu -o /tmp/tcu \
  && echo "nvcc+host OK"

# 3. patch SGLang -- NOT optional, and it must be the SGLang you just installed
./tools/patch_sglang.sh
./tools/patch_sglang.sh                  # idempotent: must say "already patched"

# 4. fetch the head weights (public release, no token needed)
curl -L -o heads/draft_head.pt \
  https://github.com/importZL/AIBuildAI-SpecDec/releases/download/v1.0.0/draft_head.pt
sha256sum heads/draft_head.pt
# expect 7a744703b50f49f9d5cb6ad04256e6a5ee81746b70d4426e3bd900980e371343

# 5. check prerequisites only (~2 min)
export HF_HOME="$HOME/hf-cache"          # needs ~28 GB for the target
export TMPDIR=/var/tmp                   # if /tmp is small
PYTHON="$(which python)" ./eval/run_eval.sh --check

# 6. run it -- 70 min+, so do it inside tmux (an ssh drop otherwise kills it)
tmux new -s eval

#    INSIDE tmux you get a fresh shell: the env and the exports above are NOT
#    inherited. Set them up again before running anything.
conda activate sglang-baseline
cd ~/AIBuildAI-SpecDec
export HF_HOME="$HOME/hf-cache"
export TMPDIR=/var/tmp

PYTHON="$(which python)" ./eval/run_eval.sh --phase mtbench   # ~70 min, the headline number
PYTHON="$(which python)" ./eval/run_eval.sh --phase quality   # ~90 min, GSM8K + HumanEval
PYTHON="$(which python)" ./eval/run_eval.sh                   # all phases, ~4 h
```

Detach with `Ctrl-b d`, come back with `tmux attach -t eval`. If you skip the
re-activation, `which python` resolves to the system interpreter and the run
fails on the first import.

`--check` should print the versions, `sglang patch: OK`, `draft module :
imports`, `registry : Qwen2ForCausalLMEagle registered`, a GPU with >= 60 GB
free, and all three data files found.

`run_eval.sh` ends by running `summarise.py` itself; to re-print the comparison
later without re-measuring:

```bash
python eval/summarise.py --results eval/results --reference eval/REFERENCE.md
```

Compare your output with `eval/REFERENCE.md`. Anything within ~2% is a match --
that is the instrument's resolution, not slack. **A cold clone on a second
machine measured 1.9943x against the published 2.0089x -- 0.73% apart, with tau
identical to four decimals.**

## Quickstart B — train your own head

    cd train
    python prepare.py          # harvest teacher features (hours; writes data/)
    python train.py            # -> runs/<output_dir>/head.pt
    python eval.py             # advisory dev acceptance
    python export.py ../out    # -> ../out/full/{head.pt, head_config.json}

Everything is driven by `train/recipe.yaml`.

**The published head needs more than these four commands.** `recipe.yaml` names
`train_v4.pt`, the step-10 corpus, which exists only after the incremental chain
in `train/corpus/` -- `prepare.py` writes `train.pt`. Run the four commands above
and `train.py` says so and trains on `train.pt` instead, which yields a valid
head but not the published one. To reproduce the published corpus:

    python prepare.py                       # -> data/train.pt
    python corpus/gen_chat_aug.py           # -> data/chat_aug.pt
    python corpus/merge_corpus.py           # -> data/train_v2.pt
    python corpus/gen_rc_aug.py             # -> data/rc_aug.pt
    python corpus/merge_corpus_v3.py        # -> data/train_v3.pt
    python corpus/gen_rc_aug2.py            # -> data/rc_aug2.pt
    python corpus/merge_corpus_v4.py        # -> data/train_v4.pt

Both paths have been run from a clean clone: the four-command path trains on
`train.pt` and `export.py` hands `make_draft.py` a head that packages into a
16-tensor draft directory SGLang loads.

`train/README.md` explains what each increment added and why. The chain exists
because regenerating 6M tokens of teacher features costs hours; it was grown
across search steps rather than built in one pass.

## Quickstart C — re-run the end-to-end search

This is the agent that *found* the head, not the head. It needs no trained head
and no weights to start; it builds one from the task contract.

    # 0. `aibuildai run` is gated: it fetches a signed lease and fails closed.
    #    From source you can skip that verdict; a frozen binary cannot.
    export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1      # or: aibuildai login

    # 1. install the framework (the order matters -- packages/ first).
    #    Assumes conda exists; if not, install miniforge first (Quickstart A
    #    step 1). Use conda, not venv: the framework needs Python >= 3.11,
    #    Ubuntu 22.04 ships 3.10, and `python3 -m venv` fails without
    #    python3-venv.
    conda create -y -n aibuildai python=3.12
    conda activate aibuildai

    #    The toolchain goes INTO this env, not into base or sglang-baseline:
    #    the framework builds PostgreSQL 18.6 from source on first run, cannot
    #    use a system one, and looks for bison/flex/make/gcc in the active env.
    #    Running this before `conda activate aibuildai` installs it in the
    #    wrong place and the build fails.
    conda install -y -c conda-forge bison flex make gxx_linux-64=12.4 gcc_linux-64=12.4

    export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=21c3972447b7b51b0c86bb53f4e2552bac2526a4
    cd framework
    pip install -c requirements.lock -e ./packages/execution
    pip install -c requirements.lock -e ./packages/conversation
    pip install -c requirements.lock -e .
    cd ..

    # 2. confirm it installed as the revision that produced the result
    aibuildai -V                 # AIBuildAI science-2026-08-04+439.gd4c0692e7.dirty
                                 # 0.0.0+unknown here means _version.py did not survive the clone

    # 3. patch SGLang -- NOT optional, and patch the env the RUN will use:
    #    run_config names `conda_env: sglang-baseline`, which must already exist
    #    (Quickstart A step 1). launch.sh verifies it before starting.
    conda activate sglang-baseline && ./tools/patch_sglang.sh
    conda activate aibuildai

    # 4. launch (10 steps, 48 h budget; the published run used 36 h).
    #    launch.sh ends in `exec aibuildai run`, i.e. a FOREGROUND process, so
    #    run it inside tmux or an ssh drop kills the search.
    tmux new -s search
    #    inside tmux the shell is fresh -- nothing above is inherited:
    conda activate aibuildai
    cd ~/AIBuildAI-SpecDec
    export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=21c3972447b7b51b0c86bb53f4e2552bac2526a4
    export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1       # if you used the skip flag
    export HF_HOME="$HOME/hf-cache" TMPDIR=/var/tmp

    ./search/launch.sh [playground-dir]      # default ~/aibuildai-playground
    #    detach with Ctrl-b d, come back with `tmux attach -t search`

    # 5. the workspace UI. launch.sh prints
    #      AIBuildAI Workspace: http://127.0.0.1:<port>/run/<id>
    #    That loopback address is the REMOTE machine's. To open it on your
    #    laptop, forward the port first (the port changes every run):
    #      ssh -N -L <port>:127.0.0.1:<port> user@host
    #    then open the printed URL locally, unchanged.

    # 5. watch it
    ./search/watch_run.sh ~/aibuildai-playground
    for f in $(find ~/aibuildai-playground -name .program-result.json \
                    -path '*score_program*' | sort -V); do
      python3 -c "import json;o=json.load(open('$f'))['output'];print(round(o['score'],4), o['components'])"
    done
    cat ~/aibuildai-playground/*/workspace/search_1/worker_*/state/artifacts/README.md

A step is sound when `gate_passed` is 1.0 **and** its baseline sits in the
52.2--52.5 tok/s band; a baseline outside that band means the instrument moved
and the ratio is not comparable.

**What a from-scratch run produced**, on one A100, starting from this repo:

| step | what the agent changed | score | tau | baseline |
|---|---|---|---|---|
| 1 | built the pipeline; tree sweep, width, then on-policy data (1.41M tokens) | 1.8671 | 3.028 | 52.457 |
| 2 | on-policy volume 1.41M -> 2.49M | 1.9612 | 3.194 | 52.434 |
| 3 | on-policy volume 2.49M -> 3.71M | 2.0068 | 3.279 | 52.460 |

About 5--6 hours per step. **Three steps is enough to verify the machinery** --
it crosses 2.0x and the baselines agree to 0.05%; the remaining seven chase the
end point. If the machine dies mid-run, the run does not resume: the scored
steps and their written records survive on disk, the in-flight step does not.
Read the records, then launch a fresh run rather than restarting the old one.

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
without it. The script keeps the original beside the patched file, so to revert:

```bash
F=$CONDA_PREFIX/lib/python3.12/site-packages/sglang/srt/models/qwen2_eagle.py
cp "$F.orig" "$F"
```

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
