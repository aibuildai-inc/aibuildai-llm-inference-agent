# Re-running the search

This directory holds everything needed to re-run the autonomous 10-step search
that found the head in `../heads/`, **except the agent framework itself**. See
"What is missing" below.

The rest of this repository lets you verify the published number, retrain the
head, or reapply the recipe by hand. This directory is for the different and
larger claim: that the *search* reproduces.

## What is here

| path | what it is |
|---|---|
| `task/` | the task folder, verbatim, in the layout the framework consumes as `data_root` |
| `task/README.md` | the task statement and the `score(output_dir)` contract |
| `task/skills/` | the seven knowledge documents every worker reads (measurement, exactness, methods, performance model, replication, recipe, shared-host hygiene) |
| `task/harness/` | the instrument: packaging, gate, benchmark |
| `task/reference/eagle_head.py` | the reference proposer implementation the gate checks against |
| `run_config.yaml` | the configuration that produced the published head |
| `score.reference.py` | the scorer the winning run's setup agent wrote (see below) |

`task/` here is the complete original-layout folder. The repository's top-level
`task/` is the same instrument reorganised for use by hand; this one is what the
framework reads.

## The framework

`../framework/` is a source snapshot of the agent framework that runs this
configuration, at the revision that produced the published head
(`science-2026-08-04+439.gd4c0692e7.dirty`). It is MIT licensed like the rest of
the repository.

Read `../framework/EXCLUDED.md` first. It lists what the snapshot leaves out --
git history, the two submodules (neither is loaded at runtime), and a 42.8 MB
prebuilt Go proxy you need only if you route to GPT models -- and the four local
modifications the tree carried, none of which is on this search's code path.

## The scorer is written per run -- read this before comparing two runs

A run does **not** load `score.reference.py`. The setup agent implements the
contract in `task/README.md` from scratch, so a fresh search writes its own
`private/score.py`.

This is deliberate -- the task is specified by its contract, not by an
implementation -- and it is also the largest source of drift between two runs of
nominally the same search. Two scorers that both satisfy the contract can still
differ in sweep count, warm-up, or which failures they treat as gate failures.

So when a re-run's numbers differ from ours, diff the scorers first:

    diff <run>/private/score.py search/score.reference.py

To hold scoring fixed while something else varies, copy `score.reference.py`
over `<run>/private/score.py` after SETUP finishes and before the first step.

## Install and launch

Verified on a clean Python 3.12 virtualenv. The three pip commands are the order
`framework/release/guide.md` documents; the env var is the sanctioned way to
install the bundled proxy without rebuilding it from Go source.

    python3 -m venv ~/aibuildai-venv && . ~/aibuildai-venv/bin/activate
    export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=21c3972447b7b51b0c86bb53f4e2552bac2526a4
    cd framework
    pip install -c requirements.lock -e ./packages/execution
    pip install -c requirements.lock -e ./packages/conversation
    pip install -c requirements.lock -e .
    cd ..
    ./search/launch.sh [playground-dir]          # default ~/aibuildai-playground

Without that env var the third-party build hook refuses with *"The build
requires the exact CLIProxyAPI source commit"*, because building the proxy from
source needs the submodule at its pinned commit and Go 1.26+. The prebuilt
binary ships here with its `identity.json` and LICENSE, so the hook verifies the
bytes and skips the Go build.

`launch.sh` substitutes real paths into `run_config.yaml`, checks the
preconditions, and launches. It refuses a playground inside the repository,
because a run writes tens of GB there.

`run_config.yaml` carries `__REPO__` and `__PLAYGROUND__` placeholders rather
than environment variables **on purpose**: the framework does not expand `${VAR}`
in path fields -- `infra/fs/pathing.normalize_path()` is expanduser plus
absolutize, nothing more -- so a `${VAR}` path would be taken literally and
joined to the current directory. Substitute, do not interpolate.

## What the machine must supply

| | |
|---|---|
| authorisation for `aibuildai run` | Either sign in (`aibuildai login`) **or**, on a source build like this one, `export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1`. `cli_impl.py` honours that flag only when `is_frozen()` is false, so it works from source and not from a shipped binary. Without one or the other the run stops at *"You are not signed in"* |
| model access | **a Claude Code login is enough -- no Anthropic API key is required** (`framework/release/guide.md`). An `ANTHROPIC_API_KEY` works too. This is a *separate* login from `aibuildai login`: the first supplies model tokens, the second authorises the run |
| internet | the framework self-provisions its own tools on first run -- ruff, tectonic and caddy as prebuilt binaries, and **PostgreSQL 18.6 built from official source** -- plus the ~28 GB target model |
| a build toolchain in the FRAMEWORK environment | that PostgreSQL build needs `gcc`, `make`, `bison` and `flex`. It cannot reuse a system PostgreSQL: `infra/owned_tools.py` states "Neither mode uses executables from PATH". It configures `--without-readline --without-zlib --without-icu`, so no dev headers are required. Without them the launch dies in `configure` with e.g. `configure: error: bison not found`. No sudo needed: `conda install -y -c conda-forge bison flex make gxx_linux-64=12.4 gcc_linux-64=12.4` into the environment you launch from |
| conda | `run.conda_env: sglang-baseline` names a host environment cloned into the run's program environment; cloning needs conda initialised in the launching shell. It must hold torch, SGLang and flashinfer (`../requirements.txt`). If no such environment exists the framework creates a fresh Python 3.11 one instead, and the run's own agents would then have to install the serving stack -- slower, and not what the published run did |
| SGLang patch | `../tools/patch_sglang.sh`, applied to whichever SGLang the run will use |
| GPU | one card with >= 60 GB free |
| disk | ~200 GB for the target, the teacher-feature corpus and per-step heads |
| time | 48 h budget; the published run used 36 h |

## What has been verified, and what has not

Verified on this machine, from a fresh copy of this repository:

* all three packages install into a clean Python 3.12 virtualenv;
* the `aibuildai` CLI appears and reports version
  `2026.8.4+439.gd4c0692e7.dirty`;
* `launch.sh`'s substitution leaves no placeholder, and the generated config
  passes `AgentConfig.model_validate`;
* `search.kind: autoresearch` resolves to `AutoResearchSearch` and its own
  `validate_config` accepts the config;
* all seven skills are on disk with valid frontmatter -- a missing `name:` field
  refuses the launch outright;
* the task folder holds everything SETUP reads.

Not verified: an actual run. That needs a free GPU, a model login and 36 hours,
so the end-to-end search has not been re-executed from this copy.


## If you have no AIBuildAI account

You can still run the search: this is a source build, so
`AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1` skips the licence verdict
(`cli_impl.py`, guarded by `not is_frozen()`). `subscription/run_grant.py` is
otherwise a hard gate -- a signed lease from AIBuildAI's server, verified
against a compiled-in public key, failing closed with no offline-only mode -- so
that flag is the sanctioned path for running from source.

Everything else in this repository needs no account and no flag:

* **Quickstart A** -- verify the published number with `eval/` and the released
  head. No framework, no account.
* **Quickstart B** -- train your own head with `train/`. No framework, no
  account.
* `RECIPE.md` and `docs/` state what moved the number and what did not, with the
  measurements, so the method is reusable without ever running the agent.

What you cannot do is re-run the *search*, because `aibuildai run` will refuse.
`search/task/` is still worth reading even then: the task contract, the scoring
contract, and the seven knowledge documents the agents read are the actual
specification of the problem, and the step records quoted in `../RECIPE.md` show
what the search concluded and why.

## What to expect, and what not to

The published run reached **2.2116** on the search's own 30-prompt harness,
which corresponds to **2.0089x** on MT-Bench (`../RESULTS.md`). Its per-step
trajectory is in `../RECIPE.md`.

A re-run will not follow the same path. Each step is one agent session forming
its own hypothesis; the published run's step 2 read the 30 scored prompts and
contradicted its own recipe skill, and step 8 found a decode-time win that no
prior run had looked for. Expect a different trajectory, and judge the re-run by
where it ends up rather than by whether step 4 matches step 4.

What *should* reproduce is the baseline: the plain SGLang arm landed between
52.30 and 52.43 tok/s across all ten scored steps, a spread of 0.25%. If your
baseline is outside about 2% of that, the instrument is wrong before the search
is interesting -- fix that first.

## Cost

Ten steps over 36 h, with a 48 h wall-clock and $250 cap, and a $50 per-agent
cap. A scoring call loads the target twice and takes about 40 min at this size,
so roughly a third of each step is spent being measured. Budget accordingly: the
same task on a 123B target spends about an hour per step on scoring alone.
