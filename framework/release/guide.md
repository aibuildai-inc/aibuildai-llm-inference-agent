# AIBuildAI

AIBuildAI builds a machine-learning model for you. You give it a folder holding your data and a description of what you want. It designs several candidate models, trains them, scores them on data it held back, and leaves you the best one together with the work that produced it.

## Before you start

You need all of this, except that Kubernetes start mode uses its Pod instead of a systemd user manager. The run checks two local requirements itself and says which one failed: the systemd user manager and enough visible GPUs. It does not check the other requirements, so a machine that is missing one fails less clearly.

- Linux x86_64 — no macOS, no native Windows, no ARM. On Windows, WSL2 with systemd enabled works. glibc 2.28 or newer, so Ubuntu 20.04 and later, Debian 10 and later, RHEL and CentOS 8 and later.
- A systemd user manager for local start mode. A local run is placed in a systemd service so the kernel can hold it to the limits you set, and it will not start without one. A normal login session already has one. Over a bare SSH command, run `sudo loginctl enable-linger $USER` and open a new login shell. In a local container, use a base image with systemd enabled. Kubernetes start mode does not need user systemd: the Pod owns the fresh run and its cgroup bounds it.
- No Python and no conda: AIBuildAI ships its own environment tooling and creates each run's Python environment itself. The one exception is the optional `run.conda_env` setting, which clones one of your existing conda environments into the run — using it needs conda in a shell where it is initialized (`conda init` once, or `source <conda>/etc/profile.d/conda.sh`; `conda` on `PATH` alone is not enough).
- An NVIDIA GPU and its driver. There is no CUDA version for you to match: AIBuildAI reads the driver's own CUDA version and installs a matching build of PyTorch for each task. A machine with no GPU does not stop the run: each candidate is admitted with no card and trains on the CPU, far more slowly, and nothing warns you. On such a machine set `resources.cuda_visible_devices: []` so that CPU-only is what you chose rather than what you got.
- An AIBuildAI account with an active subscription, and a way to call a model. These are two separate things — see "Sign in, and give it a model".
- Network access, and a writable disk for the outputs.

## What is in the package

Unpacking the tarball gives you a directory holding:

- `aibuildai` — the program you run.
- `install.sh` — the installer.
- `README.md` — this file.
- `_internal/` — everything the program needs at runtime.
- `tools/` — the helper programs the product runs (LaTeX, the Workspace gateway, the model adapters, the environment tool).
- `SHA256SUMS` — checksums for the contents.
- `VERSION` — the exact build, used by the installer and by the updater.

## Install

From inside the unpacked directory:

```bash
./install.sh
```

It checks this build before replacing the installed AIBuildAI, restores the installed build if replacement fails, and leaves exactly one version after a successful install. The `~/.local/bin/aibuildai` command points to that version, and the installer adds `~/.local/bin` to your `PATH` in `~/.bashrc` if it is not there already.

Check it worked:

```bash
aibuildai --help
```

If your shell still cannot find it, run `source ~/.bashrc` and `hash -r`, then try again.

To remove AIBuildAI: `rm -rf ~/.local/bin/aibuildai ~/.local/lib/aibuildai`. Your account and your memory stay in `~/.aibuildai`.

## Install the Conversation library from source

The Linux x86_64 Conversation library uses Python 3.11 through 3.13. Initial builds from a fixed ML-agent checkout require Go 1.26 or later and the pinned proxy source. The resulting wheel and sdist carry the proxy executable; installed calls and sdist rebuilds do not use Go or the product checkout. Prepare the source and install the library:

```bash
git submodule update --init submodules/cli-proxy-api
python -m pip install ./packages/execution
python -m pip install ./packages/conversation
```

For product development, install the library before the product:

```bash
python -m pip install -c requirements.lock -e ./packages/execution
python -m pip install -c requirements.lock -e ./packages/conversation
python -m pip install -c requirements.lock -e .
```

Import `Conversation` and `AgentSpec` from `aibuildai_conversation`. A Claude Code login supplies model access; no Anthropic API key is needed for that login. This model login is separate from `aibuildai login`. The generated [Conversation API reference](../plugins/aibuildai-builtin-marketplace/aibuildai-builtin/skills/meta-search-design/references/api/aibuildai_conversation.md) describes the public calls. This change does not publish the library to PyPI.

## Coming from v2.5

A `task.yaml` written for v2.5 will not run, and it is not worth repairing key by key. Too much moved: fields were renamed, blocks were replaced, whole sections went away, and some keys kept their name while changing what they hold.

Start from a fresh file instead:

```bash
aibuildai config > task.yaml
```

Then copy your own values across — your task name, your paths, your model, your budget — and run it. AIBuildAI refuses any key it does not recognise and names every one of them, so if you paste an old block by mistake it tells you exactly which lines to remove rather than ignoring them.

Three things worth knowing while you do that, because no config file will mention them. The task statement is no longer a config field: it lives in the task folder, and `run.data_root` is now that folder itself rather than a parent above it. A run folder no longer holds `progress.pdf` — open the finished run in the local Web Workspace instead. And an agent stops at the time and cost limit you set, with no extra allowance past it.

To upgrade an installed copy, run `install.sh` from the new package. Your `aibuildai login` session is kept. AIBuildAI also updates itself when a newer release exists; set `AIBUILDAI_AUTO_UPDATE=0` to stop that. Library caches (Hugging Face models and datasets, torch, Triton) are the ones your shell already points at, so a model downloads once and every run on the machine reuses it; set `AIBUILDAI_RUN_LOCAL_CACHE=1` to keep each run's caches inside its own run folder instead.

## Sign in, and give it a model

Two separate things, and it is easy to mix them up. Your AIBuildAI subscription is what authorizes a run. Your model credential is what the run spends while it works.

Sign in to AIBuildAI once:

```bash
aibuildai login      # opens a browser; on a server with no browser: aibuildai login --no-browser
aibuildai whoami     # shows your account, your plan, and whether this machine can start a run
```

`aibuildai account` opens your account portal, where the Plans tab is. `aibuildai logout` signs out.

Then choose a model by setting `llm.default.model` in your config:

- **Anthropic** — a `claude-*` id. If Claude Code is already signed in on this machine, no Anthropic API key is needed; AIBuildAI uses that login. Otherwise run `claude auth login`, or put your Anthropic API key in `AIBUILDAI_API_KEY`. On a machine with no browser, Claude Code's own `CLAUDE_CODE_OAUTH_TOKEN` counts as that login and AIBuildAI uses it; a token written to `~/.aibuildai/claude_token` takes precedence and is read again at every conversation start, so overwriting that file moves a running machine onto another login without stopping it (the conversations already open finish on the old one).
- **OpenAI** — a bare `gpt-*` id, for example `gpt-5.6`. If this machine is signed in to Codex, that login is used; otherwise export your OpenAI key as `AIBUILDAI_API_KEY`. There is no endpoint to set: the adapter that shipped in the package handles it.
- **DeepSeek** — a bare `deepseek-*` id, for example `deepseek-v4-flash`. Export your DeepSeek key as `AIBUILDAI_API_KEY`.
- **OpenRouter** — use a provider-prefixed model id such as `deepseek/deepseek-v4-flash`, export its key as `AIBUILDAI_API_KEY`, and set `AIBUILDAI_SMALL_FAST_MODEL` to a small model id that OpenRouter serves. There is no endpoint to set.
- **Another Anthropic-compatible endpoint** — export `AIBUILDAI_BASE_URL` with its base URL, `AIBUILDAI_API_KEY` with its key, and `AIBUILDAI_SMALL_FAST_MODEL` with a small model id that endpoint serves.

## Write the task folder

One folder is the whole input. Put your data in it and write what you want, in plain language, in any file you like. AIBuildAI reads the folder once at the start, keeps what it says as the brief for the run, and never writes to it.

There is no required layout and no required file name. An example folder:

```text
~/data/spooky-author-identification/
  task.md
  train.csv
  test.csv
  sample_submission.csv
```

And `task.md`:

```md
Task: predict the author of each text sample.

Goal:
- Multiclass classification
- Judged on log loss, lower is better

Data:
- train.csv has the training rows
- test.csv has the rows to predict
- sample_submission.csv shows the output format

Requirements:
- Use Python
- Write the predictions as submission.csv, in the same format as sample_submission.csv
```

Say what you want produced and how it should be judged. If you already have a scoring script, put it in the folder and say so. Useful things to say: what kind of task it is, the metric, what each file is, anything banned or preferred, and what the machine has.

### Starting with no data

You can start from a description alone. Set `run.setup_from_scratch: true` and write the problem in `run.task_prompt`. AIBuildAI then finds the data itself, works out how the result should be scored, holds back its own answer key, and writes the scoring program before the search begins. `run.data_root` still points at the task folder, and that folder starts empty and stays read-only: what AIBuildAI fetches goes to its own working directory, and the parts the candidates are allowed to see are copied into the run's outputs.

## Run it

Generate a config, edit it, run it:

```bash
aibuildai config > task.yaml
```

The fields marked `REQUIRED -- replace` have no default. A run will not start until each is set:

- `run.task_name` — a short id; it names the output directory
- `run.data_root` — the task folder
- `run.playground_root` — where outputs are written
- `llm.default.model` — which model to use
- `work_units` — how long each kind of agent may take; the starter config already carries a working set
- `search.parallel` — how many candidates run at once
- `resources.work_unit.cpu_max_cores` and `resources.work_unit.memory_max_gb` — the ceiling for one unit of work

Look at `run.budget` before a long run. It is not marked, because the starter config already fills it in — with 24 hours of wall clock and no spending limit at all. `run.budget.cost_usd` limits the whole run and `run.budget.per_agent_cost_cap_usd` limits one agent's own spend, not the work that agent starts. At the whole-run limit AIBuildAI stops starting new work and lets what is already running finish.

Then:

```bash
aibuildai run task.yaml
```

The Web Workspace shows the search as it happens in your browser. Press `Ctrl+C` in the terminal to stop the run. A run can take hours — training steps are the slow part.

Every run is also served to the local AIBuildAI Workspace in the browser at `http://127.0.0.1:47321/`. This default Workspace has the name `default`, and Caddy owns its fixed endpoint while private backend generations change behind it. To start an independent Workspace from a worktree, set `AIBUILDAI_WEB_NAME` to a unique lowercase name and set `AIBUILDAI_WEB_PORT` to its unique port in that worktree's shell, tmux session, or direnv; do not put these deployment settings in the run YAML. The run prints its own page with the selected Workspace port, `http://127.0.0.1:<Workspace port>/run/<run_id>`; open it to see the same live run state, drill into each Search, Composite, and WorkUnit, read agent transcripts, and browse the run's workspace files. The Workspace home page lists every run on the machine, running or finished, and the run selector at the top left switches between them; a finished run keeps the same page after its process has exited, restored from its journal. The page is the only place to control a run after it starts: `Pause run` stops the current process and keeps every recorded step, `Resume` continues the same run, and Replay reads its recorded history without starting a run process. AIBuildAI saves the original launch environment in a private owner-only context at run birth, so Resume keeps arbitrary model and MCP credentials even when the Workspace process does not have them. Those values are passed to systemd through a private file; they do not appear in the command line, run journal, archived config, logs, or Web responses. A finished run gets a replay strip at the bottom of the page: play, scrub, and change speed to watch the recorded history while the workspace panel keeps showing the final files. The page stays local to your machine. Closing or refreshing the browser never affects the run, and `Ctrl+C` in the terminal still stops it. On a remote GPU host, forward the Workspace port with SSH or your editor's port forwarding.

| Workspace deployment variable | Default | Use |
| --- | --- | --- |
| `AIBUILDAI_WEB_NAME` | `default` | Sets the Workspace name. Use only lowercase letters, digits, and hyphens. |
| `AIBUILDAI_WEB_PORT` | `47321` | Sets the Workspace port. A non-default name must set this variable. |

To continue an inactive unfinished run, open its page in the local Web Workspace and select `Resume`.

Kubernetes start mode is fresh-run-only. The Pod owns that run, and the product does not start a local Workspace or offer Pause, Resume, or Replay for it.

To reopen a finished run and look at it again, open its page in the local Web Workspace.

## Choose a search method

`search.kind` decides how the run explores designs. There are two.

`tree` is the default. It fans out several candidate designs, scores them, keeps the best, revises them, and combines what the best ones learned.

`nb_tree` suits scientific tasks. One worker takes a candidate from idea to finished output on its own, and each finished candidate is scored by the task's own scoring program.

The generated config carries `search.kind` commented out, in the Advanced block at the bottom. Uncomment it there, or add the block yourself:

```yaml
search:
  kind: nb_tree
```

## Fit the run to your machine

Three fields together decide how much of your machine a run uses.

`resources.work_unit.cpu_max_cores` and `resources.work_unit.memory_max_gb` are the ceiling for one unit of work — the kernel enforces them, so a runaway step cannot take the machine down. `search.parallel` is how many units run at once. Multiply them to size the run: four units at 16 GiB each needs 64 GiB.

To choose which GPUs the run may touch, set `resources.cuda_visible_devices` to the card indices you want:

```yaml
resources:
  cuda_visible_devices: [0, 1]
```

Exporting `CUDA_VISIBLE_DEVICES` in your shell instead will stop the run at startup. This field is the one place to say it.

## Where the outputs go

Under the playground directory you named, AIBuildAI makes one directory per run:

```text
<playground>/<task-name>/<run-id>/
```

The run id looks like `20260712T201625Z_4f2b91c0d83e47a6b15e9c7d2a604f38`. Inside it:

- `deliverable/` — **what you asked for.** This is the one to look at first.
- `private/` — the scoring program and the answer key AIBuildAI held back.
- `public/` — the task statement and everything the agents were allowed to read.
- `workspace/` — the work itself: one directory per execution, holding that role's own artifacts, plus the Research First Board index that links to them.
- `submitter/` — present only when the run submitted to an external scoring service: every result that service returned, one JSON line each.

## Write a paper about a finished run

AIBuildAI can write up a run as a NeurIPS-format paper and compile it to a PDF.

```bash
aibuildai write-paper <run-dir>
```

The command prints the path of the PDF it produced, inside the Writer's own execution directory. To have every run write one when it finishes, set `writer.enable: true` in your config.

## Summarize past runs

After a few runs on the same task, AIBuildAI can fold what they learned into notes it reads on the next run:

```bash
aibuildai memorize task.yaml
```

It writes one summary per run, then folds them into `~/.aibuildai/memory/tasks/<task_name>.md`. You can edit that file; your edits are used on the next run. Running `memorize` again rewrites it from the per-run summaries, so keep your own notes elsewhere if you want them to survive.

## When something goes wrong

**`aibuildai: command not found`** — run `source ~/.bashrc`, then `hash -r`.

**It stops saying it cannot find a host conda** — you set `run.conda_env`, which clones one of your own conda environments into the run, on a machine where conda is not initialized. Start AIBuildAI from a shell where conda is initialized (`conda` on `PATH` is not enough), or remove `run.conda_env` and let AIBuildAI create the environment itself.

**It stops saying a design requested more GPUs than are visible** — a design asked for more cards than this machine has. The run sends that refusal back to the design agent to lower its request; on a machine with no GPU at all, set `resources.cuda_visible_devices: []`.

**Training is far slower than you expected** — check that a GPU is really being used. A machine with no visible card does not stop the run; it trains on the CPU without saying so.

**It stops saying a field is required** — a `REQUIRED -- replace` field is missing. The simplest fix is `aibuildai config > task.yaml` and copying your values into the fresh file.

**It stops saying it cannot tell how the task is judged** — the task folder does not say what counts as good. Add the metric and what the result is compared against, or put your own scoring script in the folder and say so.

**It stops saying your session has expired** — run `aibuildai login` again.

If you report a problem, include the command you ran, the run directory, and what the screen said when it stopped.
