# What was removed from this snapshot, and how to get it back

This is a source snapshot of the agent framework, not a clone of its
repository. Four things were left out. Three are build artefacts; one is a
binary you may need.

## 1. Git history

Not included. The upstream repository has ~4,000 commits; this is the working
tree at the revision that produced the published result. `aibuildai_version.py`
reports the version; there is no commit to diff against.

## 2. `submodules/`

Not included, and not needed: `aibuildai_version.py` declares

    RUNTIME_SUBMODULE_PATHS: list[str] = []

so no submodule is loaded at runtime. Both are public upstream if you want them:

| submodule | upstream |
|---|---|
| `cli-proxy-api` | https://github.com/t2ance/CLIProxyAPI |
| `nature-bench` | https://github.com/FrontisAI/NatureBench |

`nature-bench` belongs to an unrelated task. `cli-proxy-api` is the authoring
source for the proxy binary below.

## 3. The proxy binary IS included

`packages/conversation/src/aibuildai_conversation/_native/cli-proxy-api` (42.8 MB,
statically linked Go) ships here with its `identity.json` and its upstream
LICENSE. It was briefly excluded; that was a mistake, because the conversation
package's build hook then refuses with *"The build requires the exact
CLIProxyAPI source commit"* and building it needs the submodule at its pinned
commit plus Go 1.26+.

With the binary present, set

    export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=21c3972447b7b51b0c86bb53f4e2552bac2526a4

and the hook verifies the declared bytes and skips the Go build:

    commit   21c3972447b7b51b0c86bb53f4e2552bac2526a4
    version  7.2.51
    sha256   5cb4f397b8bf456912383231fcd51763bbdef3a8ecce036f13c918b6d66a5d9c

The binary matters only if you route to GPT models -- `openai_proxy.py` is a
"Local Anthropic-compatible CLIProxyAPI process for GPT routing", and
`infra/owned_tools.py` resolves it only when `cli-proxy-api` is in the required
set. The published search ran every role on Claude and never asked for it. It is
shipped anyway because the *build* wants it even when the *run* does not.

## 4. Build artefacts

`__pycache__/`, `*.pyc`, `aibuildai.egg-info/`. Regenerated on install.

## 5. Local modifications carried in this snapshot

The tree that ran the search was `.dirty` — four tracked files differed from the
`science-2026-08-04` line, by 23 added and 3 removed lines in total. They are
included as they were, because this snapshot is the tree that produced the
result rather than a clean tag. None is on the code path this search takes:

| file | diff | on our path? |
|---|---|---|
| `cli_impl.py` | +4 −1 | CLI entry |
| `engine/builtin/tree/candidate.py` | +5 −1 | no — `tree` is a different search kind; ours is `autoresearch` |
| `engine/builtin/tree/programs.py` | +8 −1 | no — same |
| `packages/conversation/.../claude/event_adapter.py` | +6 −0 | event plumbing |

Two further patches exist that are **not** in this snapshot, because they were
made on a different machine for the large-target runs, and the published
Qwen2.5-14B search predates both. Anyone taking this framework to a target that
needs `tp_size > 1` will need the equivalent of them:

* **The score program's GPU grant** (`engine/builtin/aibuildai/capability.py`).
  A scorer gets its own claim, which is not the run's visible set, and the
  default is 0 cards. On a tensor-parallel target this makes SETUP pass and then
  every step fail, with an error about CUDA rather than about allocation.
* **The cgroup drain window** (`packages/execution/.../command_launcher.py`).
  2.0 s is allowed for a cgroup to drain after SIGKILL; a four-GPU engine takes
  longer, and the assertion kills an otherwise healthy run.

Both are described, with what they cost, in the large-target notes referenced
from `../search/README.md`.
