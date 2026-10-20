#!/usr/bin/env bash
# Launch the search from a copy of this repository on any machine.
#
# run_config.yaml carries __REPO__ and __PLAYGROUND__ placeholders rather than
# environment variables, because the framework does NOT expand ${VAR} in path
# fields -- infra/fs/pathing.normalize_path() is expanduser + absolutize only, so
# a ${VAR} path would be taken literally and joined to the cwd. This script
# substitutes real absolute paths and launches the generated config.
#
#   ./search/launch.sh [playground-dir]
#
# Default playground is ~/aibuildai-playground. It must NOT be inside this
# repository: a run writes tens of GB there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PLAYGROUND="${1:-$HOME/aibuildai-playground}"
mkdir -p "$PLAYGROUND"
case "$(cd "$PLAYGROUND" && pwd)" in
  "$REPO"|"$REPO"/*) echo "refusing: playground must be outside the repository" >&2; exit 2 ;;
esac

GEN="$PLAYGROUND/run_config.generated.yaml"
sed -e "s|__REPO__|$REPO|g" -e "s|__PLAYGROUND__|$(cd "$PLAYGROUND" && pwd)|g" \
    "$HERE/run_config.yaml" > "$GEN"
if grep -q '__REPO__\|__PLAYGROUND__' "$GEN"; then
  echo "substitution incomplete -- leftover placeholders in $GEN" >&2; exit 1
fi

echo "=== generated config ==="
printf '  %s\n' "$GEN"
grep -E '^\s+(data_root|playground_root|task_name|conda_env):' "$GEN" | sed 's/^/  /'
echo "  skill paths:"; grep -E '^\s+path: ' "$GEN" | sed 's/^/    /'

echo "=== preconditions ==="
command -v aibuildai >/dev/null \
  && echo "  aibuildai  : $(command -v aibuildai) ($(aibuildai -V 2>/dev/null | head -1))" \
  || { echo "  aibuildai  : NOT INSTALLED -- see Install in search/README.md" >&2; exit 1; }
V_PY="$(command -v python3)"

# The SGLang patch belongs in the SERVING environment -- the one run.conda_env
# names, which the framework clones for each run's program environment -- not in
# whichever environment the launcher happens to run from. Launching from the
# framework env means `python3` here has no sglang at all, so patching with it
# would both fail and be pointless. Verify the serving env instead.
SERVING_ENV="$(grep -E '^\s+conda_env:' "$GEN" | head -1 | sed 's/.*conda_env:[[:space:]]*//')"
if [ -n "$SERVING_ENV" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  SERVING_PY="$CONDA_BASE/envs/$SERVING_ENV/bin/python"
  if [ -x "$SERVING_PY" ]; then
    if "$SERVING_PY" - <<'PYCHK'
import sys, pathlib
try:
    import sglang
except Exception as e:
    sys.exit(f"sglang not importable in the serving env: {e}")
f = pathlib.Path(sglang.__file__).parent / "srt" / "models" / "qwen2_eagle.py"
s = f.read_text()
if "OUR-HEAD-PATCH" not in s or "del self.input_layernorm" in s:
    sys.exit(f"NOT PATCHED: {f}")
import importlib
importlib.import_module("sglang.srt.models.qwen2_eagle")
from sglang.srt.models.registry import ModelRegistry
if "Qwen2ForCausalLMEagle" not in ModelRegistry.models:
    sys.exit("patched but Qwen2ForCausalLMEagle is not in ModelRegistry")
PYCHK
    then
      echo "  serving env: $SERVING_ENV -- patched, imports, architecture registered"
    else
      echo "  serving env: $SERVING_ENV -- PATCH/REGISTRY PROBLEM (see above)" >&2
      echo "               run this inside that env:  $REPO/tools/patch_sglang.sh" >&2
      exit 1
    fi
  else
    echo "  serving env: '$SERVING_ENV' not found at $SERVING_PY" >&2
    echo "               the framework will create a bare environment instead, and the" >&2
    echo "               run's own agents will have to install the serving stack." >&2
  fi
fi
# Model access comes from a Claude Code login; no Anthropic API key is needed
# for it (framework/release/guide.md). ANTHROPIC_API_KEY also works if you
# prefer. This login is separate from `aibuildai login`.
if claude_bin=$("$V_PY" -c 'import claude_agent_sdk,pathlib;print(pathlib.Path(claude_agent_sdk.__file__).parent/"_bundled"/"claude")' 2>/dev/null) \
   && [ -x "$claude_bin" ]; then
  echo "  claude SDK : $claude_bin"
else
  echo "  claude SDK : NOT FOUND -- the conversation package did not install its bundled executable" >&2
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] \
  && echo "  model auth : ANTHROPIC_API_KEY is set" \
  || echo "  model auth : no API key set -- relying on your Claude Code login (that is supported)"

# `aibuildai run` is gated by subscription/run_grant.py. A SOURCE build (this
# one) can skip that verdict with AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1, which
# cli_impl.py honours only when is_frozen() is false. A frozen binary cannot.
if [ "${AIBUILDAI_SKIP_SUBSCRIPTION_CHECK:-}" = "1" ]; then
  echo "  run gate   : skipped (AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1, source build)"
elif "$V_PY" -c 'import sys; sys.path.insert(0,"'"$REPO"'/framework");
from subscription import store
sys.exit(0 if store.load() is not None else 1)' 2>/dev/null; then
  echo "  run gate   : signed in"
else
  echo "  run gate   : NOT signed in, and the skip flag is not set." >&2
  echo "               Either:  aibuildai login" >&2
  echo "               or:      export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1   (source builds only)" >&2
  exit 1
fi

echo "=== launching ==="
exec aibuildai run "$GEN"
