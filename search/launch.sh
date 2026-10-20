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
"$REPO/tools/patch_sglang.sh" >/dev/null 2>&1 \
  && echo "  sglang     : patched" \
  || echo "  sglang     : PATCH FAILED -- run $REPO/tools/patch_sglang.sh and read its output" >&2
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

echo "=== launching ==="
exec aibuildai run "$GEN"
