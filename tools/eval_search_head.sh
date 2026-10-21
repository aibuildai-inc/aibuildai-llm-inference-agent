#!/usr/bin/env bash
# Evaluate the head a Quickstart C search produced, on the Quickstart A protocol.
#
# A search delivers `head.pt` + `head_config.json` side by side; run_eval.sh wants
# `<name>.pt` + `<name>.config.json`. Staging them under those names is all this
# does -- run_eval.sh then reads sglang_overrides out of the config itself and
# writes them into the draft dir, so the head is measured on its own tree.
#
#   ./tools/eval_search_head.sh <dir-with-head.pt> [phase]
#
set -euo pipefail
SRC="${1:?usage: $0 <dir containing head.pt and head_config.json> [phase]}"
PHASE="${2:-mtbench}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${SPECDEC_STAGE:-$HOME/specdec-eval-head}"

[ -f "$SRC/head.pt" ]          || { echo "no head.pt in $SRC" >&2; exit 1; }
[ -f "$SRC/head_config.json" ] || { echo "no head_config.json in $SRC" >&2; exit 1; }

mkdir -p "$STAGE"
cp "$SRC/head.pt"          "$STAGE/mine.pt"
cp "$SRC/head_config.json" "$STAGE/mine.config.json"
echo "staged: $STAGE/mine.pt ($(stat -c %s "$STAGE/mine.pt") bytes)"

# The tree. make_draft.py does not carry sglang_overrides into the draft dir, and
# bench_*.py reads them from _overrides.json INSIDE that dir, so without this the
# run silently uses the task defaults and under-reports a head tuned for another tree.
python3 - "$STAGE/mine.config.json" "$HERE/task/models.json" <<'PY'
import json, sys
cfg  = json.load(open(sys.argv[1]))
task = json.load(open(sys.argv[2]))
ov   = cfg.get("sglang_overrides") or {}
sg   = task["sglang"]
keys = ("speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens")
mine = {k: int(ov[k]) for k in keys if k in ov}
if not mine:
    print("  tree: head_config.json names none; the task defaults apply "
          f"({sg[keys[0]]}/{sg[keys[1]]}/{sg[keys[2]]})")
else:
    shown = "/".join(str(mine.get(k, sg[k])) for k in keys)
    dflt  = "/".join(str(sg[k]) for k in keys)
    print(f"  tree: head wants {shown}, task default is {dflt}"
          + ("  <-- DIFFERENT, overrides required" if shown != dflt else ""))
PY

echo
echo "NOTE: eval/REFERENCE.md describes the PUBLISHED head. summarise.py will"
echo "      compare yours against it and report a mismatch -- that is a different"
echo "      head, not a failure. Read the plain/draft tok/s pair from your own run."
echo
PYTHON="${PYTHON:-$(which python)}" "$HERE/eval/run_eval.sh" --head "$STAGE/mine.pt" --phase "$PHASE"
