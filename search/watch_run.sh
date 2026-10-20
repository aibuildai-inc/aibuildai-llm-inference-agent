#!/usr/bin/env bash
# Terminal view of a live AIBuildAI run, from the files the web UI reads.
# Read-only: it opens nothing the run writes to and holds no locks.
#
#   ./watch_run.sh [playground-dir] [interval-seconds]
#
# Defaults: ~/aibuildai-playground, 60s. Ctrl-C to stop; the run is unaffected.
set -uo pipefail
PG="${1:-$HOME/aibuildai-playground}"
EVERY="${2:-60}"
PY="$(command -v python3)"

# The run directory sits at playground/<task_name>/<timestamp>_<runid>/, not
# directly under the playground. Accept either depth and identify the run by the
# file every run has, rather than by position -- an earlier version globbed one
# level and silently picked the TASK directory, whose workspace/ does not exist.
RUN=""
for cand in $(ls -dt "$PG"/*/ "$PG"/*/*/ 2>/dev/null); do
  if [ -f "$cand/run_config.json" ]; then RUN="$cand"; break; fi
done
[ -n "$RUN" ] || { echo "no run directory under $PG (looked for run_config.json one and two levels down)" >&2; exit 1; }
SEARCH="$RUN/workspace/search_1"
echo "watching: $RUN"
echo

while true; do
  printf '\n===== %s =====\n' "$(date +%H:%M:%S)"

  # elapsed, from the run directory's own creation
  started=$(stat -c %W "$RUN" 2>/dev/null); [ "${started:-0}" -eq 0 ] && started=$(stat -c %Y "$RUN")
  budget=$("$PY" -c "
import json,sys
try:
    c=json.load(open('$RUN/run_config.json'))
    print(c['run']['budget']['wall_clock_minutes'])
except Exception: print('?')" 2>/dev/null)
  printf '  elapsed      : %d min of %s\n' $(( ($(date +%s) - started) / 60 )) "$budget"

  # which work units exist, and which have finished
  if [ -d "$SEARCH" ]; then
    setup=$(ls -d "$SEARCH"/setup_* 2>/dev/null | wc -l)
    workers=$(ls -d "$SEARCH"/worker_* 2>/dev/null | wc -l)
    scorers=$(ls -d "$SEARCH"/score_program_* 2>/dev/null | wc -l)
    printf '  units        : setup %s  workers %s  scorers %s\n' "$setup" "$workers" "$scorers"
    newest=$(ls -dt "$SEARCH"/*/ 2>/dev/null | head -1)
    printf '  newest unit  : %s  (last write %s)\n' "$(basename "$newest")" \
           "$(date -d "@$(stat -c %Y "$newest")" +%H:%M:%S)"
  else
    printf '  units        : SETUP has not created the search workspace yet\n'
  fi

  # the scorer is the authority: one .program-result.json per scored step
  printf '  scored steps :\n'
  found=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    found=1
    st=$(echo "$f" | grep -o 'score_program_[0-9]*' | grep -o '[0-9]*')
    "$PY" - "$f" "$st" <<'PYS'
import json, sys
try:
    j = json.load(open(sys.argv[1]))
except Exception:
    sys.exit()
o = j.get("output") or {}
s = o.get("score")
if s is None:
    print(f"    step {sys.argv[2]:>2}  FAILED  {str(o.get('failed'))[:60]}")
else:
    c = o.get("components") or {}
    extra = "  ".join(f"{k} {v:.4g}" for k, v in list(c.items())[:3] if isinstance(v, (int, float)))
    print(f"    step {sys.argv[2]:>2}  {s:.4f}   {extra}")
PYS
  done < <(find "$SEARCH" -name '.program-result.json' -path '*score_program*' 2>/dev/null | sort -V)
  [ "$found" = 0 ] && printf '    none yet (step 1 scores several hours in)\n'

  # the scorer the setup agent wrote, and how it differs from the published one
  if [ -f "$RUN/private/score.py" ]; then
    printf '  own scorer   : %s lines' "$(wc -l < "$RUN/private/score.py")"
    ref="${SPECDEC_REPO:-$(cd "$(dirname "$0")" && pwd)}/search/score.reference.py"
    [ -f "$ref" ] || ref="$(cd "$(dirname "$0")" && pwd)/AIBuildAI-SpecDec/search/score.reference.py"
    [ -f "$ref" ] && printf '  (%s lines differ from score.reference.py)' \
      "$(diff "$RUN/private/score.py" "$ref" 2>/dev/null | grep -c '^[<>]')"
    printf '\n'
  fi

  # GPU and the newest step record's headline
  printf '  gpu          : %s\n' "$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                                    --format=csv,noheader | paste -sd' | ')"
  rec=$(ls -t "$SEARCH"/records/worker_*/README.md 2>/dev/null | head -1)
  [ -n "$rec" ] && printf '  latest record: %s\n' "$(head -1 "$rec" | cut -c1-96)"

  sleep "$EVERY"
done
