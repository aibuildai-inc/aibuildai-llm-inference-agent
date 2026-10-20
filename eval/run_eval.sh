#!/usr/bin/env bash
# Run the full literature-standard evaluation on this machine.
#
# Everything resolves from this script's own location, so it works wherever the release
# is unpacked. No paths to edit.
#
#   ./run_eval.sh --check              prerequisites only, ~2 min
#   ./run_eval.sh --phase mtbench      MT-Bench T=0 + T=1, both arms   (~70 min)
#   ./run_eval.sh --phase eager        HF eager reference on MT-Bench  (~60 min)
#   ./run_eval.sh --phase quality      GSM8K 500 + HumanEval 164       (~90 min)
#   ./run_eval.sh                      all three phases                (~4 h)
#
# Needs: one GPU with >=60 GB free, SGLang PATCHED (./tools/patch_sglang.sh from the repo root), and the
# target model available to huggingface_hub. Set HF_HOME and TMPDIR.
#
# Compare your output against REFERENCE.md. Anything inside ~2% is a match -- that is the
# instrument's resolution.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="$(cd "$HERE/.." && pwd)"
TASK="${SPECDEC_TASK:-$REL/task}"
HEAD="${SPECDEC_HEAD:-$REL/heads/draft_head.pt}"
PY="${PYTHON:-$(command -v python3)}"
OUT="$HERE/results"
PHASE="all"; CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    --head)  HEAD="$2"; shift 2 ;;
    --task)  TASK="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export TMPDIR="${TMPDIR:-/tmp}"
export SPECDEC_JIT_CACHE="${SPECDEC_JIT_CACHE:-$TMPDIR/specdec-jit-cache}"
mkdir -p "$SPECDEC_JIT_CACHE" "$OUT"

echo "=== setup ==="
printf '  release   : %s\n  task      : %s\n  head      : %s\n' "$REL" "$TASK" "$(basename "$HEAD")"
printf '  python    : %s\n  TMPDIR    : %s\n  JIT cache : %s\n' "$PY" "$TMPDIR" "$SPECDEC_JIT_CACHE"
[ -n "${HF_HOME:-}" ] && printf '  HF_HOME   : %s\n' "$HF_HOME" \
  || echo "  HF_HOME   : UNSET -- model will download to ~/.cache/huggingface (~30 GB)"
# The head is NOT required to verify prerequisites -- versions, the SGLang patch
# and the GPU are all independent of it -- so its absence is reported with the
# other checks below and only refused when an arm is about to load it.
[ -f "$TASK/models.json" ] || { echo "task not found: $TASK/models.json" >&2; exit 1; }

echo "=== prerequisites ==="
"$PY" - <<'PY'
import importlib, sys
need = {"torch":"2.13.0+cu130","transformers":"5.12.1","sglang":"0.5.21",
        "flashinfer":"0.6.18","accelerate":"1.15.0"}
bad=[]
for m,w in need.items():
    try: got = getattr(importlib.import_module(m), "__version__", "?")
    except ImportError: print(f"  {m:12s}: MISSING (reference used {w})"); bad.append(m); continue
    print(f"  {m:12s}: {got}" + ("" if got==w else f"   <-- reference used {w}"))
if bad: sys.exit("\nmissing: " + ", ".join(bad))
PY
"$PY" - <<'PY'
import pathlib, sys, sglang
f = pathlib.Path(sglang.__file__).parent/"srt"/"models"/"qwen2_eagle.py"
s = f.read_text()
ok = "OUR-HEAD-PATCH" in s and "del self.input_layernorm" not in s
print("  sglang patch:", "OK" if ok else "MISSING")
if not ok: sys.exit(f"\nRun ./tools/patch_sglang.sh from the repo root first. Without it the head computes a\n"
                    f"different function (relative error 1.84 vs 7.7e-17).\nFile: {f}")

# The marker only proves the EDIT happened. SGLang discovers models by importing
# every module under sglang/srt/models and SWALLOWS import errors
# ("Ignore import error when loading ..."), so a module that no longer imports is
# silently dropped from the registry and the failure surfaces much later as
#   ValueError: Cannot find model module. 'Qwen2ForCausalLMEagle' is not a
#   registered model in the Transformers library
# 20 minutes into a run. Check importability and registration here instead.
import importlib, traceback
try:
    importlib.import_module("sglang.srt.models.qwen2_eagle")
    print("  draft module : imports")
except Exception:
    traceback.print_exc()
    sys.exit("\nThe patched draft module does not import, so SGLang will not register it.\n"
             "Restore it with:  cp %s.orig %s\n"
             "then re-run ./tools/patch_sglang.sh and read its output." % (f, f))
from sglang.srt.models.registry import ModelRegistry
arch = "Qwen2ForCausalLMEagle"
if arch in ModelRegistry.models:
    print(f"  registry     : {arch} registered")
else:
    sys.exit(f"\n{arch} is missing from SGLang's ModelRegistry even though the module imports.\n"
             f"Registered eagle architectures: "
             f"{sorted(k for k in ModelRegistry.models if 'Eagle' in k)}\n"
             f"models.json declares draft_architecture -- it must match one of those exactly.")
import torch
if not torch.cuda.is_available(): sys.exit("  no CUDA device visible")
free,total = torch.cuda.mem_get_info(0)
print(f"  gpu         : {torch.cuda.get_device_name(0)}  {free/2**30:.1f}/{total/2**30:.1f} GB free")
if free < 60*2**30: sys.exit(f"  need >=60 GB free, have {free/2**30:.1f}")
PY
for f in mt_bench_question.jsonl gsm8k_test.jsonl humaneval.jsonl; do
  [ -f "$HERE/$f" ] && echo "  data: $f ($(wc -l < "$HERE/$f") lines)" \
    || { echo "  data MISSING: $f" >&2; exit 1; }
done
if [ -f "$HEAD" ]; then
  echo "  head        : $(basename "$HEAD") ($(stat -c %s "$HEAD") bytes)"
else
  echo "  head        : NOT PRESENT at $HEAD"
  echo "                a release asset, not in git -- see heads/README.md (release v1.0.0)"
  echo "                needed only by eval/; search/ and train/ do not use it"
fi

if [ "$CHECK" = "1" ]; then
  echo
  if [ -f "$HEAD" ]; then
    echo "--check passed. Drop --check to run."
  else
    echo "--check passed except the head weights. Fetch them, then drop --check to run."
  fi
  exit 0
fi

# Past this point the head is required: every arm below loads it.
if [ ! -f "$HEAD" ]; then
  echo "cannot measure without the head weights: $HEAD" >&2
  echo "See heads/README.md (release v1.0.0). Re-run with --check to verify the rest." >&2
  exit 1
fi

# package the head into an SGLang draft dir once
SUB="$(mktemp -d "$TMPDIR/specdec_sub_XXXXXX")"
DRAFT="$TMPDIR/specdec_eval_draft"
trap 'rm -rf "$SUB"' EXIT
cp "$HEAD" "$SUB/head.pt"; cp "${HEAD%.pt}.config.json" "$SUB/head_config.json"
rm -rf "$DRAFT"
echo "=== packaging draft ==="
"$PY" "$TASK/harness/make_draft.py" --submission "$SUB" --models "$TASK/models.json" \
      --out "$DRAFT" >/dev/null
echo "  $DRAFT ($(du -sh "$DRAFT" | cut -f1))"

mtb() {  # $1=label $2=temp $3=extra args
  echo "=== $1  $(date +%H:%M:%S) ==="
  "$PY" "$HERE/bench_mtbench.py" --models "$TASK/models.json" --harness "$TASK/harness" \
      --questions "$HERE/mt_bench_question.jsonl" --temperature "$2" \
      --max-new-tokens 1024 --repeats 1 $3 --out "$OUT/$1.json" 2>&1 | tail -6
  echo "--- $1 done $(date +%H:%M:%S) ---"
}
qual() {  # $1=label $2=bench $3=extra args
  echo "=== $1  $(date +%H:%M:%S) ==="
  "$PY" "$HERE/bench_quality.py" --bench "$2" --models "$TASK/models.json" \
      --harness "$TASK/harness" --max-new-tokens 512 $3 --out "$OUT/$1.json" 2>&1 | tail -6
  echo "--- $1 done $(date +%H:%M:%S) ---"
}

case "$PHASE" in
  mtbench|all)
    mtb mtbench_t0_plain 0.0 ""
    mtb mtbench_t0_draft 0.0 "--draft $DRAFT"
    mtb mtbench_t1_plain 1.0 ""
    mtb mtbench_t1_draft 1.0 "--draft $DRAFT"
    ;;
esac
case "$PHASE" in
  eager|all) mtb mtbench_t0_hfeager 0.0 "--backend hf" ;;
esac
case "$PHASE" in
  quality|all)
    qual gsm8k_plain     gsm8k     "--limit 500"
    qual gsm8k_draft     gsm8k     "--limit 500 --draft $DRAFT"
    qual humaneval_plain humaneval ""
    qual humaneval_draft humaneval "--draft $DRAFT"
    ;;
esac

echo
echo "=== SUMMARY  $(date +%H:%M:%S) ==="
"$PY" "$HERE/summarise.py" --results "$OUT" --reference "$HERE/REFERENCE.md"
