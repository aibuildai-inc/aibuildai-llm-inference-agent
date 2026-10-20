#!/usr/bin/env bash
# Restore layer-0 input_layernorm in SGLang's Qwen2 EAGLE draft model.
#
# WHY THIS IS MANDATORY
#   SGLang deletes layer 0's input_layernorm for parity with the reference EAGLE
#   implementation (cnets.py L427). The heads in ../heads/ HAVE it and use it, so
#   without this patch the converted weights compute a DIFFERENT FUNCTION.
#   Measured: relative error 1.84 unpatched, 7.7e-17 patched.
#   harness/check_draft.py refuses to run unpatched, so you cannot do this by accident.
#
# USAGE
#   ./patch_sglang.sh                      # auto-discovers SGLang in the active python
#   ./patch_sglang.sh <path-to-sglang>/srt/models/qwen2_eagle.py
#   PYTHON="$(command -v python3)" ./patch_sglang.sh    # or any interpreter
#
# Idempotent. Keeps a .orig backup beside the file. Re-run after upgrading SGLang.
set -euo pipefail

F=${1:-}
if [ -z "$F" ]; then
  PY=${PYTHON:-$(command -v python3)}
  F=$("$PY" - <<'PY'
import pathlib, sys
try:
    import sglang
except ImportError:
    sys.exit("sglang is not importable by this interpreter; activate the env or pass the file path")
p = pathlib.Path(sglang.__file__).parent / "srt" / "models" / "qwen2_eagle.py"
print(p if p.is_file() else "", end="")
PY
)
  [ -n "$F" ] || { echo "could not locate qwen2_eagle.py; pass it as \$1" >&2; exit 1; }
  echo "auto-discovered: $F"
fi
[ -f "$F" ] || { echo "not found: $F" >&2; exit 1; }

if grep -q 'OUR-HEAD-PATCH' "$F"; then echo "already patched: $F"; exit 0; fi
[ -f "$F.orig" ] || cp "$F" "$F.orig"

python3 - "$F" <<'PY'
import sys, pathlib, re
p = pathlib.Path(sys.argv[1]); s = p.read_text()

# SGLang writes this block slightly differently across versions, so match on the
# structure (the layer_id == 0 guard that removes input_layernorm) rather than text.
pat = re.compile(
    r"\n(?P<i>[ \t]*)# Skip the input_layernorm\n"
    r"(?:[ \t]*#[^\n]*\n)*"
    r"[ \t]*if layer_id == 0:\n"
    r"[ \t]*del self\.input_layernorm\n")
m = pat.search(s)
if not m:
    if "del self.input_layernorm" in s:
        sys.exit("FOUND 'del self.input_layernorm' but not in the expected shape.\n"
                 "Patch by hand: delete the 'if layer_id == 0: del self.input_layernorm'\n"
                 "block and add a '# OUR-HEAD-PATCH' marker comment.")
    sys.exit("no input_layernorm deletion found -- this SGLang may already keep it.\n"
             "Add a '# OUR-HEAD-PATCH' marker comment so check_draft.py is satisfied.")
i = m.group("i")
s = s[:m.start()] + f"\n{i}# OUR-HEAD-PATCH: layer-0 input_layernorm RETAINED on purpose.\n" \
                    f"{i}# Our draft head has and uses it; deleting it changes the function\n" \
                    f"{i}# computed (relative error 1.84 vs 7.7e-17). Do not restore the delete.\n" \
         + s[m.end():]
p.write_text(s)
print("patched")
PY

python3 - "$F" <<'PY'
import sys, pathlib
s = pathlib.Path(sys.argv[1]).read_text()
ok = "OUR-HEAD-PATCH" in s and "del self.input_layernorm" not in s
print("verify:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
