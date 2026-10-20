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
# The guard's BODY may hold more than the delete -- 0.5.21 also rebinds
# input_layernorm to an identity lambda on the next line. Match the guard, then
# consume every following line indented deeper than the `if`, or the leftovers
# dangle at the wrong depth and the module stops parsing ("unexpected indent").
pat = re.compile(
    r"\n(?P<i>[ \t]*)# Skip the input_layernorm\n"
    r"(?:[ \t]*#[^\n]*\n)*"
    r"(?P<g>[ \t]*)if layer_id == 0:\n"
    r"(?P<body>(?:(?:[ \t]*\n)|(?:(?P=g)[ \t]+[^\n]*\n))+)")
m = pat.search(s)
if not m:
    if "del self.input_layernorm" in s:
        sys.exit("FOUND 'del self.input_layernorm' but not in the expected shape.\n"
                 "Patch by hand: delete the 'if layer_id == 0: del self.input_layernorm'\n"
                 "block and add a '# OUR-HEAD-PATCH' marker comment.")
    sys.exit("no input_layernorm deletion found -- this SGLang may already keep it.\n"
             "Add a '# OUR-HEAD-PATCH' marker comment so check_draft.py is satisfied.")
i = m.group("i")
if "del self.input_layernorm" not in m.group("body"):
    sys.exit("the layer_id == 0 guard does not delete input_layernorm; patch by hand.")
# The body pattern also swallows blank lines, which at the end of the guard are
# the separation before the next top-level class. Hand them back rather than
# leaving the replacement jammed against it.
body = m.group("body")
tail = body[len(body.rstrip("\n\t ")):]
trailing_blanks = "\n" * max(tail.count("\n") - 1, 0)
# `_ = layer_id` keeps the parameter used, since removing the guard was its only
# reader, and keeps the replacement a statement rather than bare comments.
s = s[:m.start()] + f"\n{i}# OUR-HEAD-PATCH: layer-0 input_layernorm RETAINED on purpose.\n" \
                    f"{i}# Our draft head has and uses it; deleting it changes the function\n" \
                    f"{i}# computed (relative error 1.84 vs 7.7e-17). Do not restore the delete.\n" \
                    f"{i}_ = layer_id\n" + trailing_blanks \
         + s[m.end():]
# Refuse to write a file that does not parse. The previous version checked only
# that the marker was present and the delete was gone, which a syntactically
# broken file satisfies -- and SGLang then drops the module from its registry
# silently, so the damage surfaced 20 minutes later inside the draft arm.
try:
    compile(s, str(p), "exec")
except SyntaxError as e:
    sys.exit(f"refusing to write: the result would not parse ({e.msg} at line {e.lineno}).\n"
             f"{p} is unchanged; its backup is {p}.orig")
p.write_text(s)
print("patched")
PY

python3 - "$F" <<'PY'
import sys, pathlib
s = pathlib.Path(sys.argv[1]).read_text()
ok = "OUR-HEAD-PATCH" in s and "del self.input_layernorm" not in s
if ok:
    # Text alone is not enough: the file must still parse AND still import, or
    # SGLang drops it from the model registry without failing.
    try:
        compile(s, sys.argv[1], "exec")
    except SyntaxError as e:
        print(f"verify: FAILED -- does not parse ({e.msg} at line {e.lineno})")
        print(f"        restore with: cp {sys.argv[1]}.orig {sys.argv[1]}")
        sys.exit(1)
    # The import check is only meaningful when the file we just edited IS the one
    # the interpreter would import. When patching a copy out of tree, importing
    # `sglang` would either fail for unrelated reasons or test a different file.
    patched = pathlib.Path(sys.argv[1]).resolve()
    try:
        import importlib.util
        spec = importlib.util.find_spec("sglang.srt.models.qwen2_eagle")
        installed = pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
    except Exception:
        installed = None
    if installed == patched:
        try:
            import importlib
            importlib.import_module("sglang.srt.models.qwen2_eagle")
            print("verify: imports and is the module SGLang will load")
        except Exception as e:
            print(f"verify: FAILED -- parses but does not import: {type(e).__name__}: {e}")
            print(f"        restore with: cp {sys.argv[1]}.orig {sys.argv[1]}")
            sys.exit(1)
    else:
        print(f"verify: parse-only (this file is not the installed sglang module)")
print("verify:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
