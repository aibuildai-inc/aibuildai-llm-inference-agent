"""Score a submission: SGLang tok/s WITH its draft head over SGLang tok/s WITHOUT.

This is the authoritative measurement. It replaces the eager task's
"speedup over a HuggingFace generate() loop", which rewarded work a real serving stack
discards -- two earlier heads scored well there and cannot be loaded by any framework.

Both arms run the SAME SGLang version with the SAME engine settings on the SAME prompts;
only `speculative_*` differs. Each arm is a separate process because two 14B engines
cannot be resident at once, so the two cannot be interleaved -- instead both are measured
back to back within one scoring call, and each arm reports its own sweep spread so drift
is visible rather than hidden.

Prints one JSON object on stdout. Exit 0 means "measurement completed", not "good":
read the JSON.

  python benchmark.py --submission DIR [--json OUT] [--keep-draft]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent

FALLBACK_PATTERNS = [
    r"using the V1 model runner instead",
    r"[Aa]sync scheduling not supported",
    r"falling back",
    r"fall ?back to eager",
    r"speculative.*disabled",
    r"draft.*failed",
]


def run_arm(models: Path, prompts: Path, out: Path, log: Path, draft: Path | None,
            repeats: int) -> dict:
    cmd = [sys.executable, str(HERE / "_engine_arm.py"),
           "--models", str(models), "--prompts", str(prompts),
           "--out", str(out), "--repeats", str(repeats)]
    if draft is not None:
        cmd += ["--draft", str(draft)]
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        raise SystemExit(f"{'draft' if draft else 'plain'} arm failed "
                         f"(exit {proc.returncode}):\n{tail}")
    return json.loads(out.read_text())


def warnings_in(log: Path) -> list[str]:
    text = log.read_text(errors="replace")
    hits = []
    for pat in FALLBACK_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            line = text[text.rfind("\n", 0, m.start()) + 1:
                        text.find("\n", m.end()) if text.find("\n", m.end()) > 0 else None]
            hits.append(line.strip()[:200])
    return sorted(set(hits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="sweeps per arm. 3 is the floor for a real median: "
                         "statistics.median of TWO values is their mean, so "
                         "--repeats 2 rejects no outlier and one stalled sweep "
                         "moves the reported number by half the stall")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-draft", action="store_true")
    a = ap.parse_args()

    models, prompts = TASK / "models.json", TASK / "prompts.json"
    work = Path(tempfile.mkdtemp(prefix="sglang_score_"))
    draft = work / "draft"

    pack = subprocess.run(
        [sys.executable, str(HERE / "make_draft.py"), "--submission", a.submission,
         "--models", str(models), "--out", str(draft)],
        capture_output=True, text=True)
    if pack.returncode != 0:
        # An unloadable head is a zero, and the reason is the useful part.
        result = {"speedup": 0.0, "gate_passed": False,
                  "reason": (pack.stdout + pack.stderr).strip()[-1500:]}
        print(json.dumps(result, indent=2))
        if a.json:
            Path(a.json).write_text(json.dumps(result, indent=2) + "\n")
        return
    pack_info = json.loads(pack.stdout)

    # tunables the submission may propose, within the bounds models.json declares
    ov = Path(a.submission, "head_config.json")
    if ov.is_file():
        hc = json.loads(ov.read_text())
        if isinstance(hc.get("sglang_overrides"), dict):
            (draft / "_overrides.json").write_text(json.dumps(hc["sglang_overrides"]))

    plain_log, draft_log = work / "plain.log", work / "draft.log"
    plain = run_arm(models, prompts, work / "plain.json", plain_log, None, a.repeats)
    drafted = run_arm(models, prompts, work / "draft.json", draft_log, draft, a.repeats)

    base = plain["tokens_per_second"]
    meth = drafted["tokens_per_second"]
    speedup = (meth / base) if base else 0.0

    # Exactness note: under greedy decoding SGLang's verify kernel only ever emits the
    # TARGET's own argmax (draft_token == target_token is the accept condition), so a
    # proposer cannot change the output. bf16 ties can still round differently between a
    # cached single-token decode and a batched tree verify, so token streams are REPORTED,
    # not required to match -- see check_draft.py for what is actually enforced.
    identical = sum(1 for p, d in zip(plain.get("output_ids") or [],
                                      drafted.get("output_ids") or []) if p == d)

    fallbacks = warnings_in(plain_log) + warnings_in(draft_log)
    active = drafted.get("spec_verify_ct", 0) > 0
    result = {
        "speedup": speedup,
        # A silently disabled speculator measures ~1.0 and would otherwise read as a
        # genuine null, so "it actually drafted" is part of the gate, not a footnote.
        "gate_passed": (not fallbacks) and active,
        "method_tokens_per_second": meth,
        "baseline_tokens_per_second": base,
        "method_sweeps": drafted["tokens_per_second_sweeps"],
        "baseline_sweeps": plain["tokens_per_second_sweeps"],
        "tau": drafted.get("tau", 0.0),
        "spec_verify_ct": drafted.get("spec_verify_ct", 0),
        "speculation_active": drafted.get("spec_verify_ct", 0) > 0,
        "identical_prompts": identical,
        "prompts": len(plain.get("output_ids") or []),
        "draft": pack_info,
        "fallback_warnings": fallbacks,
        "engine_args_plain": plain["engine_args"],
        "engine_args_draft": drafted["engine_args"],
        "protocol": ("batch 1, greedy, same ids both arms, separate processes, "
                     "median of sweeps; ratio of SGLang tok/s with draft to without"),
    }
    print(json.dumps(result, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2) + "\n")
    if not a.keep_draft:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
