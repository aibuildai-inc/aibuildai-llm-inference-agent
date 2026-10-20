"""Pre-flight for a submission, cheap and without a full benchmark.

Checks, in order, the four things that make a score meaningless if wrong:

  1. SGLang is importable and carries the layer-0 `input_layernorm` patch this task
     depends on. Upstream deletes that norm; the required architecture uses it, so an
     unpatched environment silently computes a different function (measured: relative
     error 1.84 unpatched versus 7.7e-17 patched).
  2. The head is the required architecture and packages into a draft directory.
  3. SGLang loads the draft and reports NO fallback.
  4. Speculation is actually RUNNING -- every request reports a non-zero
     `spec_verify_ct` in its meta_info. The test is "did it draft", NOT "did it draft
     well": an untrained head drafts correctly and has everything rejected, which is a
     real score near 1.0. What must be ruled out is a silently DISABLED speculator,
     which also scores ~1.0 and is otherwise indistinguishable from an honest null.
     Read per request, because SGLang logs aggregate spec metrics only on a ~10s timer
     and a short check emits none at all.

  python check_draft.py --submission DIR [--json OUT]
"""
from __future__ import annotations

import argparse
import atexit
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def check_patch() -> dict:
    """Is SGLang present, does it host this target's family, and is it patched?

    The EAGLE model file is per-architecture, so its name comes from models.json rather
    than being hardcoded here -- see sglang._draft_architecture_note.
    """
    cfg = json.loads((TASK / "models.json").read_text())
    model_type = cfg["sglang"]["draft_model_type"]
    try:
        import sglang
    except Exception as exc:                                  # noqa: BLE001
        return {"ok": False, "why": f"sglang not importable: {type(exc).__name__}: {exc}"}
    src = Path(sglang.__file__).parent / "srt" / "models" / f"{model_type}_eagle.py"
    if not src.is_file():
        return {"ok": False, "why": f"{src} missing: this SGLang has no EAGLE model for "
                                    f"model_type={model_type!r}"}
    text = src.read_text()
    patched = "OUR-HEAD-PATCH" in text
    deletes = "del self.input_layernorm" in text
    return {"ok": patched and not deletes, "sglang": sglang.__version__,
            "eagle_model_file": str(src),
            "why": "" if (patched and not deletes) else
                   (f"{src.name} still deletes layer-0 input_layernorm; apply the "
                    "task's patch before trusting any measurement")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    report: dict = {"patch": check_patch()}
    if not report["patch"]["ok"]:
        report["passed"] = False
        print(json.dumps(report, indent=2))
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(1)

    work = Path(tempfile.mkdtemp(prefix="sglang_check_"))
    # Every exit from here is `raise SystemExit`, so register the cleanup rather
    # than threading try/finally through each branch. Without this each call left
    # a 3.4 GB packaged draft behind: 16 of them had accumulated (54 GB), and the
    # projection across 4 runs x 10 steps was ~272 GB -- enough to fill the disk
    # and kill every concurrent run at once.
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    draft = work / "draft"
    pack = subprocess.run(
        [sys.executable, str(HERE / "make_draft.py"), "--submission", a.submission,
         "--models", str(TASK / "models.json"), "--out", str(draft)],
        capture_output=True, text=True)
    report["package"] = {"ok": pack.returncode == 0,
                         "detail": (pack.stdout or pack.stderr).strip()[-1200:]}
    if pack.returncode != 0:
        report["passed"] = False
        print(json.dumps(report, indent=2))
        if a.json:
            Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(1)

    # one short arm, 3 prompts, is enough to see loading, fallback and spec metrics
    out, log = work / "arm.json", work / "arm.log"
    cmd = [sys.executable, str(HERE / "_engine_arm.py"),
           "--models", str(TASK / "models.json"), "--prompts", str(TASK / "prompts.json"),
           "--out", str(out), "--draft", str(draft), "--repeats", "1", "--limit", "3"]
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")

    fallback = sorted({m.group(0) for m in re.finditer(
        r"(using the V1 model runner instead|[Aa]sync scheduling not supported|"
        r"falling back|speculative[^\n]{0,40}disabled)", text)})
    arm = json.loads(out.read_text()) if out.is_file() else {}

    report["load"] = {"ok": proc.returncode == 0 and out.is_file(),
                      "exit": proc.returncode}
    report["fallback_warnings"] = fallback
    report["spec_verify_ct"] = arm.get("spec_verify_ct", 0)
    report["tau"] = arm.get("tau", 0.0)
    report["tokens_per_second"] = arm.get("tokens_per_second", 0.0)
    # "did it draft", not "did it draft well": tau near 1.0 is a legitimate result for an
    # untrained head and must not be confused with a disabled speculator.
    report["speculation_active"] = report["spec_verify_ct"] > 0
    if not report["speculation_active"]:
        report["note"] = ("no request reported spec_verify_ct -- the speculator did not "
                          "run. A disabled speculator scores ~1.0 and looks like an "
                          "honest null, so the score cannot be trusted until this is "
                          "resolved.")
    report["passed"] = bool(report["load"]["ok"] and not fallback
                            and report["speculation_active"])
    print(json.dumps(report, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
