"""Pin the JIT toolchain and cache so a measurement never pays for compilation.

Two separate failure modes, both silent, both measured:

1. flashinfer JIT-compiles kernels on first use by shelling out to `ninja`, which shells
   out to a C++ compiler. Both live in this environment's bin/, but AIBuildAI invokes the
   program environment's python directly WITHOUT putting that bin/ on PATH -- so the
   compile dies with `No such file or directory: 'ninja'` or
   `cannot execute 'cc1plus'`, minutes into what looked like a working run.

2. The JIT cache location follows the environment. With TMPDIR unset, the cache landed
   somewhere cold and compilation happened INSIDE the timed window: the same head, the
   same code, measured **19.45 tok/s instead of 37.93** -- a 2x error from one unset
   variable. A warm, fixed cache is therefore part of the measuring instrument, not a
   convenience.

So this pins both: the interpreter's bin/ onto PATH, and the JIT caches to a stable
directory beside the task. Import it before constructing an SGLang engine. Idempotent.
"""
from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path


# NOT inside the task folder: AIBuildAI mounts that read-only for the whole run, so a
# cache there silently fails to populate and every engine recompiles. A host path outside
# it stays warm across steps, workers AND runs, so only the very first engine ever pays.
# Override with SPECDEC_JIT_CACHE if this host lays things out differently.
CACHE_ROOT = Path(
    os.environ.get("SPECDEC_JIT_CACHE")
    or os.path.join(tempfile.gettempdir(), "specdec-jit-cache")
)


def ensure(cache_root: Path | None = None) -> dict:
    bindir = Path(sys.executable).resolve().parent
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(bindir) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(bindir), *parts])

    found = {}
    for var, names in (("CXX", ("g++", "c++", "x86_64-conda-linux-gnu-g++")),
                       ("CC", ("gcc", "cc", "x86_64-conda-linux-gnu-gcc"))):
        if os.environ.get(var) and Path(os.environ[var]).exists():
            found[var] = os.environ[var]
            continue
        for n in names:
            cand = bindir / n
            if cand.exists():
                os.environ[var] = str(cand)
                found[var] = str(cand)
                break
    # nvcc invokes a plain `g++` from PATH to interrogate the host compiler, and
    # conda-forge's gxx_linux-64 ships only `x86_64-conda-linux-gnu-g++`. On a
    # machine with no system g++ that combination fails flashinfer's JIT with
    #     nvcc fatal : Failed to preprocess host compiler properties
    # even though CXX above points at a perfectly good compiler. Tell nvcc which
    # host compiler to use rather than relying on the name it finds on PATH.
    cxx = found.get("CXX")
    if cxx and "-ccbin" not in os.environ.get("NVCC_PREPEND_FLAGS", ""):
        prev = os.environ.get("NVCC_PREPEND_FLAGS", "").strip()
        os.environ["NVCC_PREPEND_FLAGS"] = (prev + " " if prev else "") + f"-ccbin {cxx}"
        found["NVCC_PREPEND_FLAGS"] = os.environ["NVCC_PREPEND_FLAGS"]

    # One warm cache for every JIT that matters here. Set only if the caller has not
    # already chosen a location.
    root = Path(cache_root or CACHE_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        for var, sub in (("TMPDIR", "tmp"),
                         ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
                         ("SGLANG_CACHE_DIR", "sglang"),
                         ("TRITON_CACHE_DIR", "triton"),
                         ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
            if not os.environ.get(var):
                d = root / sub
                d.mkdir(parents=True, exist_ok=True)
                os.environ[var] = str(d)
            found[var] = os.environ[var]
        found["cache_root"] = str(root)
    except OSError as exc:
        # Not fatal -- compilation still works -- but it means every engine recompiles and
        # a measurement can absorb that cost, so say so loudly rather than degrade quietly.
        found["cache_root"] = f"<UNWRITABLE, measurements will be cold: {exc}>"
        print(f"WARNING: JIT cache root {root} is not writable ({exc}). Engines will "
              f"recompile and timings may be inflated. Set SPECDEC_JIT_CACHE to a "
              f"writable path.", file=sys.stderr)

    found["PATH_head"] = str(bindir)
    found["ninja"] = str(bindir / "ninja") if (bindir / "ninja").exists() else "<missing>"
    return found
