"""Build-time tool: Cython-compile a staged source tree's product modules to native ``.so``, in place, dropping the plaintext ``.py``.

Run with ONE argument — the staging tree root produced by release.sh's ``build_runtime_tree`` (every ``.py`` plus the build-generated bundles already present). It:

  - imports the runtime manifest FROM the staging tree (``aibuildai_version.RUNTIME_PACKAGES`` + ``aibuildai_version.RUNTIME_TOP_LEVEL_MODULES``);
  - collects every ``*.py`` under each runtime package RECURSIVELY (including ``__init__.py``, the generated resource bundles, and the generated ``startup/_plugin_bundle/`` chunk package), plus the top-level modules, EXCEPT what must not become a ``.so``:
      * ``cli.py`` — the PyInstaller entry point.
      * ``engine/run_state.py`` — eventsourcing reads its ``__init__`` as a Python function. Cython changes it into a different function type.
      * anything under a ``resources/`` directory — those files are shipped text, not modules. Nothing imports them; the product reads them and copies them out at runtime. The plugin bundle is no longer excluded: ``build_plugin_bundle.py`` now emits it as a package of bounded ``_chunk_NN.py`` modules (the single-module form expanded to a ~500 MB ``.c`` gcc could not build), so every chunk compiles to ``.so`` here like the rest of the tree;
  - strips docstrings from the generated C (``Cython.Compiler.Options.docstrings = False``) so the shipped ``.so`` files carry no plaintext module/class/function design commentary — without this, ``strings`` on any product ``.so`` reads back the source docstrings (#673). LLM-facing pydantic model descriptions are NOT lost: they live in ``json_schema_extra`` string literals in Agent IO and other schema-bearing modules, which survive the strip;
  - cythonizes each module with ``compiler_directives={"language_level": "3", "annotation_typing": False}`` — ``annotation_typing=False`` is MANDATORY: the Cython default of True makes a builtin-type annotation a RUNTIME type check that rejects subclasses (e.g. a ``str``-Enum passed where ``str`` is annotated), which breaks the product;
  - adds a NULL guard to the generated C of every module that carries an async generator, in Cython's ``aclose()``-mode ``throw`` tail, so a throw that makes the generator suspend again inside its cleanup no longer dereferences a NULL ``PyErr_Occurred()`` and kills the process with SIGSEGV (#3065; the pattern above names the Cython text measured on 3.0.12 through 3.3.0 and fails the build when a Cython no longer emits it);
  - builds the extensions in place with a parallel ``build_ext``;
  - renames each successfully-compiled module's ``.py`` -> ``.py.bak`` (the ``.so`` then wins on import, and ``aibuildai.spec``'s existing ``.py.bak`` AST-scan handling reads the renamed source for its third-party import scan).

Fail-fast: ANY module that fails to cythonize (``.py`` -> ``.c``) or fails the gcc build aborts the whole script with a non-zero exit naming the offending module(s). There is no silent skip and no error-swallowing — a failure here is a real signal that a module is not Cython-clean.

All paths are derived from the argv stage root and ``aibuildai_version.py`` (no personal/host paths); the file is force-tracked under ``scripts/``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options as CythonOptions
from setuptools import Distribution, Extension

# These modules must stay as Python.
#   cli.py: the PyInstaller entry point, must stay a plain script.
#   engine/run_state.py: eventsourcing reads RunState.__init__ as a
#   Python function. A Cython function breaks that check.
#   engine/mcp.py: AnyIO rejects Cython coroutine objects passed to start_soon.
# The plugin data bundle USED to be excluded here for the same reason cli is
# kept out (gcc could not build the single ~500 MB .c it expanded to). It no
# longer is: build_plugin_bundle.py now emits it as a package of bounded
# ``startup/_plugin_bundle/_chunk_NN.py`` modules, each of which compiles to .so
# cleanly, so the whole bundle is Cython-compiled like the rest of the tree.
_SKIP_RELATIVE_PATHS = frozenset({"cli.py", "engine/mcp.py", "engine/run_state.py"})

# Cython 3.0.12, 3.1.6, 3.2.2, 3.2.5, and 3.3.0 (every release checked) emit this ``aclose()``-mode
# tail in ``__Pyx_async_gen_athrow_throw`` (Utility/AsyncGen.c). When the
# exception thrown into a suspended async generator's ``aclose()`` awaitable
# makes the generator SUSPEND AGAIN inside its cleanup (it awaits, catches the
# error, and awaits once more), ``__Pyx_Coroutine_Throw`` returns the new
# awaitable with no exception set, so ``PyErr_Occurred()`` is NULL and
# ``__Pyx_PyErr_GivenExceptionMatches2`` dereferences it: SIGSEGV. CPython's
# own ``async_gen_athrow_throw`` tolerates the NULL. asyncio reaches this path
# through its async-generator finalizer hook, which schedules ``aclose()`` as a
# task and then cancels it or feeds it a failed future; a Science run died this
# way with status 11 the moment its setup Agent's CLI exited (#3065). The
# minimal reproduction and the crash trace are on that issue. Every product
# module is compiled here, so the guard is applied to the generated C of every
# module that carries the function, and a module that carries it WITHOUT the
# expected tail fails the build: a new Cython release changed the code and
# must be re-examined, not silently shipped.
#
# The tail's spelling drifts across releases and the pattern absorbs the
# measured drift: 3.0.12 writes ``__Pyx_PyExc_StopAsyncIteration``, 3.1.6 and
# 3.3.0 write ``PyExc_StopAsyncIteration`` at eight spaces, and 3.2.2 and 3.2.5
# indent the whole block by ten. The pattern binds the indent once and reuses it, so a
# tail whose lines do not share one indent is not recognized.
_ASYNC_GEN_ATHROW_THROW = "__Pyx_async_gen_athrow_throw("
_ACLOSE_THROW_TAIL = re.compile(
    r"(?P<indent>[ \t]+)exc_type = PyErr_Occurred\(\);\n"
    r"(?P=indent)if \((?P<call>__Pyx_PyErr_GivenExceptionMatches2\(exc_type, "
    r"(?:__Pyx_)?PyExc_StopAsyncIteration, (?:__Pyx_)?PyExc_GeneratorExit\))\) \{\n"
    r"(?P=indent)    PyErr_Clear\(\);\n"
    r"(?P=indent)    PyErr_SetNone\((?:__Pyx_)?PyExc_StopIteration\);\n"
    r"(?P=indent)\}\n"
    r"(?P=indent)return retval;\n"
)
_GUARDED_ACLOSE_THROW_MARK = "if (exc_type && __Pyx_PyErr_GivenExceptionMatches2(exc_type, "


def guard_async_gen_aclose_throw(c_path: Path) -> bool:
    """Add the NULL guard to a generated C file's ``aclose()`` throw tail.

    Returns True when the file carries ``__Pyx_async_gen_athrow_throw`` and was
    patched, False when it has no async generator at all. Raises when the
    function is present but its tail is not the one this guard knows.
    """
    source = c_path.read_text(encoding="utf-8")
    if _ASYNC_GEN_ATHROW_THROW not in source:
        return False
    if _GUARDED_ACLOSE_THROW_MARK in source:
        return True
    matches = list(_ACLOSE_THROW_TAIL.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"{c_path}: __Pyx_async_gen_athrow_throw is present but its aclose() "
            f"tail matched {len(matches)} times, not once; the installed Cython "
            "no longer emits the tail this build guards against the NULL "
            "PyErr_Occurred() dereference (#3065); re-examine it before shipping"
        )
    match = matches[0]
    guarded = match.group(0).replace(
        f"if ({match.group('call')})", f"if (exc_type && {match.group('call')})", 1
    )
    c_path.write_text(
        source[: match.start()] + guarded + source[match.end() :], encoding="utf-8"
    )
    return True




def _stays_python(stage_root: Path, path: Path) -> bool:
    """Is this staged ``*.py`` one Cython must leave alone?

    Two reasons, and they are different. ``_SKIP_RELATIVE_PATHS`` names modules whose runtime needs Python function behavior. A file under ``resources/`` is not a module at all -- it is text the product reads and copies (``metric_recorder.py`` reads ``work_unit/resources/aibuildai.py`` with ``read_text``), so compiling it renames away the file the binary is built to ship.
    """
    relative = path.relative_to(stage_root)
    return relative.as_posix() in _SKIP_RELATIVE_PATHS or "resources" in relative.parts[:-1]

_BUILD_PARALLELISM = 16


def collect_modules(stage_root: Path) -> tuple[list[Path], list[str]]:
    """Return ``(modules_to_compile, left_as_python)`` under ``stage_root``.

    Imports the runtime manifest from the staging tree, then collects each runtime package's ``*.py`` recursively (``__init__.py``, the generated resource bundles, and the ``_plugin_bundle`` chunk package included) plus the top-level modules. Whatever ``_stays_python`` excludes is returned in the second list rather than dropped, so the build reports every file it left alone by name. Both lists are sorted and unique.
    """
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    from aibuildai_version import RUNTIME_PACKAGES, RUNTIME_TOP_LEVEL_MODULES

    collected: set[Path] = set()
    left: set[str] = set()
    for relative in RUNTIME_TOP_LEVEL_MODULES:
        path = stage_root / relative
        if _stays_python(stage_root, path):
            left.add(Path(relative).as_posix())
            continue
        if not path.exists():
            raise FileNotFoundError(f"runtime top-level module missing from stage: {relative}")
        collected.add(path)

    for package in RUNTIME_PACKAGES:
        package_dir = stage_root / package
        if not package_dir.is_dir():
            raise FileNotFoundError(f"runtime package missing from stage: {package}")
        for path in package_dir.rglob("*.py"):
            if _stays_python(stage_root, path):
                left.add(path.relative_to(stage_root).as_posix())
                continue
            collected.add(path)

    return sorted(collected), sorted(left)


def _module_name(stage_root: Path, py_path: Path) -> str:
    """Dotted module name of ``py_path`` relative to ``stage_root``."""
    return ".".join(py_path.relative_to(stage_root).with_suffix("").parts)


def compile_tree(stage_root: Path) -> tuple[list[str], list[str], list[str]]:
    """Cythonize, build, and rename every collected module under ``stage_root``.

    Returns ``(compiled_module_names, skipped_relative_paths, aclose_throw_guarded_module_names)``. Raises on the first cythonize failure (named) and aggregates per-module gcc-build outcomes by checking for the produced ``.so``, raising if any module lacks one.
    """
    # cythonize resolves Extension source paths, and build_ext --inplace places
    # each .so, RELATIVE TO THE CWD. release.sh invokes this script from the
    # source checkout, so operate from the staging tree instead: otherwise the
    # stage-only generated bundles (engine/infra/memory _resource_bundle.py)
    # are not found, and the .c/.so would land in (and corrupt) the source tree.
    os.chdir(stage_root)
    # Drop docstrings from the generated C so shipped `.so` files carry no
    # plaintext module/class/function design commentary (#673). Cython reads
    # this module-global at `cythonize()` time, so set it before the loop.
    CythonOptions.docstrings = False
    py_files, skipped = collect_modules(stage_root)

    ext_modules = []
    cythonize_failures: list[str] = []
    guarded: list[str] = []
    for py_path in py_files:
        modname = _module_name(stage_root, py_path)
        source_rel = py_path.relative_to(stage_root).as_posix()
        try:
            cythonized = cythonize(
                [Extension(modname, [source_rel])],
                compiler_directives={"language_level": "3", "annotation_typing": False},
                quiet=True,
                force=True,
            )
        except Exception as exc:
            cythonize_failures.append(f"{modname}: {type(exc).__name__}: {exc}")
            continue
        for ext in cythonized:
            for source in ext.sources:
                if source.endswith(".c") and guard_async_gen_aclose_throw(Path(source)):
                    guarded.append(modname)
        ext_modules.extend(cythonized)

    if cythonize_failures:
        joined = "\n  ".join(cythonize_failures)
        raise RuntimeError(
            "Cython failed to translate the following module(s) to C "
            f"(the .py is NOT Cython-clean):\n  {joined}"
        )

    # build_ext's `inplace` (put each .so next to its source so the rename below
    # finds it) and `parallel` (speed the gcc pass) are distutils dynamic
    # options pyright cannot type on the Command class, so set them with setattr
    # (no static attribute-access check) on the finalized command object.
    distribution = Distribution({"ext_modules": ext_modules})
    command = distribution.get_command_obj("build_ext")
    setattr(command, "inplace", 1)
    setattr(command, "parallel", _BUILD_PARALLELISM)
    command.ensure_finalized()
    command.run()

    compiled: list[str] = []
    build_failures: list[str] = []
    for ext in ext_modules:
        module_path = stage_root / ext.name.replace(".", "/")
        produced_so = list(module_path.parent.glob(module_path.name + ".*.so"))
        py_path = module_path.with_suffix(".py")
        if not produced_so:
            build_failures.append(ext.name)
            continue
        py_path.rename(py_path.with_suffix(".py.bak"))
        compiled.append(ext.name)

    if build_failures:
        joined = "\n  ".join(sorted(build_failures))
        raise RuntimeError(
            "gcc/build_ext produced no .so for the following module(s):\n  " + joined
        )

    return sorted(compiled), skipped, sorted(guarded)


def main(stage_root: Path) -> None:
    compiled, skipped, guarded = compile_tree(stage_root)
    print(f"Cython compile complete: {len(compiled)} module(s) -> native .so")
    print("Skipped (ship as source .py): " + ", ".join(skipped))
    print(
        "Async-generator aclose() throw guard applied (#3065): "
        + ", ".join(guarded)
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <stage_root>", file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]).resolve())
