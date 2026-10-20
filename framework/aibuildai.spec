# -*- mode: python ; coding: utf-8 -*-
import glob
import ast
import importlib.util
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

project_root = os.path.abspath(os.getcwd())
source_scan_root = os.path.abspath(os.environ.get("AIBUILDAI_AST_ROOT", project_root))

# Import the runtime manifest from aibuildai_version.py — single source of truth shared
# with release.sh. The build tree may not have aibuildai_version.py on sys.path when
# PyInstaller runs, so inject source_scan_root explicitly.
if source_scan_root not in sys.path:
    sys.path.insert(0, source_scan_root)
from aibuildai_conversation.assets import claude_path, proxy_path

from aibuildai_version import (
    RUNTIME_TOP_LEVEL_MODULES,
    RUNTIME_PACKAGES,
    RUNTIME_SUBMODULE_PATHS,
    RUNTIME_ROOT_DATA_FILES,
)

# PyJWT verifies RS256 id_tokens via cryptography's C backend, which the AST
# import scan cannot see (we never `import cryptography` directly); jwt's
# algorithm submodules load lazily. Collect both for the subscription check.
crypto_hiddenimports = collect_submodules("cryptography")
jwt_hiddenimports = collect_submodules("jwt")
eventsourcing_sqlalchemy_hiddenimports = collect_submodules("eventsourcing_sqlalchemy")
local_root_modules = (
    {Path(m).stem for m in RUNTIME_TOP_LEVEL_MODULES}
    | set(RUNTIME_PACKAGES)
)


def _compiled_package_modules(package_name):
    # A compiled .so cannot be AST-scanned by PyInstaller, so EVERY first-party
    # submodule at any depth must be declared as an explicit hiddenimport.
    # Recurse the whole package and build the dotted module name from the path
    # RELATIVE to project_root; an __init__ maps to its package's own dotted name.
    package_dir = Path(project_root) / package_name
    modules = set()
    for pattern in ("*.py", "*.py.bak", "*.so"):
        for path in package_dir.rglob(pattern):
            relative = path.relative_to(project_root)
            parts = list(relative.parts)
            parts[-1] = parts[-1].split(".", 1)[0]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            modules.add(".".join(parts))
    return sorted(modules)


def _runtime_source_files():
    files = []
    for relative_path in RUNTIME_TOP_LEVEL_MODULES:
        path = Path(source_scan_root) / relative_path
        if path.exists():
            files.append(path)
        elif path.with_name(path.name + ".bak").exists():
            files.append(path.with_name(path.name + ".bak"))
    for package in RUNTIME_PACKAGES:
        package_dir = Path(source_scan_root) / package
        for pattern in ("*.py", "*.py.bak"):
            files.extend(sorted(package_dir.rglob(pattern)))
    return files


def _third_party_hiddenimports():
    modules = set()
    for path in _runtime_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                candidates = []
                if node.module:
                    candidates.append(node.module)
                    root_name = node.module.split(".", 1)[0]
                    if (
                        root_name not in local_root_modules
                        and root_name not in sys.stdlib_module_names
                    ):
                        module_spec = importlib.util.find_spec(node.module)
                        if (
                            module_spec is not None
                            and module_spec.submodule_search_locations is not None
                        ):
                            candidates.extend(
                                f"{node.module}.{alias.name}"
                                for alias in node.names
                                if importlib.util.find_spec(
                                    f"{node.module}.{alias.name}"
                                )
                                is not None
                            )
            else:
                continue

            for module_name in candidates:
                if not module_name or module_name == "__future__":
                    continue
                root_name = module_name.split(".", 1)[0]
                if root_name in local_root_modules or root_name in sys.stdlib_module_names:
                    continue
                modules.add(module_name)
    return sorted(modules)

# Shipped small text resources. Executables are NOT bundled here: every
# AIBuildAI-owned executable ships in the one tools/ directory release.sh
# stages next to the installed binary (infra.owned_tools resolves it there).
# The document scaffold is not listed here either: it ships inside the built-in
# plugin tree with the rest of the research-paper-writing skill.
# The WorkUnit metric recorder ships as a text resource beside its owner.
# metric_recorder.py reads this same package-relative path in source and frozen modes.
_metric_recorder_src = (
    Path(project_root) / "engine" / "work_unit" / "resources" / "aibuildai.py"
)
resource_data_files = [
    (str(_metric_recorder_src), str(Path("engine") / "work_unit" / "resources"))
]

# Repo-root data files bundled into the _internal/ bundle root (target "."),
# driven by aibuildai_version.py RUNTIME_ROOT_DATA_FILES (now empty). config.py no longer
# reads example.yaml at runtime — it generates the text directly.
root_data_files = [
    (os.path.join(project_root, name), ".")
    for name in RUNTIME_ROOT_DATA_FILES
]

# The built Web display frontend (output/web/frontend/dist, committed as a
# generated tree). Bundled verbatim so the Web Workspace needs no Node.js at
# runtime; output.web.app.frontend_dist_dir resolves this exact target under
# _MEIPASS, and verify_frontend_assets fails the run fast when it is absent.
web_data_files = [
    (
        os.path.join(project_root, "output", "web", "frontend", "dist"),
        "output/web/frontend/dist",
    ),
    (
        os.path.join(project_root, "output", "web", "frontend", "src", "api", "schema.hash"),
        "output/web/frontend/src/api",
    ),
]

_non_entry_top_level = [
    Path(m).stem for m in RUNTIME_TOP_LEVEL_MODULES if Path(m).stem != "cli"
]
project_hiddenimports = list(_non_entry_top_level)
for package in RUNTIME_PACKAGES:
    project_hiddenimports.extend(_compiled_package_modules(package))

third_party_hiddenimports = _third_party_hiddenimports()

# Pinned third-party submodule sources (aibuildai_version.py RUNTIME_SUBMODULE_PATHS) join the
# search path so PyInstaller can RESOLVE the imports it already DISCOVERS: the AST scan
# in _third_party_hiddenimports() collects such a root because it is neither a local
# module nor stdlib, but without its directory here it resolves nowhere and the binary
# raises ModuleNotFoundError on the user's first run. The list is empty today.
submodule_pathex = [
    os.path.join(project_root, relative) for relative in RUNTIME_SUBMODULE_PATHS
]

a = Analysis(
    ["cli.py"],
    pathex=[project_root] + submodule_pathex,
    binaries=[(str(claude_path()), "claude_agent_sdk/_bundled"), (str(proxy_path()), "aibuildai_conversation/_native")],
    datas=root_data_files + resource_data_files + web_data_files + collect_data_files("aibuildai_conversation", excludes=["_native/cli-proxy-api"]),
    hiddenimports=[
        # Runtime project modules are release-packaged from a staged source tree.
        # PyInstaller does not always discover every dynamically-referenced import,
        # so collect the local packages explicitly.
        "claude_agent_sdk",
        # Build-time version injection: release.sh writes a plain `_version.py`
        # into this tree that aibuildai_version.py imports at runtime. Declared explicitly
        # because the import is resolved at runtime, not statically visible.
        "_version",
    ] + project_hiddenimports + third_party_hiddenimports + crypto_hiddenimports + jwt_hiddenimports + eventsourcing_sqlalchemy_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "transformers",
        "tensorflow",
        "tensorboard",
        "nvidia",
        "datasets",
        "mlebench",
        # IPython is installed only in the build host's dev environment and
        # would also collect Tk and other development tools.
        "IPython",
        # These are installed only in the build host's dev environment. No
        # shipped source imports them, and the runtime lock does not list them.
        "numpy",
        "psutil",
        # urllib3 2.7 imports `backports.zstd` for optional HTTP zstd decoding,
        # guarded by `except ImportError`. PyInstaller bundles only the package's
        # `_zstd` C extension (not the wrapper that defines `ZstdError`), so the
        # import SUCCEEDS but `zstd.ZstdError` raises AttributeError at urllib3
        # import time -> the binary crashes on the first model_catalog import
        # (every `run`/`config`). Excluding it makes the import fail cleanly, so
        # urllib3 sets HAS_ZSTD=False and runs without zstd (the product's HTTP
        # calls never use zstd). Verified: urllib3.response imports fine with
        # backports.zstd absent.
        "backports.zstd",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aibuildai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["cli-proxy-api"],
    name="aibuildai",
)
