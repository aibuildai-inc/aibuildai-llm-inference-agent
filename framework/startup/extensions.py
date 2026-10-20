"""Load user Search classes during startup."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import features
from config import StartupUserInputError


class ExtensionImportError(StartupUserInputError):
    """A user extension file failed while Python imported it."""

    def __init__(self, path: Path, cause: BaseException) -> None:
        self.path = path
        self.cause = cause
        super().__init__(
            f"cannot import your extension file {path}: {type(cause).__name__}: {cause}"
        )


_IMPORTED_EXTENSION_FILES: set[Path] = set()


def import_extension_classes(extensions_dir: str) -> None:
    """Load user Search classes before state is restored."""
    if "extensions" in features.hidden_config_fields():
        return  # this build does not offer the plugin point, so nothing is loaded
    ext_dir = Path(extensions_dir).expanduser() / "searches"
    if not ext_dir.is_dir():
        return
    for path in sorted(ext_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        source = path.resolve()
        if source in _IMPORTED_EXTENSION_FILES:
            continue
        module_name = f"aibuildai_user_searches_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ExtensionImportError(path, ImportError(f"cannot load {path}"))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 -- user code can raise any error
            raise ExtensionImportError(path, exc) from exc
        _IMPORTED_EXTENSION_FILES.add(source)
