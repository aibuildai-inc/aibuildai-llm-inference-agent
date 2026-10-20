"""Build the coordinated package and its pinned Linux proxy asset.

Initial source builds require Go 1.26+ and the initialized CLIProxyAPI submodule. An sdist carries the compiled asset and does not need Go or the product checkout.
"""

from email.parser import Parser
from pathlib import Path
import hashlib
import json
import os
import platform
import runpy
import subprocess

from setuptools import Command, setup
from setuptools.command.build import build
from setuptools.command.sdist import sdist

root = Path(__file__).resolve().parent
metadata = root / "PKG-INFO"
assets = root / "src" / "aibuildai_conversation" / "_native"
asset_names = ("cli-proxy-api", "identity.json", "LICENSE")
proxy_version = "7.2.51"


class BuildRuntimeAssets(Command):
    user_options = []
    editable_mode = False

    def initialize_options(self) -> None:
        self.build_lib: str | None = None

    def finalize_options(self) -> None:
        self.set_undefined_options("build_py", ("build_lib", "build_lib"))

    def get_outputs(self) -> list[str]:
        if self.build_lib is None:
            raise RuntimeError("The build output directory has not been resolved")
        return [str(Path(self.build_lib) / "aibuildai_conversation" / "_native" / name) for name in asset_names]

    def get_output_mapping(self) -> dict[str, str]:
        return dict(zip(self.get_outputs(), (str(assets / name) for name in asset_names)))

    def get_source_files(self) -> list[str]:
        return [str(Path("src") / "aibuildai_conversation" / "_native" / name) for name in asset_names if (assets / name).is_file()]

    def run(self) -> None:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Conversation runtime assets require Linux x86_64")
        source = root.parents[1] / "submodules" / "cli-proxy-api"
        identity_path = assets / "identity.json"
        binary = assets / "cli-proxy-api"
        identity = json.loads(identity_path.read_text()) if identity_path.is_file() else None
        if metadata.is_file():
            commit = identity["commit"] if identity is not None else None
        else:
            product = root.parents[1]
            if (product / ".git").exists():
                commit = subprocess.check_output(["git", "-C", str(product), "rev-parse", "HEAD:submodules/cli-proxy-api"], text=True).strip()
                actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel", "HEAD"], text=True).splitlines()
                if actual != [str(source), commit]:
                    raise RuntimeError("Initialize the CLIProxyAPI submodule at the product gitlink before building")
                if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
                    raise RuntimeError("Build CLIProxyAPI from a clean pinned submodule")
            else:
                commit = os.environ.get("AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT")
            if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise RuntimeError("The build requires the exact CLIProxyAPI source commit")
        if identity is not None and binary.is_file() and identity["commit"] == commit and identity["version"] == proxy_version:
            with binary.open("rb") as stream:
                valid = hashlib.file_digest(stream, "sha256").hexdigest() == identity["sha256"]
            if valid and (assets / "LICENSE").is_file() and os.access(binary, os.X_OK):
                return
        if metadata.is_file():
            raise RuntimeError("The sdist is missing its declared proxy asset or its bytes differ")
        assets.mkdir(parents=True, exist_ok=True)
        flags = f"-s -w -X main.Version={proxy_version}+aibuildai.{commit} -X main.Commit={commit}"
        subprocess.run(["go", "build", "-buildvcs=false", "-trimpath", "-ldflags", flags, "-o", str(binary), "./cmd/server"], cwd=source, check=True, env={**os.environ, "CGO_ENABLED": "0"})
        binary.chmod(0o755)
        (assets / "LICENSE").write_bytes((source / "LICENSE").read_bytes())
        with binary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        identity_path.write_text(json.dumps({"commit": commit, "version": proxy_version, "sha256": digest}) + "\n")


class Build(build):
    sub_commands = [("build_runtime_assets", None), *build.sub_commands]


class Sdist(sdist):
    def run(self) -> None:
        self.run_command("build_runtime_assets")
        super().run()


if metadata.is_file():
    version = Parser().parsestr(metadata.read_text())["Version"]
else:
    version = os.environ.get("AIBUILDAI_BUILD_VERSION") or runpy.run_path(str(root.parents[1] / "aibuildai_version.py"))["APP_VERSION"]
setup(version=version, cmdclass={"build": Build, "build_runtime_assets": BuildRuntimeAssets, "sdist": Sdist}, options={"bdist_wheel": {"plat_name": "linux_x86_64"}})
