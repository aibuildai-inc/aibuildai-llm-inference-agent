"""Resolve the coordinated source version once into distribution metadata."""

from email.parser import Parser
from pathlib import Path
import os
import runpy

from setuptools import setup

root = Path(__file__).resolve().parent
metadata = root / "PKG-INFO"
if metadata.is_file():
    version = Parser().parsestr(metadata.read_text())["Version"]
else:
    version = os.environ.get("AIBUILDAI_BUILD_VERSION") or runpy.run_path(str(root.parents[1] / "aibuildai_version.py"))["APP_VERSION"]
setup(version=version)
