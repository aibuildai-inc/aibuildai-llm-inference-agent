"""Copy the trained head into the attempt directory the framework scores.

    python export.py <attempt_dir>

The task's scored layout is <OUTPUT_DIR>/full/ containing head.pt and
head_config.json directly. The framework passes the `full` directory itself as
<attempt_dir>; if a bare parent is passed we append the `full` leaf so the
produced layout always has a `full/` directory with the head in it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python export.py <attempt_dir>")
    cfg = yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))
    leaf = cfg["export"]["dirname"]
    src = HERE / cfg["train"]["output_dir"]
    head_pt = src / "head.pt"
    head_cfg = src / "head_config.json"
    if not head_pt.is_file() or not head_cfg.is_file():
        raise SystemExit(f"export.py: no trained head in {src} (run train.py first)")

    dest = Path(sys.argv[1]).resolve()
    if dest.name != leaf:
        dest = dest / leaf
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(head_pt, dest / "head.pt")
    shutil.copy2(head_cfg, dest / "head_config.json")
    # sanity: the config is valid json and names the geometry
    json.loads((dest / "head_config.json").read_text())
    print(f"exported {dest}/head.pt and head_config.json")


if __name__ == "__main__":
    main()
