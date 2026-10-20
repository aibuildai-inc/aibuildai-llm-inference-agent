"""Download every model in models.json and prove it loads offline.

SETUP runs this once. Scoring runs later with no network, so a model that is
not cached here makes every scored step fail.

  python fetch_models.py [--models models.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(TASK / "models.json"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    config = common.read_json(args.models)
    wanted: list[str] = []
    for key in ("speed_pair", "gate_pair"):
        for role in ("target", "draft"):
            model_id = config[key].get(role)
            # draft is null when the task leaves the proposer open; there is
            # nothing to fetch for it.
            if model_id and model_id not in wanted:
                wanted.append(model_id)

    report = []
    for model_id in wanted:
        path = snapshot_download(model_id)
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True)
        report.append({"model": model_id, "path": path, "offline_load": "ok"})
        print(f"cached {model_id}", flush=True)

    print(json.dumps({"models": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
