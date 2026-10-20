"""Shared helpers for the speculative-decoding harness.

Loaded by make_draft.py, gate_eager_reference.py and fetch_models.py. Nothing here
imports the
candidate's code; the submission is always loaded through load_submission().
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompts(path: str | Path, limit: int | None = None) -> list[str]:
    """Return the frozen prompt list, in file order (never shuffled)."""
    prompts = read_json(path)["prompts"]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts loaded from {path}")
    return prompts


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str, dtype: str, device: str):
    """Load one model in the requested dtype onto exactly one device.

    Never device_map="auto": sharding across cards invalidates latency numbers.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPES[dtype])
    model.to(device)
    model.eval()
    return model


def stop_token_ids(model, tokenizer) -> list[int]:
    """Every id that ends generation, from the generation config, not just eos_token_id."""
    ids: list[int] = []
    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.append(configured)
    elif configured:
        ids.extend(int(value) for value in configured)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return sorted(set(ids))


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """One chat-formatted prompt as ids. Baseline and submission get these same ids."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors=None)["input_ids"]


def sync(device: str) -> None:
    """CUDA is async; without this you time the launch queue, not the work."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_submission(submission_dir: str | Path):
    """Import <submission_dir>/submission.py and return its Method class."""
    path = Path(submission_dir) / "submission.py"
    if not path.is_file():
        raise FileNotFoundError(f"submission.py not found in {submission_dir}")
    spec = importlib.util.spec_from_file_location("_specdec_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method = getattr(module, "Method", None)
    if method is None:
        raise RuntimeError("submission.py must define a class named Method")
    return method


def method_config(pair: dict, device: str, submission_dir: str, eos_ids: list[int]) -> dict:
    return {
        "target": pair["target"],
        "draft": pair.get("draft"),   # None = the submission picks its own proposer
        "dtype": pair["dtype"],
        "device": device,
        "assets_dir": str(submission_dir),
        "eos_ids": list(eos_ids),
    }


def offline_env_note() -> str:
    return (
        "HF_HUB_OFFLINE=%s" % os.environ.get("HF_HUB_OFFLINE", "0")
    )
