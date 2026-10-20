"""The post-training codebase every AutoResearch run starts from.

Held as a string map so a source run and the frozen binary write the same bytes
and neither reads a developer checkout. Once written the files are the
project's own source, and Workers rewrite them like any other file. The map IS
the shipped tree, in the order a step runs it: the initializer writes exactly
these names and nothing else.
"""

from __future__ import annotations

POST_TRAIN_FILES: dict[str, str] = {
    "recipe.yaml": '''\
# The research project's own configuration, and ordinary source: change any
# value, add keys, or delete what a step you rewrote no longer reads. Check
# `base_model` and `data` against the task README before the first run -- the
# README names the model this task post-trains and the rules its data must
# obey, and the values here are only a runnable starting point.
base_model: Qwen/Qwen3-1.7B-Base
# A Jinja chat template file to install on the tokenizer, or null for the
# tokenizer's own. A base model often ships none and the grader usually
# evaluates through one, so point this at the template the task provides.
chat_template: null
data:
  dataset: openai/gsm8k
  name: main
  split: train
  prompt_field: question
  completion_field: answer
  max_samples: 4000
  max_prompt_chars: 1200
  max_completion_chars: 1200
  eval_samples: 100
  system_prompt: "Solve the problem step by step. Finish with the final answer on its own last line, written as '#### <answer>'."
  prepared: data
sft:
  enabled: true
  epochs: 1
  learning_rate: 1.0e-5
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 1024
  bf16: true
  gradient_checkpointing: true
  output_dir: runs/sft
  lora:
    enabled: false
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
# Reinforcement learning on top of the SFT model. Off by default: SFT alone is
# a complete result, and GRPO costs far more per step.
grpo:
  enabled: false
  steps: 100
  learning_rate: 1.0e-6
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  num_generations: 4
  max_prompt_length: 512
  max_completion_length: 512
  temperature: 1.0
  prompts: 512
  output_dir: runs/grpo
eval:
  samples: 100
  max_new_tokens: 512
  batch_size: 8
export:
  dirname: final_model
''',
    "prepare.py": '''\
"""Turn the recipe's dataset into the records train.py and eval.py read.

    python prepare.py

Writes <data.prepared>/train.jsonl and <data.prepared>/dev.jsonl, each line one
conversational prompt/completion pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from datasets import load_dataset

HERE = Path(__file__).resolve().parent


def recipe() -> dict:
    return yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))


def records(config: dict) -> list[dict]:
    """Every usable row of the configured dataset, in prompt/completion shape."""
    data = config["data"]
    rows = load_dataset(data["dataset"], data.get("name"), split=data["split"])
    system = data.get("system_prompt") or ""
    out: list[dict] = []
    for row in rows:
        prompt = str(row[data["prompt_field"]]).strip()
        completion = str(row[data["completion_field"]]).strip()
        if not prompt or not completion:
            continue
        if len(prompt) > data["max_prompt_chars"]:
            continue
        if len(completion) > data["max_completion_chars"]:
            continue
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        out.append(
            {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    return out


def main() -> None:
    config = recipe()
    data = config["data"]
    rows = records(config)
    held = int(data["eval_samples"])
    limit = int(data["max_samples"])
    out_dir = HERE / data["prepared"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("dev", rows[:held]), ("train", rows[held : held + limit])):
        path = out_dir / (name + ".jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\\n")
        print(str(path) + ": " + str(len(part)) + " records")


if __name__ == "__main__":
    main()
''',
    "train.py": '''\
"""Post-train the recipe's base model: SFT, then optional GRPO.

    python train.py

Each enabled stage saves a plain, fully merged model directory, so the next
stage and export.py both read an ordinary Hugging Face model folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
FINAL_ANSWER = re.compile(r"####\\s*(-?[\\d.,]+)")


def recipe() -> dict:
    return yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))


def tokenizer_for(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    template = config.get("chat_template")
    if template:
        tokenizer.chat_template = Path(template).read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def split(config: dict, name: str):
    path = HERE / config["data"]["prepared"] / (name + ".jsonl")
    return load_dataset("json", data_files=str(path), split="train")


def final_answer(text: str) -> str:
    """The answer the grader compares, or the last line when there is no marker."""
    found = FINAL_ANSWER.search(text)
    if found:
        return found.group(1).replace(",", "")
    return text.strip().split("\\n")[-1].strip()


def match_reward(completions, reference, **kwargs) -> list[float]:
    """1.0 for a sampled reply whose final answer equals the reference answer."""
    del kwargs
    scores = []
    for reply, target in zip(completions, reference):
        text = reply[0]["content"] if isinstance(reply, list) else reply
        scores.append(1.0 if final_answer(text) == final_answer(target) else 0.0)
    return scores


def save(trainer, tokenizer, merge: bool, out: str) -> str:
    model = trainer.model
    if merge:
        model = model.merge_and_unload()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def run_sft(config: dict, tokenizer) -> str:
    stage = config["sft"]
    out = str(HERE / stage["output_dir"])
    lora = stage["lora"]
    peft_config = None
    if lora["enabled"]:
        peft_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        )
    trainer = SFTTrainer(
        model=config["base_model"],
        args=SFTConfig(
            output_dir=out,
            num_train_epochs=stage["epochs"],
            learning_rate=float(stage["learning_rate"]),
            per_device_train_batch_size=stage["per_device_batch_size"],
            gradient_accumulation_steps=stage["gradient_accumulation_steps"],
            max_length=stage["max_length"],
            bf16=stage["bf16"],
            gradient_checkpointing=stage["gradient_checkpointing"],
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=split(config, "train"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    return save(trainer, tokenizer, peft_config is not None, out)


def run_grpo(config: dict, tokenizer, start: str) -> str:
    stage = config["grpo"]
    out = str(HERE / stage["output_dir"])
    # GRPO generates from the prompt, so the prompt must be padded and
    # truncated on the LEFT. trl 0.27.2 builds exactly this for itself when
    # processing_class is None and takes a supplied one as it comes, so a
    # tokenizer carried over from SFT would keep right padding here.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    rows = split(config, "train")
    rows = rows.select(range(min(int(stage["prompts"]), len(rows))))
    rows = rows.map(
        lambda row: {"reference": row["completion"][0]["content"]},
        remove_columns=["completion"],
    )
    trainer = GRPOTrainer(
        model=start,
        reward_funcs=match_reward,
        args=GRPOConfig(
            output_dir=out,
            max_steps=stage["steps"],
            learning_rate=float(stage["learning_rate"]),
            per_device_train_batch_size=stage["per_device_batch_size"],
            gradient_accumulation_steps=stage["gradient_accumulation_steps"],
            num_generations=stage["num_generations"],
            max_prompt_length=stage["max_prompt_length"],
            max_completion_length=stage["max_completion_length"],
            temperature=stage["temperature"],
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=rows,
        processing_class=tokenizer,
    )
    trainer.train()
    return save(trainer, tokenizer, False, out)


def main() -> None:
    config = recipe()
    tokenizer = tokenizer_for(config)
    trained = config["base_model"]
    if config["sft"]["enabled"]:
        trained = run_sft(config, tokenizer)
    if config["grpo"]["enabled"]:
        trained = run_grpo(config, tokenizer, trained)
    print("trained model: " + trained)


if __name__ == "__main__":
    main()
''',
    "eval.py": '''\
"""Development evaluation on the held-out slice prepare.py wrote.

    python eval.py [model_dir]

Advisory only. The framework scores the exported attempt with its own frozen
score program; this number never becomes the result of a step.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
FINAL_ANSWER = re.compile(r"####\\s*(-?[\\d.,]+)")


def final_answer(text: str) -> str:
    found = FINAL_ANSWER.search(text)
    if found:
        return found.group(1).replace(",", "")
    return text.strip().split("\\n")[-1].strip()


def main() -> None:
    config = yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))
    # The same rule export.py applies, so development evidence and the
    # submitted attempt are never two different checkpoints.
    stage = config["grpo"] if config["grpo"]["enabled"] else config["sft"]
    model_dir = sys.argv[1] if len(sys.argv) > 1 else str(HERE / stage["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto"
    )
    lines = (HERE / config["data"]["prepared"] / "dev.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows = [json.loads(line) for line in lines][: int(config["eval"]["samples"])]
    size = int(config["eval"]["batch_size"])
    correct = 0
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        texts = [
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )
            for row in chunk
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left"
        ).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=int(config["eval"]["max_new_tokens"]),
            do_sample=False,
        )
        width = encoded["input_ids"].shape[1]
        for row, output in zip(chunk, generated):
            reply = tokenizer.decode(output[width:], skip_special_tokens=True)
            target = row["completion"][0]["content"]
            correct += final_answer(reply) == final_answer(target)
        print("scored " + str(start + len(chunk)) + "/" + str(len(rows)))
    print("dev accuracy: " + str(correct / max(1, len(rows))))


if __name__ == "__main__":
    main()
''',
    "export.py": '''\
"""Copy the trained model into the attempt directory the framework scores.

    python export.py <attempt_dir>

The exported folder is the whole submission: the framework opens it with the
task's own score program after the session ends.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer

HERE = Path(__file__).resolve().parent


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python export.py <attempt_dir>")
    config = yaml.safe_load((HERE / "recipe.yaml").read_text(encoding="utf-8"))
    stage = config["grpo"] if config["grpo"]["enabled"] else config["sft"]
    source = HERE / stage["output_dir"]
    if not (source / "config.json").is_file():
        raise SystemExit("export.py: no saved model in " + str(source))
    target = Path(sys.argv[1]).resolve() / config["export"]["dirname"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    AutoConfig.from_pretrained(target)
    AutoTokenizer.from_pretrained(target)
    print("exported " + str(target))


if __name__ == "__main__":
    main()
''',
}
