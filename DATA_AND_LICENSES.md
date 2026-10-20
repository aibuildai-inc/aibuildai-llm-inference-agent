# Licensing

## This repository's own code and documentation: MIT

See `LICENSE`. That covers everything we wrote: `task/`, `eval/*.py`,
`eval/run_eval.sh`, `tools/`, `docs/`, and the Markdown.

It does **not** cover the three vendored datasets or the model weights. Those are
addressed below.

# Third-party content bundled here

This repository vendors three evaluation datasets for reproducibility. They are
**not ours** and carry their own terms. Check them before redistributing.

| file | dataset | upstream | note |
|---|---|---|---|
| `eval/mt_bench_question.jsonl` | MT-Bench questions (80) | LMSYS FastChat | Apache-2.0 at source; verify current terms |
| `eval/gsm8k_test.jsonl` | GSM8K test split (1319 rows; we score the first 500) | OpenAI GSM8K | MIT at source |
| `eval/humaneval.jsonl` | HumanEval (164 problems) | OpenAI HumanEval | MIT at source |

The target model `Qwen/Qwen2.5-14B-Instruct` is **not** redistributed here; it is
downloaded from HuggingFace under its own licence.

The draft head weights are derived from self-distilled outputs of that target
model. Whether that makes them a derivative work under the target's licence is a
question for whoever publishes them -- it has not been resolved here.

## A safety note on HumanEval

`eval/bench_quality.py` executes model-generated Python to score HumanEval. It
sandboxes each candidate in a separate process with a throwaway temp directory,
`RLIMIT_CPU` 10 s, `RLIMIT_AS` 2 GiB, `RLIMIT_NPROC` 64 and a 15 s wall timeout.
That is adequate for grading a model you trained; it is **not** a security
boundary for untrusted code. Do not point it at arbitrary generations.
