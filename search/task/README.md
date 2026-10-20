# Train a speculative-decoding draft head that a real serving stack can use

Make SGLang generate tokens faster from a fixed target model, by giving it a better
EAGLE draft head. The score is measured **inside SGLang**: tokens/second with your head
divided by tokens/second without it, same engine, same prompts, same settings.

```
score = SGLang tok/s  (target + your draft head)
        ----------------------------------------
        SGLang tok/s  (target alone)
```

Both arms run the same SGLang build with identical engine settings on the same 30
prompts, batch 1, greedy, 128 new tokens. Only the `speculative_*` arguments differ.

## Why the score is a framework ratio, and not a speedup over a plain loop

An earlier version of this task scored speedup over a HuggingFace `generate()` loop.
That loop is launch-bound at batch 1, so it rewarded work that a real serving stack
throws away, and it did so convincingly: the best submission reached 4.33x there while
being **slower than stock SGLang**. Worse, two high-scoring heads from that task cannot
be loaded by any serving framework at all -- one had an extra residual MLP and a final
norm, another had no attention whatsoever. They were excellent answers to the wrong
question.

So the question changed. Beat SGLang using SGLang's own machinery.

## What you submit

A trained draft head, not an inference loop. SGLang owns decoding.

```
<attempt>/full/
  head.pt            state dict, SGLang key names (see reference/eagle_head.py)
  head_config.json   geometry you chose, plus optional sglang_overrides
```

`head_config.json` may carry `"sglang_overrides"` with any of
`speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`.
Those three are yours to tune. Everything else about the engine is fixed by
[`models.json`](models.json) so arms stay comparable.

### The scoring path contract -- read this before writing a score program

`score(output_dir)` receives **the `full/` directory itself**: the directory that
contains `head.pt` and `head_config.json` *directly*. **Do not append `full/` to it.**

The layout above is drawn from the attempt's point of view, which is why it shows the
`full/` component. The scorer is handed the leaf, not the parent. From a real run's
framework record:

```
"output_dir": ".../worker_N/state/artifacts/full"
```

So:

```python
def score(output_dir: str) -> dict:
    head = Path(output_dir) / "head.pt"              # correct
    # head = Path(output_dir) / "full" / "head.pt"   # WRONG -- never exists
```

This has already cost one machine a complete run. A scorer that looks for
`<output_dir>/full/head.pt` finds nothing and returns 0.0 for **every** step, while
setup verification still passes -- a verifier that synthesises its own nested fixture
takes the wrong branch successfully. The symptom is green setup and ten zero scores,
with no error anywhere. If you must be tolerant, resolve both and prefer the leaf:

```python
sub = out / "full" if (out / "full" / "head.pt").is_file() else out
```

## The architecture is FIXED. This is the constraint, not a suggestion.

SGLang hosts the draft as the EAGLE class [`models.json`](models.json) names, a fixed module tree. A head of a
different shape cannot be loaded however well it scores offline, and
`harness/make_draft.py` will reject it by name.

[`reference/eagle_head.py`](reference/eagle_head.py) **is** the architecture and a
working implementation of it. Train that module. In short:

```
fused = fc(cat([token_embedding, target_feature]))   # EMBEDDING FIRST
h = fused
for layer in layers:
    h = h + attn(input_layernorm(h))                 # qkv bias, RoPE, GQA allowed
    h = h + mlp(post_attention_layernorm(h))
return h                                             # NO final norm
```

Free to choose: number of layers, MLP width, kv-head count, the training corpus, the
loss, the schedule, and the three speculative tree parameters.

Not available: a final norm, extra residual MLP blocks, attention-free blocks, a
different concat order, or your own output projection. Candidate logits come from the
target's **frozen** `lm_head`, which the harness supplies.

## Exactness

**You cannot break it, and you do not have to prove it.** Under greedy decoding
SGLang's verify kernel accepts a draft token only when it equals the target's own token,
and stores the target's token either way, so every emitted token is the target's argmax.
Exactness is a property of the verify/accept logic, and that logic is SGLang's, not
yours. A proposer only changes *speed*.

What the harness enforces instead is that the measurement is real:

* the head is the required architecture and packages cleanly
* SGLang loads the draft with **no fallback** (any fallback warning zeroes the gate)
* speculation is **actually firing** -- the engine must report a mean acceptance length
  above 1.05, because a silently disabled speculator scores ~1.0 and looks like an
  honest null result

Token streams between arms are reported, not required to match: in bf16 a near-tie can
round differently between a cached single-token decode and a batched tree verify. That
is the documented tie-flip, not an exactness failure.

## Harness

| file | what it does |
|---|---|
| `harness/benchmark.py` | **the score.** Runs both arms, each in its own process, returns the ratio |
| `harness/make_draft.py` | packages `head.pt` into an SGLang draft dir; rejects wrong architectures |
| `harness/check_draft.py` | cheap pre-flight: patch present, head loads, speculation active |
| `harness/_engine_arm.py` | one SGLang arm, one process (two 14B engines cannot coexist) |
| `harness/gate_eager_reference.py` | the old eager fp32 gate, kept for debugging a head in isolation |

```
python harness/check_draft.py --submission <attempt>/full     # minutes
python harness/benchmark.py  --submission <attempt>/full      # authoritative, ~40 min
```

## The environment carries one patch, and the gate checks for it

Upstream SGLang deletes layer 0's `input_layernorm` for parity with the reference EAGLE
code. The required architecture uses it. Without the patch a loaded head computes a
different function -- measured relative error **1.84** unpatched versus **7.7e-17**
patched. The run's environment ships patched; `check_draft.py` asserts it.

## Measurement notes that cost earlier runs real time

* **Two engines cannot be resident at once**, so the arms cannot be interleaved. Both are
  measured back to back in one scoring call and each reports its own sweep spread.
* **The first SGLang start JIT-compiles kernels.** That inflated a baseline by 13% once
  and nearly produced a false null. The warmup pass absorbs it; do not compare a cold
  arm against a warm one.
* **Check what the framework silently turned off.** A vLLM comparison was void twice --
  once from a quiet model-runner fallback, once from async scheduling being disabled in
  only one arm. Both showed up only as WARNING lines.
* **The target is fixed** -- whichever model [`models.json`](models.json) names. Read it
  for the id, dtype and geometry; do not hardcode a model name anywhere.
