# Training a head

This is the working code that produced the published head, with absolute paths
removed. It is **not** a polished library: it grew across the ten steps of an
automated search, and several files are step-specific increments rather than a
general pipeline. It is included so the result can be reproduced and extended,
not as an example of clean design.

`recipe.yaml` drives everything. `prepare.py`, `train.py`, `eval.py` and
`export.py` all read it from this directory.

## The flow

```bash
cd train
export HF_HOME="$HOME/hf-cache"       # the target is ~28 GB
# optional: export SPECDEC_HARNESS="$PWD/../task/harness"   (this is the default)

# 1. harvest teacher features: teacher-force the target over a 4-domain mix,
#    saving per-token post-final-norm hidden states + token ids
python prepare.py                      # -> data/train.pt, data/dev.pt

# 2. train the head by multi-step (unroll) feature regression
python train.py                        # -> runs/<output_dir>/head.pt

# 3. advisory dev check (never the reported score)
python eval.py                         # single-step top-1 + chain accept length

# 4. package for evaluation
python export.py ../out                # -> ../out/full/{head.pt, head_config.json}
```

Then evaluate with the repo's harness — that is the number worth reporting:

```bash
cd ..
SPECDEC_HEAD="$PWD/out/full/head.pt" ./eval/run_eval.sh --phase mtbench
```

`export.py` writes `<dest>/full/`, the layout the search's scorer expected.
`run_eval.sh` reads `${HEAD%.pt}.config.json`, so when pointing it at an exported
head either rename the pair or pass `--head` explicitly.

## The config that produced the published head

From `recipe.yaml`:

```
head.inter                              13824     (~328M params, 1 layer)
train.epochs                            4
train.lr                                3.0e-4
data.train_file                         train_v4.pt
sglang_overrides.speculative_num_steps  4
sglang_overrides.speculative_eagle_topk 24
sglang_overrides.speculative_num_draft_tokens 64
```

That head scores **2.0089x on MT-Bench T=0** (see `../RESULTS.md`).

## corpus/ — read this before using it

The published corpus was **not** produced by one `prepare.py` run. It was grown
incrementally across search steps, because regenerating 6M tokens of teacher
features costs hours:

| file | what it added |
|---|---|
| `gen_chat_aug.py` | a second, disjoint chat slice (alpaca-cleaned, empty-input rows) when dev showed chat was the weakest domain |
| `gen_rc_aug.py`, `gen_rc_aug2.py` | additional disjoint reasoning + code slices to un-starve those domains |
| `merge_corpus.py`, `_v3`, `_v4` | concatenate each increment onto the previous corpus (`train.pt` -> `train_v2.pt` -> ... -> `train_v4.pt`) |

The `_v3`/`_v4` suffixes are the search's own versioning, kept so the lineage of
`train_v4.pt` is legible. Each augmenter reuses `prepare.py`'s
`build_prompt` / `generate_continuations` / `collect`, so the features are
produced identically to the base corpus.

**A single `prepare.py` run with the mix in `recipe.yaml` builds an equivalent
corpus** and is the simpler path if you are starting fresh. Use the incremental
scripts only if you are extending an existing corpus and do not want to pay for
regeneration.

## What the recipe says matters

In priority order, with the measured evidence in `../RECIPE.md`:

1. **Corpus size.** 2.37M -> 4.37M tokens moved a fixed geometry from 1.6876 to
   1.8461. `inter 13824` needs >= ~5M tokens; below ~3M use `inter 8192`.
2. **Composition.** Weight toward the weakest domain (chat), which is what the
   `corpus/` increments were for.
3. **Epochs**, to a point: 2 -> 4 epochs was worth a further +5.5%.
4. **Multi-step unroll (deep supervision)** — the largest late gain, tau 3.163
   -> 3.701.
5. **The tree**, re-tuned after the head is trained: `4/24/64` beat `4/16/64` by
   5.7% on MT-Bench *with slightly lower tau*, i.e. a per-cycle-cost win.
6. **Width is a null.** `inter` 13824 -> 16384 changed nothing twice.
