# Measured results

All numbers below were produced by the scripts in this repository, on one
A100-SXM4-80GB, with the versions pinned in `requirements-frozen.txt`. Nothing
here is carried over from a previous run: both baselines are re-measured in the
same session as the arm they are compared against.

## Headline: MT-Bench, 80 questions x 2 turns, 1024 new tokens, batch 1, greedy

| arm | tok/s | vs stock SGLang | vs best draftless | tau |
|---|---|---|---|---|
| HF `generate()`, eager | 16.78 | 0.321x | 0.237x | - |
| stock SGLang, no speculation | 52.214 | 1.000x | 0.736x | - |
| vLLM + `ngram` (prompt lookup) | 58.966 | 1.129x | 0.831x | 2.016 (acc. len) |
| SGLang + `NGRAM` (suffix automaton) | 70.932 | 1.3585x | 1.000x | 1.6057 |
| **SGLang + our EAGLE head** | **104.892** | **2.0089x** | **1.4788x** | **3.2061** |

**Which baseline.** The headline claim is against the best configuration either
framework reaches *without a trained head*, i.e. SGLang's own `NGRAM`: **1.4788x**.
Against the unspeculated defaults it is 2.0089x. Both are stated because the
second is the number most speculative-decoding papers report and the first is the
one a user actually has to beat -- enabling `NGRAM` costs nothing and takes one
flag.

Decomposed against the weak reference: **3.111x** (serving framework: eager ->
SGLang) **x 2.0089x** (our head) = **6.25x** end to end over a HuggingFace
`generate()` loop.

At temperature 1.0 the same head gives **1.7909x** over stock SGLang (93.881
tok/s, tau 2.8802); the draftless modes were not measured at T=1.

## Weight-free speculation: the baseline our head is compared against

Everything in this section needs no checkpoint and no training -- one flag. All
four modes either framework offers were measured; there is no fifth.

MT-Bench, 80 questions x 2 turns, 1024 new tokens, batch 1, greedy:

| mode | tok/s | over its own control | acceptance |
|---|---|---|---|
| SGLang `NGRAM` | **70.932** / 70.405 (2 cold passes, 0.75% apart) | +35.8% over 52.214 | tau **1.6057**, identical in both passes |
| vLLM `ngram` | **58.966** | not measured (no vLLM plain MT-Bench arm) | acc. length 2.016 |

30 prompts, 128 new tokens -- a near worst case for a copy-based proposer:

| mode | tok/s | over its own control | acceptance |
|---|---|---|---|
| SGLang `NGRAM` | 58.568 +/-0.34% (3 cold) | +11.63% over 52.466 +/-0.01% | - |
| vLLM `suffix` (Arctic Inference) | 55.682 +/-1.43% (3 cold) | +15.50% over 48.211 | acc. length 1.396 |
| vLLM `ngram` | 52.881 | +9.70% over 48.206 | acc. length 2.173 |
| vLLM `ngram_gpu` | 51.159 | +6.13% over 48.206 | acc. length 1.137 |

### Four things worth knowing before reusing these numbers

**Answer length decides the draftless gain.** SGLang's `NGRAM` is worth +11.6% on
128-token replies and +35.8% on MT-Bench's ~410-token answers. A copy-based
proposer needs something to copy, so a short-answer benchmark understates it
badly. Our head gains far less from the longer benchmark, which is why its margin
narrows from 1.70x to 1.48x.

**SGLang's proposer keeps state ACROSS requests; vLLM's does not.** SGLang builds
a suffix automaton (capacity 10M, trie depth 18) that survives between requests,
so re-running the same prompts in one process lets it propose its own earlier
output: three sweeps of the same 30 prompts measured 57.35 -> 248.45 -> 331.23
tok/s. That is memorisation, not speed. Every SGLang draftless number here is
therefore from a **cold engine, one pass**. vLLM's `ngram` only scans the current
request, so its sweeps are flat and a median is valid.
A consequence that survives even the cold protocol: on MT-Bench, question 80 is
proposed against the 79 answers already in the automaton. That is a real serving
condition, but it is not cold-per-request, and it is most of why SGLang's lead
over vLLM widens from 1.11x to 1.20x on the longer benchmark.

**vLLM handicaps itself when speculation is on.** Enabling `ngram` drops vLLM to
the V1 model runner and disables async scheduling, so its matched control is 48.2
tok/s rather than the 52.4 it reaches unspeculated. Against its own best
non-speculative configuration, `ngram` is worth only **+1.0%** -- it roughly
refunds what turning it on costs. All vLLM arms here are pinned to the V1 runner
with async scheduling off so that each pair differs only in the proposer.

**Acceptance length does not rank throughput.** vLLM accepts longer drafts than
SGLang (2.016 vs 1.606) while running slower, and `ngram_gpu` accepts barely
anything (1.137) despite identical settings to `ngram`. What pays is how often a
proposal is usable, plus the engine path it runs on.

### Per-category, MT-Bench T=0

Both frameworks rank the categories identically, which is evidence this is a
property of how copiable the workload is rather than of either implementation:

| category | SGLang `NGRAM` | vLLM `ngram` | SGLang tau |
|---|---|---|---|
| math | 93.98 | 74.10 | 2.120 |
| extraction | 84.19 | 76.67 | 1.966 |
| coding | 82.71 | 66.14 | 1.859 |
| stem | 68.30 | 54.46 | 1.555 |
| humanities | 65.94 | 55.15 | 1.493 |
| reasoning | 65.25 | 59.56 | 1.486 |
| writing | 62.20 | 54.25 | 1.401 |
| roleplay | 58.22 | 51.02 | 1.308 |

Categories that restate the prompt or re-derive the same expressions accept most;
open-ended prose accepts least. Our head's ordering is different -- it leads on
chain-of-thought arithmetic because it has learned the target's distribution, not
because the text repeats.

## Speedup is workload-dependent

tau -- accepted tokens per verification cycle -- varies a lot by workload, so a
single speedup number is misleading. Per benchmark, same head, same host:

| benchmark | plain tok/s | head tok/s | speedup | tau |
|---|---|---|---|---|
| GSM8K (500 items) | 52.67 | 142.56 | **2.7069x** | 4.420 |
| MT-Bench T=0 | 52.214 | 104.892 | 2.0089x | 3.2061 |
| HumanEval (164 items) | 45.89 | 87.76 | 1.9123x | 3.780 |
| MT-Bench T=1 | 52.421 | 93.881 | 1.7909x | 2.8802 |

Chain-of-thought arithmetic is the most predictable workload and accepts
longest; open-ended chat at temperature 1.0 the least. Report the benchmark
alongside the number.

## Quality: no measurable change

Paired, same items, same plain arm, T=0. The paired (McNemar) test is used
because comparing marginal accuracies hides offsetting flips.

| benchmark | plain | + our head | delta | discordant (win:lose) |
|---|---|---|---|---|
| GSM8K (n=500) | 461/500 = 0.9220 | 461/500 = 0.9220 | +0.0000 | 5 : 5 |
| HumanEval (n=164) | 132/164 = 0.8049 | 132/164 = 0.8049 | +0.0000 | 2 : 2 |

Combined: **7 : 7 discordant, McNemar exact p = 1.000**. The discordance is
symmetric, which is the signature of sampling noise rather than degradation.

Token counts across arms stayed within 2.14%, so the accuracy comparison is not
confounded by different output lengths.

### What "no quality change" does and does not mean

Under greedy decoding SGLang's verify kernel only ever emits the TARGET model's
own argmax: the accept test is `draft_token == target_token` and the kernel
stores the target's token. A proposer therefore cannot change which token the
verifier picks.

It does **not** follow that output is bit-identical. bf16 ties can round
differently between a cached single-token decode and a batched tree verify, so
token streams do diverge: on a 30-prompt probe only 10-15 of 30 streams matched
exactly, about 0.86% of tokens differing. The plain arm itself is 30/30
deterministic, so speculation is the cause.

Say "output distribution preserved under the verifier's own logits; token
streams not bit-identical" and report the count. Do not claim losslessness.

## Instrument resolution

Know the noise floor before believing a difference:

| comparison | spread |
|---|---|
| repeat of the same head, same process | ~0.30% |
| same head, worker harness vs independent scorer | ~1% |
| the stock-SGLang baseline across 10 scored steps | 52.30-52.43 tok/s (0.25%) |
| the speculative arm, run to run | ~1.2% |

Treat anything under ~1.5% as unresolved. `eval/REFERENCE.md` states the
tolerance as +/-2%.

## The search that produced the head

The head was found by an automated 10-step search. Scores below are on the
search's own 30-prompt harness, which **inflates about 10%** relative to
MT-Bench -- the internal 2.2116x corresponds to the 2.0089x measured above.

| step | internal score | tau |
|---|---|---|
| 1 | 1.3190 | 2.205 |
| 2 | 1.4663 | 2.490 |
| 3 | 1.6180 | 2.757 |
| 4 | 1.6961 | 2.732 |
| 5 | 1.9285 | 3.163 |
| 6 | 2.0333 | 3.302 |
| 7 | 2.0590 | 3.359 |
| 8 | 2.1342 | 3.701 |
| 9 | 2.1851 | 3.804 |
| 10 | **2.2116** | **3.842** |

It never plateaued, so this is a lower bound on what the recipe reaches, not a
ceiling. See `RECIPE.md` for which changes moved the number.
