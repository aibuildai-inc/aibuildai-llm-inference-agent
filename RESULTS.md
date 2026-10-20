# Measured results

All numbers below were produced by the scripts in this repository, on one
A100-SXM4-80GB, with the versions pinned in `requirements-frozen.txt`. Nothing
here is carried over from a previous run: both baselines are re-measured in the
same session as the arm they are compared against.

## Headline: MT-Bench, 80 questions x 2 turns, 1024 new tokens, batch 1, greedy

| arm | tok/s | vs stock SGLang | tau |
|---|---|---|---|
| HF `generate()`, eager | 16.78 | 0.321x | - |
| stock SGLang, no speculation | 52.214 | 1.000x | - |
| **SGLang + our EAGLE head** | **104.892** | **2.0089x** | **3.2061** |

Decomposed: **3.111x** (serving framework: eager -> SGLang) **x 2.0089x**
(our head) = **6.25x** end to end over a HuggingFace `generate()` loop.

At temperature 1.0 the same head gives **1.7909x** (93.881 tok/s, tau 2.8802).

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
