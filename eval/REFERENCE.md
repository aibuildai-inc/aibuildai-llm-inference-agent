# Reference numbers — diff your run against these

Produced by `run_eval.sh` in this repository, on the head in `heads/`
(`sha256 7a744703b50f...`, inter 13824, tree 4/24/64).

**Host**: 1x NVIDIA A100-SXM4-80GB. **Versions**: torch 2.13.0+cu130,
transformers 5.12.1, sglang 0.5.21 (patched), flashinfer 0.6.18,
accelerate 1.15.0.

**Tolerance: +/-2%.** The speculative arm resolves to about +/-1.2% run to run,
so anything inside 2% is a match, not slack. A difference you cannot reproduce
twice is not a difference.

## MT-Bench — 80 questions x 2 turns, 1024 new tokens, batch 1

| arm | T=0 tok/s | T=1 tok/s |
|---|---|---|
| stock SGLang (plain) | 52.214 | 52.421 |
| SGLang + head (draft) | 104.892 | 93.881 |
| **speedup** | **2.0089x** | **1.7909x** |
| tau | 3.2061 | 2.8802 |

HF `generate()` eager on the same protocol: **16.783 tok/s** (T=0).
End to end over eager: **6.25x** = 3.111x framework x 2.0089x head.

Token counts across the three T=0 arms stayed within **2.14%**, so no arm was
given an easier generation length.

## Quality — paired, T=0

| benchmark | n | plain | draft | discordant (win:lose) | McNemar p |
|---|---|---|---|---|---|
| GSM8K | 500 | 461 (0.9220) | 461 (0.9220) | 5 : 5 | 1.000 |
| HumanEval | 164 | 132 (0.8049) | 132 (0.8049) | 2 : 2 | 1.000 |

Combined **7 : 7, p = 1.000**. Speed on those workloads: GSM8K **2.7069x**
(tau 4.420), HumanEval **1.9123x** (tau 3.780).

## If your numbers differ

Check these in order — each has produced a wrong number on this project:

1. **Is SGLang patched?** `run_eval.sh --check` verifies it. Unpatched, the head
   computes a different function (relative error 1.84 vs 7.7e-17).
2. **Is the JIT cache warm?** A cold Triton/flashinfer cache inflated one
   baseline by 13% and turned a 1.975x into 2.0994x. Run the phase twice and use
   the second.
3. **Did you re-measure the baseline in the same session?** A carried-over
   baseline is the single easiest way to get a wrong ratio. Both arms here are
   always measured together.
4. **One sweep, or three?** `statistics.median` of two values is their mean, so
   two sweeps reject no outlier. Throughput noise is one-sided — a stall can only
   slow a sweep — so prefer max over median.
5. **Different GPU?** tau is hardware-independent but per-cycle cost is not, so
   the speedup will move while tau should not. If tau differs by more than ~1%,
   suspect the patch or the head, not the GPU.

A prior head measured on this same harness gave 1.6637x / tau 2.6579 at T=0;
those numbers are preserved in `REFERENCE.md.prior-head` for anyone comparing
against the earlier release.
