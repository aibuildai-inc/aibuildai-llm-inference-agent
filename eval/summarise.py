"""Summarise an evaluation run and compare it against the reference numbers.

  python summarise.py --results DIR [--reference REFERENCE.md]

Prints whatever arms are present, so it works on a partial run.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

# These MUST match REFERENCE.md, which is the document of record. They were
# stale once -- carrying a previous head's figures while REFERENCE.md had been
# updated -- and the consequence is the worst a verification tool can have: a
# correct reproduction was reported as "+19.87% OUTSIDE +-2%". check_reference()
# below now refuses to let the two drift apart silently.
REF = {  # measured on A100-80GB, sglang 0.5.21 -- see REFERENCE.md
    "mtbench_t0_speedup": 2.0089, "mtbench_t0_tau": 3.2061,
    "mtbench_t1_speedup": 1.7909, "mtbench_t1_tau": 2.8802,
    "mtbench_t0_plain": 52.214, "mtbench_t0_draft": 104.892, "mtbench_t0_hfeager": 16.783,
    "gsm8k_plain": 0.9220, "gsm8k_draft": 0.9220,
    "humaneval_plain": 0.8049, "humaneval_draft": 0.8049,
}


def check_reference(path) -> None:
    """Warn loudly if REFERENCE.md disagrees with REF above.

    --reference was accepted and then ignored, which is how the two came apart.
    Parsing is deliberately crude: it only looks for the speedup row, because
    that is the number a reproducer is judged against.
    """
    import re
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return
    m = re.search(r"\|\s*\*\*speedup\*\*\s*\|\s*\*\*([0-9.]+)x\*\*\s*\|\s*\*\*([0-9.]+)x\*\*",
                  text)
    if not m:
        print(f"  note: could not find the speedup row in {path}; not cross-checked")
        return
    doc = {"mtbench_t0_speedup": float(m.group(1)), "mtbench_t1_speedup": float(m.group(2))}
    bad = [(k, REF[k], v) for k, v in doc.items() if abs(REF[k] - v) > 1e-4]
    if bad:
        print("  *** REFERENCE MISMATCH -- this script's built-in numbers disagree with")
        print(f"  *** {path}. Trust the document, not this script, and fix REF:")
        for k, mine, theirs in bad:
            print(f"  ***   {k}: script {mine}  document {theirs}")
TOL = 2.0  # percent; the method arm resolves to about +-1.2%


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    if a.reference:
        check_reference(a.reference)
    R = Path(a.results)

    def load(n):
        p = R / f"{n}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def verdict(got, ref):
        d = 100 * (got - ref) / ref
        return f"{d:+6.2f}%  {'MATCH' if abs(d) <= TOL else 'OUTSIDE ±%.0f%%' % TOL}"

    print("\n--- MT-Bench speedup ---")
    for t in ("t0", "t1"):
        pl, dr = load(f"mtbench_{t}_plain"), load(f"mtbench_{t}_draft")
        if not (pl and dr):
            print(f"  {t}: not run"); continue
        s = dr["tokens_per_second"] / pl["tokens_per_second"]
        print(f"  T={dr['temperature']:.0f}  plain {pl['tokens_per_second']:7.3f}  "
              f"draft {dr['tokens_per_second']:7.3f}  tau {dr['tau']:.4f}")
        print(f"        speedup {s:.4f}x   reference {REF[f'mtbench_{t}_speedup']:.4f}x   "
              f"{verdict(s, REF[f'mtbench_{t}_speedup'])}")
        print(f"        per category: " + "  ".join(
            f"{c[:4]} {dr['per_category'][c]['tokens_per_second']/pl['per_category'][c]['tokens_per_second']:.2f}"
            for c in sorted(dr["per_category"])))

    hf, pl, dr = load("mtbench_t0_hfeager"), load("mtbench_t0_plain"), load("mtbench_t0_draft")
    if hf:
        print("\n--- HF eager reference, same protocol ---")
        print(f"  hf-eager {hf['tokens_per_second']:7.3f} tok/s   "
              f"reference {REF['mtbench_t0_hfeager']:.3f}   "
              f"{verdict(hf['tokens_per_second'], REF['mtbench_t0_hfeager'])}")
        if pl and dr:
            fw = pl["tokens_per_second"] / hf["tokens_per_second"]
            hd = dr["tokens_per_second"] / pl["tokens_per_second"]
            print(f"  decomposition: framework {fw:.4f}x  x  head {hd:.4f}x  =  "
                  f"{fw*hd:.4f}x over eager   (reference 3.111 x 2.0089 = 6.25)")
        tc = [d["sweeps"][0]["tokens"] for d in (hf, pl, dr) if d]
        print(f"  token-count fairness: {[int(x) for x in tc]}  "
              f"spread {100*(max(tc)-min(tc))/min(tc):.2f}%  "
              f"{'OK' if (max(tc)-min(tc))/min(tc) < 0.1 else '*** arms generated different amounts ***'}")

    print("\n--- quality parity (paired) ---")
    from math import comb
    tot01 = tot10 = 0
    for b in ("gsm8k", "humaneval"):
        p_, d_ = load(f"{b}_plain"), load(f"{b}_draft")
        if not (p_ and d_):
            print(f"  {b}: not run"); continue
        P = {x["i"]: x["ok"] for x in p_["details"]}
        Q = {x["i"]: x["ok"] for x in d_["details"]}
        both = sorted(set(P) & set(Q))
        b01 = sum(1 for i in both if not P[i] and Q[i])
        b10 = sum(1 for i in both if P[i] and not Q[i])
        tot01 += b01; tot10 += b10
        n = b01 + b10
        p = 1.0 if n == 0 else min(1.0, 2*sum(comb(n, j) for j in range(min(b01, b10)+1))/2**n)
        print(f"  {b:10s} plain {p_['score']:.4f}  draft {d_['score']:.4f}  "
              f"diff {d_['score']-p_['score']:+.4f}  discordant {b01}:{b10}  p={p:.3f}")
        print(f"             reference: plain {REF[f'{b}_plain']:.4f}  draft {REF[f'{b}_draft']:.4f}")
    n = tot01 + tot10
    if n:
        p = min(1.0, 2*sum(comb(n, j) for j in range(min(tot01, tot10)+1))/2**n)
        print(f"  COMBINED discordant {tot01}:{tot10}  McNemar p={p:.3f}  "
              f"{'no significant difference' if p > 0.05 else 'SIGNIFICANT'}")
        print("  (reference was 7:7 combined over 664 items, McNemar exact p=1.000 --")
        print("   so a sub-1.5% degradation is not excluded by this sample size)")

    print("\nTolerance is ±2%: the speculative arm resolves to about ±1.2%, so anything")
    print("inside that is a match, not slack. See REFERENCE.md for the full context.")


if __name__ == "__main__":
    main()
