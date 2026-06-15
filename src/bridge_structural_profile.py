"""
Structural profile of operational bridge theorems vs typical theorems.

Motivation (extends a-0043): the detection half of the paper flags 280 operational
bridge theorems (proofs whose premises span >=2 distinct PURE clusters,
results/bridge_theorems.json), and a-0043 showed they are a tiny (~0.2-0.3%),
near-disjoint population from the generic cross-domain theorems the q-0011
retrieval chain measures. But the paper never asks WHAT KIND of theorem a bridge
is, structurally. Are operational bridges "deep syntheses" (long proofs, many
premises, sitting far downstream from the axioms, rarely themselves reused) or
"foundational connectors" (short, shallow, heavily reused)?

This is a cheap CPU-only / pure-stdlib census over existing data:

  data/theorem_tactics.json  -> proof length (# tactic strings) per theorem
  data/tactic_steps.jsonl    -> premise edges A->p (A's proof uses p)
  results/bridge_theorems.json (280) and bridge_theorems_filtered.json (34)

Per theorem we measure four structural features:
  proof_len      number of tactic steps (proof length proxy)
  n_premises     number of DISTINCT premises the proof uses (out-degree)
  reuse_indeg    number of DISTINCT theorems that use this theorem as a premise
                 (how foundational / reused the theorem itself is)
  dag_depth      longest premise-dependency chain below it (topological depth
                 from the leaves; deeper = more downstream synthesis)

We compare the 280 bridges (and the strict 34) against the full population of
theorems-with-tactics, reporting medians and a Mann-Whitney-style rank fraction
(P(bridge > random)) for each feature.
"""

import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"


def load():
    tactics = json.load(open(DATA / "theorem_tactics.json"))
    proof_len = {name: len(steps) for name, steps in tactics.items()}

    # premise edges: theorem -> set(premises); and reuse in-degree
    out_prem = defaultdict(set)
    with open(DATA / "tactic_steps.jsonl") as f:
        for line in f:
            o = json.loads(line)
            name = o["full_name"]
            for p in o["premises"]:
                if p != name:
                    out_prem[name].add(p)

    reuse_indeg = defaultdict(int)
    for name, prems in out_prem.items():
        for p in prems:
            reuse_indeg[p] += 1

    bridges = [b["theorem"] for b in json.load(open(RESULTS / "bridge_theorems.json"))]
    strict = [b["theorem"] for b in json.load(open(RESULTS / "bridge_theorems_filtered.json"))]
    return proof_len, out_prem, reuse_indeg, bridges, strict


def dag_depth(out_prem):
    """Longest premise-chain length below each node (DAG longest path from leaves).

    Iterative memoized DFS; premise edges A->p mean p is below A. Cycles (should
    not occur in a proof dependency graph) are broken by marking in-progress
    nodes depth 0.
    """
    depth = {}
    WHITE, GRAY = 0, 1
    state = {}
    for root in list(out_prem.keys()):
        if root in depth:
            continue
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                d = 0
                for p in out_prem.get(node, ()):  # leaves -> depth 0
                    d = max(d, depth.get(p, 0) + 1)
                depth[node] = d
                state[node] = WHITE
                continue
            if node in depth:
                continue
            if state.get(node) == GRAY:  # cycle guard
                depth[node] = 0
                continue
            state[node] = GRAY
            stack.append((node, True))
            for p in out_prem.get(node, ()):
                if p not in depth and state.get(p) != GRAY:
                    stack.append((p, False))
    return depth


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def rank_fraction(group_vals, pop_sorted):
    """Approx P(group value > random population value) via binary search.

    pop_sorted is the sorted full-population list. For each group value v,
    fraction of population strictly below v; averaged over the group. 0.5 = no
    difference, >0.5 = group skews high."""
    import bisect

    n = len(pop_sorted)
    if n == 0 or not group_vals:
        return 0.0
    tot = 0.0
    for v in group_vals:
        lo = bisect.bisect_left(pop_sorted, v)
        hi = bisect.bisect_right(pop_sorted, v)
        tot += (lo + hi) / 2 / n  # mid-rank handles ties
    return tot / len(group_vals)


def feature_report(name, getter, all_names, bridges, strict):
    pop = [getter(t) for t in all_names]
    pop_sorted = sorted(pop)
    bvals = [getter(t) for t in bridges if t in set(all_names_set)]
    svals = [getter(t) for t in strict if t in set(all_names_set)]
    return {
        "feature": name,
        "pop_median": median(pop),
        "pop_mean": round(sum(pop) / len(pop), 3),
        "bridge_median": median(bvals),
        "bridge_mean": round(sum(bvals) / len(bvals), 3) if bvals else None,
        "strict_median": median(svals),
        "strict_mean": round(sum(svals) / len(svals), 3) if svals else None,
        "bridge_rankfrac": round(rank_fraction(bvals, pop_sorted), 3),
        "strict_rankfrac": round(rank_fraction(svals, pop_sorted), 3),
        "n_bridge_scored": len(bvals),
        "n_strict_scored": len(svals),
    }


if __name__ == "__main__":
    proof_len, out_prem, reuse_indeg, bridges, strict = load()

    # Population = theorems that have a recorded proof (tactics) AND premises,
    # so all four features are defined. Use theorem_tactics keys as the spine.
    all_names = [t for t in proof_len if t in out_prem]
    all_names_set = set(all_names)

    print(f"population (theorems w/ tactics+premises): {len(all_names)}")
    print(f"bridges total {len(bridges)}, strict {len(strict)}")

    depth = dag_depth(out_prem)

    getters = {
        "proof_len": lambda t: proof_len.get(t, 0),
        "n_premises": lambda t: len(out_prem.get(t, ())),
        "reuse_indeg": lambda t: reuse_indeg.get(t, 0),
        "dag_depth": lambda t: depth.get(t, 0),
    }

    report = []
    for fname, getter in getters.items():
        r = feature_report(fname, getter, all_names, bridges, strict)
        report.append(r)
        print(
            f"\n[{fname}] pop median {r['pop_median']} mean {r['pop_mean']}"
            f"\n   bridge(280) median {r['bridge_median']} mean {r['bridge_mean']} rankfrac {r['bridge_rankfrac']}"
            f"\n   strict(34)  median {r['strict_median']} mean {r['strict_mean']} rankfrac {r['strict_rankfrac']}"
        )

    out = {
        "population": len(all_names),
        "n_bridges": len(bridges),
        "n_strict": len(strict),
        "features": report,
        "note": "rankfrac = P(bridge feature > random population theorem); 0.5 = no diff.",
    }
    json.dump(out, open(RESULTS / "bridge_structural_profile.json", "w"), indent=2)
    print("\nwrote results/bridge_structural_profile.json")
