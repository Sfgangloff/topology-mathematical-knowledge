"""
Is the ~16-20x DEPLETION-vs-config-null itself ALGORITHM-stable?

a-0052 (e-0052, bridge_algorithm_robustness) showed the operational-bridge
population is NOT algorithm-invariant in ABSOLUTE scale or named IDENTITY: raw
counts span 98 (fastgreedy) -> 295 (leiden) -> 23,134 (infomap), cross-algorithm
Jaccard ~0.03, empty all-algorithm core. But it explicitly flagged a caveat:

  "counts here are raw, not null-normalized; whether the ~16-20x depletion-vs-null
   itself is algorithm-stable is untested and a natural follow-up."

e-0049 (a-0043/a-0045, bridge_degree_null) established, on the CANONICAL Leiden
partition, that real proofs span FAR FEWER pure clusters than a degree-preserving
(configuration) null predicts: observed 0.504% vs USAGE-null 8.0% -> ~16x DEPLETION.
That is a RELATIVE, per-partition statistic (observed / null-expected at matched
premise counts). It could be algorithm-stable even when the absolute count is not:
a finer partition inflates BOTH the observed bridge rate AND the null-expected rate,
so their RATIO -- the real signal that premises are domain-LOCAL -- may survive.

This experiment re-measures the depletion ratio per algorithm. For each of the 5
community-detection partitions e-0052 used (leiden_modularity, louvain, fastgreedy,
label_propagation, infomap) over the SAME deterministically-sorted LCC, it recomputes
the e-0049 config-null ENTIRELY within that partition's pure-cluster universe:

  observed bridge rate  = P(a theorem's actual premises span >=2 distinct pure clusters)
  USAGE-null rate       = closed-form E[same] when each theorem's k premises are redrawn
                          iid from the usage-weighted global premise distribution
  TYPE-null  rate       = same with the type-weighted (distinct-premise) distribution
  enrichment            = observed_bridges / null_expected_bridges  (<1 => DEPLETION)

Both observed and null use the SAME tactic_steps premise reconstruction e-0049 used
(which reproduces the 280 canonical bridges exactly on the Leiden partition), so the
ratio is internally consistent per partition. The ONLY thing that varies across rows
is the partition (hence the pure-cluster set), exactly as in e-0052.

Validity question: is "true cross-domain bridges are ~16-20x DEPLETED relative to a
degree-preserving null" (the e-0049 result underpinning q-0006/q-0007's claim that
operational bridges are genuine rarities defying premise locality) an algorithm-agnostic
property of Mathlib's premise structure, or itself a modularity-resolution artifact?

CPU-only; no new data. Reuses the e-0052 partition generation and the e-0049 closed-form.

Run:  python3.12 -m src.bridge_algorithm_null
"""

import json
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "bridge_algorithm_null.json"

SEED = 42  # for the seeded algorithms (leiden / label_propagation)


def load_premises():
    """Reconstruct each theorem's distinct out-premises (e-0049's universe)."""
    out_prem = defaultdict(set)
    with open(DATA_DIR / "tactic_steps.jsonl") as f:
        for line in f:
            o = json.loads(line)
            n = o["full_name"]
            for p in o["premises"]:
                if p != n:
                    out_prem[n].add(p)
    return out_prem


def null_prob_ge2(k, w0, wvals):
    """Closed-form P(>=2 distinct pure clusters) for k iid premise draws (e-0049)."""
    if k <= 1:
        return 0.0
    p0 = w0 ** k
    p_exactly1 = 0.0
    for wc in wvals:
        if wc > 0.0:
            p_exactly1 += (w0 + wc) ** k - p0
    return max(0.0, 1.0 - p0 - p_exactly1)


def depletion_for_partition(out_prem, cmap, pure):
    """Recompute the e-0049 observed/null depletion within ONE partition's pure set."""
    theorems = [(n, prems) for n, prems in out_prem.items() if prems]
    n_pop = len(theorems)

    # observed bridges: actual premises span >=2 distinct pure clusters
    obs_bridge = 0
    for n, prems in theorems:
        pcl = {cmap[p] for p in prems if cmap.get(p) in pure}
        if len(pcl) >= 2:
            obs_bridge += 1

    # global premise distributions over pure clusters (type- and usage-weighted)
    usage_count = defaultdict(int)
    for _, prems in theorems:
        for p in prems:
            usage_count[p] += 1
    distinct_premises = list(usage_count.keys())

    type_pure = defaultdict(int)
    type_total = len(distinct_premises)
    for p in distinct_premises:
        c = cmap.get(p)
        if c in pure:
            type_pure[c] += 1
    type_w0 = 1.0 - sum(type_pure.values()) / type_total
    type_wvals = [type_pure[c] / type_total for c in type_pure]

    usage_pure = defaultdict(int)
    usage_total = 0
    for p, u in usage_count.items():
        usage_total += u
        c = cmap.get(p)
        if c in pure:
            usage_pure[c] += u
    usage_w0 = 1.0 - sum(usage_pure.values()) / usage_total
    usage_wvals = [usage_pure[c] / usage_total for c in usage_pure]

    exp_type = sum(null_prob_ge2(len(prems), type_w0, type_wvals) for _, prems in theorems)
    exp_usage = sum(null_prob_ge2(len(prems), usage_w0, usage_wvals) for _, prems in theorems)

    return {
        "population": n_pop,
        "observed_bridges": obs_bridge,
        "observed_rate": round(obs_bridge / n_pop, 6),
        "type_p_pure": round(1 - type_w0, 5),
        "usage_p_pure": round(1 - usage_w0, 5),
        "type_null_bridges": round(exp_type, 1),
        "type_null_rate": round(exp_type / n_pop, 6),
        "usage_null_bridges": round(exp_usage, 1),
        "usage_null_rate": round(exp_usage / n_pop, 6),
        # enrichment < 1  =>  DEPLETION; depletion_factor = 1/enrichment
        "enrichment_type": round(obs_bridge / exp_type, 4) if exp_type > 0 else None,
        "enrichment_usage": round(obs_bridge / exp_usage, 4) if exp_usage > 0 else None,
        "depletion_factor_usage": round(exp_usage / obs_bridge, 2) if obs_bridge > 0 else None,
        "depletion_factor_type": round(exp_type / obs_bridge, 2) if obs_bridge > 0 else None,
    }


def main():
    print("Loading premises (tactic_steps reconstruction)…")
    out_prem = load_premises()

    print("Loading graph…")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)

    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    nodes = sorted(G_und.subgraph(lcc).nodes())  # deterministic order (a-0047 fix)
    G_lcc = G_und.subgraph(nodes).copy()
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_lcc.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges).simplify()
    print(f"LCC: {G_ig.vcount():,} nodes, {G_ig.ecount():,} edges (simplified)")

    def part_leiden():
        p = leidenalg.find_partition(
            G_ig, leidenalg.ModularityVertexPartition, n_iterations=10, seed=SEED
        )
        return list(p.membership)

    algos = {
        "leiden_modularity": part_leiden,
        "louvain": lambda: G_ig.community_multilevel().membership,
        "fastgreedy": lambda: G_ig.community_fastgreedy().as_clustering().membership,
        "label_propagation": lambda: G_ig.community_label_propagation().membership,
        "infomap": lambda: G_ig.community_infomap().membership,
    }

    per_algo = []
    for name, fn in algos.items():
        print(f"\n--- {name} ---")
        membership = fn()
        cmap = {nodes[i]: membership[i] for i in range(len(nodes))}
        labels = label_clusters(cmap, G)
        pure = {
            c for c, info in labels.items()
            if (not info["is_mixed"]) and info["dominant_fraction"] >= 0.50
        }
        d = depletion_for_partition(out_prem, cmap, pure)
        d["algorithm"] = name
        d["n_clusters"] = len(set(membership))
        d["n_pure_clusters"] = len(pure)
        per_algo.append(d)
        print(f"  clusters={d['n_clusters']} pure={d['n_pure_clusters']} "
              f"obs={d['observed_bridges']} ({100*d['observed_rate']:.3f}%) "
              f"USAGEnull={d['usage_null_bridges']} ({100*d['usage_null_rate']:.3f}%) "
              f"depletion={d['depletion_factor_usage']}x")

    depl_usage = [d["depletion_factor_usage"] for d in per_algo if d["depletion_factor_usage"]]
    depl_type = [d["depletion_factor_type"] for d in per_algo if d["depletion_factor_type"]]
    summary = {
        "n_algorithms": len(algos),
        "algorithms": list(algos.keys()),
        "depletion_usage_range": [min(depl_usage), max(depl_usage)],
        "depletion_usage_mean": round(statistics.mean(depl_usage), 2),
        "depletion_type_range": [min(depl_type), max(depl_type)],
        "depletion_type_mean": round(statistics.mean(depl_type), 2),
        "all_partitions_depleted": all(
            d["enrichment_usage"] is not None and d["enrichment_usage"] < 1.0
            for d in per_algo
        ),
        "note": "Per-partition e-0049 config-null depletion. enrichment_usage<1 => "
                "observed cross-domain bridging is DEPLETED vs a degree-preserving null "
                "(premises are domain-LOCAL). depletion_factor_usage = null/observed. "
                "Tests whether the e-0049 ~16-20x depletion is algorithm-stable even "
                "though e-0052 showed absolute scale & identity are not.",
    }

    out = {"summary": summary, "per_algo": per_algo}
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
