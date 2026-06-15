"""
ALGORITHM-robustness of the INTEGRATIVE-HUB claim (a-0010).

a-0010 found that, by a Tononi-Edelman neural-complexity proxy
(CN = integration x differentiation over the cluster-interaction matrix),
SetTheory is the most integrative-yet-specialised domain of Mathlib -- the
dominant operational bridge hub (CN=1.037, a decisive #1). That result rests on
ONE Leiden partition (seed 42), and the seed/algorithm-robustness chain
(a-0044, a-0050, a-0051, a-0052) repeatedly flagged "SetTheory-as-hub-by-cluster-id
rests on one particular partition" as an UNRESOLVED caveat: nobody re-measured
WHICH domain is the integrative hub under a different clustering.

This experiment closes that gap. Over the SAME deterministically-sorted LCC the
algorithm-robustness work used (a-0052), it builds partitions from 5 community-detection
algorithms (leiden_modularity, louvain, fastgreedy, label_propagation, infomap),
re-runs label_clusters + find_bridge_theorems on each, recomputes the SAME CN proxy
(neural_complexity.compute_neural_complexity), and reports -- per algorithm -- which
DOMINANT-MODULE cluster is the top integrative hub, and SetTheory's hub rank.

Because cluster IDs are not comparable across partitions, the hub is reported by the
top-CN cluster's dominant module (a partition-independent, file_path-derived label).
The CN here is built from the RAW operational bridges (find_bridge_theorems, top_k=inf),
not the volatile depth-filtered strict set, so the W matrix is the stable, denser one;
the leiden row also reproduces a-0010's SetTheory-#1 under this raw construction, anchoring
the comparison.

Question: is "SetTheory is the integrative hub" an algorithm-invariant property of
Mathlib's premise structure, or an artifact of the Leiden-modularity partition a-0010 used?

CPU-only, no new data; reuses the canonical labelling/detection/CN code.

Run:  python3.12 -m src.hub_algorithm_robustness
"""

import json
import pickle
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters
from bridge_theorems import find_bridge_theorems
from depth_filter import indegree_sweep, find_elbow, filter_bridges
from neural_complexity import compute_neural_complexity

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "hub_algorithm_robustness.json"

BIG = 10 ** 9
SEED = 42


def membership_to_map(membership, nodes):
    return {nodes[i]: membership[i] for i in range(len(nodes))}


def _rank_by_cn(bridge_theorems, labels, cluster_map):
    """Build the CN proxy from a list of bridges (with cross_premises) and rank clusters."""
    cluster_metrics, global_cn, W, cluster_ids = compute_neural_complexity(
        bridge_theorems, labels, cluster_map
    )
    ranked = sorted(
        (
            {
                "cluster_id": int(cid),
                "label": m["label"],
                "dominant_module": labels[cid]["dominant_module"],
                "cn": m["neural_complexity"],
                "integration": m["integration"],
                "differentiation": m["differentiation"],
                "bridge_connections": m["total_bridge_connections"],
                "size": m["size"],
                "is_mixed": m["is_mixed"],
            }
            for cid, m in cluster_metrics.items()
        ),
        key=lambda x: -x["cn"],
    )
    set_rank, set_cn = None, None
    for i, r in enumerate(ranked, 1):
        if r["dominant_module"] == "SetTheory":
            set_rank, set_cn = i, r["cn"]
            break
    return ranked, set_rank, set_cn, global_cn


def hub_for_partition(G, cluster_map):
    """Label -> raw & strict bridges -> CN; return ranked hubs (both) and SetTheory's rank.

    The RAW construction uses all operational bridges (find_bridge_theorems); the STRICT
    construction uses the depth-filtered bridges (the exact object a-0010's CN used).
    """
    labels = label_clusters(cluster_map, G)
    n_pure = sum(
        1 for info in labels.values()
        if not info["is_mixed"] and info["dominant_fraction"] >= 0.50
    )

    raw = find_bridge_theorems(G, cluster_map, labels, top_k=BIG)
    # find_bridge_theorems output -> compute_neural_complexity shape (cross_premises tuples).
    raw_bt = [
        {
            "theorem": b["theorem"],
            "n_clusters": b["n_premise_clusters"],
            "cross_premises": [("", c, 0) for c in b["premise_cluster_ids"]],
        }
        for b in raw
    ]
    raw_ranked, raw_set_rank, raw_set_cn, raw_global = _rank_by_cn(raw_bt, labels, cluster_map)

    # STRICT: depth-filter via the auto elbow (the a-0010 construction).
    sweep = indegree_sweep(G, cluster_map, labels)
    strict_bt = []
    theta = None
    if len(sweep) == 3 and sweep[2]:
        thresholds, counts, theorem_bridges = sweep
        theta = find_elbow(thresholds, counts)
        strict_bt = filter_bridges(theorem_bridges, theta, labels)  # carry cross_premises
    if strict_bt:
        strict_ranked, strict_set_rank, strict_set_cn, strict_global = _rank_by_cn(
            strict_bt, labels, cluster_map
        )
    else:
        strict_ranked, strict_set_rank, strict_set_cn, strict_global = [], None, None, 0.0

    return {
        "n_pure": n_pure,
        "theta": theta,
        "n_raw": len(raw_bt),
        "n_strict": len(strict_bt),
        "raw": (raw_ranked, raw_set_rank, raw_set_cn, raw_global),
        "strict": (strict_ranked, strict_set_rank, strict_set_cn, strict_global),
    }


def main():
    print("Loading graph…")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)

    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    nodes = sorted(G_und.subgraph(lcc).nodes())          # deterministic order (a-0047 fix)
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

    def top5_of(ranked):
        return [
            {
                "rank": i,
                "dominant_module": r["dominant_module"],
                "label": r["label"],
                "cn": r["cn"],
                "integration": r["integration"],
                "differentiation": r["differentiation"],
                "bridge_connections": r["bridge_connections"],
            }
            for i, r in enumerate(ranked[:5], 1)
        ]

    per_algo = []
    for name, fn in algos.items():
        print(f"\n--- {name} ---")
        membership = fn()
        cluster_map = membership_to_map(membership, nodes)
        res = hub_for_partition(G, cluster_map)

        raw_ranked, raw_set_rank, raw_set_cn, raw_global = res["raw"]
        strict_ranked, strict_set_rank, strict_set_cn, strict_global = res["strict"]
        raw_top = raw_ranked[0] if raw_ranked else {}
        strict_top = strict_ranked[0] if strict_ranked else {}

        per_algo.append({
            "algorithm": name,
            "n_clusters": len(set(membership)),
            "n_pure_clusters": res["n_pure"],
            "elbow_theta": res["theta"],
            "n_raw_bridges": res["n_raw"],
            "n_strict_bridges": res["n_strict"],
            "raw": {
                "global_cn": round(raw_global, 4),
                "top_hub_module": raw_top.get("dominant_module"),
                "top_hub_cn": raw_top.get("cn"),
                "settheory_hub_rank": raw_set_rank,
                "settheory_hub_cn": raw_set_cn,
                "top5": top5_of(raw_ranked),
            },
            "strict": {
                "global_cn": round(strict_global, 4),
                "top_hub_module": strict_top.get("dominant_module"),
                "top_hub_cn": strict_top.get("cn"),
                "settheory_hub_rank": strict_set_rank,
                "settheory_hub_cn": strict_set_cn,
                "top5": top5_of(strict_ranked),
            },
        })
        print(f"  clusters={len(set(membership))} pure={res['n_pure']} theta={res['theta']} "
              f"raw={res['n_raw']} strict={res['n_strict']}")
        print(f"  RAW    top_hub={raw_top.get('dominant_module')} (CN={raw_top.get('cn')}) "
              f"SetTheory_rank={raw_set_rank}")
        print(f"  STRICT top_hub={strict_top.get('dominant_module')} (CN={strict_top.get('cn')}) "
              f"SetTheory_rank={strict_set_rank}")

    def summarise(kind):
        top_modules = [d[kind]["top_hub_module"] for d in per_algo]
        set_ranks = [d[kind]["settheory_hub_rank"] for d in per_algo]
        known = [r for r in set_ranks if r is not None]
        return {
            "top_hub_module_per_algo": dict(zip(algos.keys(), top_modules)),
            "settheory_top_hub_in_n_algos": sum(1 for m in top_modules if m == "SetTheory"),
            "settheory_top3_in_n_algos": sum(1 for r in set_ranks if r is not None and r <= 3),
            "settheory_rank_per_algo": dict(zip(algos.keys(), set_ranks)),
            "settheory_rank_range": [min(known), max(known)] if known else None,
            "distinct_top_hub_modules": sorted(set(m for m in top_modules if m)),
        }

    summary = {
        "n_algorithms": len(algos),
        "algorithms": list(algos.keys()),
        "canonical_a0010": "SetTheory CN=1.037, #1 (stored partition + strict bridges)",
        "note": ("CN here is recomputed on FRESH deterministically-sorted partitions; the "
                 "RAW construction uses all operational bridges, STRICT uses the depth-filtered "
                 "set (the a-0010 object). Hub reported by top-CN cluster's dominant module."),
        "raw": summarise("raw"),
        "strict": summarise("strict"),
    }

    out = {"summary": summary, "per_algo": per_algo}
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
