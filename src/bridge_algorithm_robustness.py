"""
ALGORITHM-robustness of bridge DETECTION (a new axis beyond seed-robustness).

Every prior robustness check on the operational-bridge population varied only the
Leiden RANDOM SEED (e-0044 partition stability; e-0050/e-0051 detection stability)
or used a degree-preserving config NULL (e-0049). None tested whether the scarce,
depth-filterable operational-bridge phenomenon is an artifact of the LEIDEN-modularity
optimizer specifically, or whether it survives a genuinely DIFFERENT community-detection
algorithm (different objective and optimizer).

This experiment runs the CANONICAL detection pipeline UNCHANGED
  label_clusters -> find_bridge_theorems (raw) -> indegree_sweep/find_elbow/filter (strict)
on partitions produced by 5 different community-detection algorithms over the SAME LCC:

  * leiden_modularity  -- the canonical baseline (ModularityVertexPartition, seed 42)
  * louvain            -- igraph community_multilevel (greedy modularity, different optimizer)
  * fastgreedy         -- agglomerative greedy modularity (Clauset-Newman-Moore)
  * label_propagation  -- non-modularity, near-linear-time label spreading
  * infomap            -- map-equation / random-walk objective (not modularity at all)

To make ALGORITHM the only varying factor (a-0047 showed PYTHONHASHSEED node ordering
silently perturbs the partition), the LCC nodes are SORTED before the igraph is built,
and the igraph is simplified (self-loops + multi-edges collapsed) once and shared.

Reports, per algorithm: n_clusters, n_pure_clusters, raw / strict bridge counts, elbow
theta, and Jaccard of the detected bridge THEOREM sets vs the leiden_modularity baseline.
Also: the cross-ALGORITHM core (theorems flagged a raw bridge under ALL 5 algorithms) and
recovery of the canonical stored sets (280 raw / 34 strict).

The validity question: is "a scarce operational-bridge population that a low-indegree
elbow cuts to a few dozen" (q-0006/q-0007) an algorithm-agnostic phenomenon, or a Leiden
artifact? CPU-only, no new data; reuses the canonical detection code.

Run:  python3.12 -m src.bridge_algorithm_robustness
"""

import json
import pickle
import statistics
from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters
from bridge_theorems import find_bridge_theorems
from depth_filter import indegree_sweep, find_elbow, filter_bridges

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "bridge_algorithm_robustness.json"

BIG = 10 ** 9     # top_k large enough to keep ALL raw bridges
SEED = 42         # for the seeded algorithms (leiden / label_propagation)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def detect_for_partition(G, cluster_map):
    """Canonical detection pipeline on one partition; return (raw_set, strict_set, theta, n_pure)."""
    labels = label_clusters(cluster_map, G)
    n_pure = sum(
        1 for info in labels.values()
        if not info["is_mixed"] and info["dominant_fraction"] >= 0.50
    )
    raw = find_bridge_theorems(G, cluster_map, labels, top_k=BIG)
    raw_set = {b["theorem"] for b in raw}

    thresholds, counts, theorem_bridges = indegree_sweep(G, cluster_map, labels)
    if not theorem_bridges:
        return raw_set, set(), None, n_pure
    theta = find_elbow(thresholds, counts)
    strict = filter_bridges(theorem_bridges, theta, labels)
    strict_set = {b["theorem"] for b in strict}
    return raw_set, strict_set, theta, n_pure


def membership_to_map(membership, nodes):
    return {nodes[i]: membership[i] for i in range(len(nodes))}


def main():
    print("Loading graph…")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)

    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    # Deterministic node order so ALGORITHM is the only varying factor (a-0047 fix).
    nodes = sorted(G_und.subgraph(lcc).nodes())
    G_lcc = G_und.subgraph(nodes).copy()
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_lcc.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges)
    G_ig = G_ig.simplify()  # drop self-loops + multi-edges; shared by all algorithms
    print(f"LCC: {G_ig.vcount():,} nodes, {G_ig.ecount():,} edges (simplified)")

    # Partition each algorithm produces (membership list aligned to `nodes`).
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

    canon_raw = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems.json"))}
    canon_strict = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems_filtered.json"))}

    raw_sets = {}
    strict_sets = {}
    per_algo = []
    for name, fn in algos.items():
        print(f"\n--- {name} ---")
        membership = fn()
        cluster_map = membership_to_map(membership, nodes)
        raw_set, strict_set, theta, n_pure = detect_for_partition(G, cluster_map)
        raw_sets[name] = raw_set
        strict_sets[name] = strict_set
        rec_raw = len(raw_set & canon_raw) / len(canon_raw) if canon_raw else 0.0
        rec_strict = len(strict_set & canon_strict) / len(canon_strict) if canon_strict else 0.0
        per_algo.append({
            "algorithm": name,
            "n_clusters": len(set(membership)),
            "n_pure_clusters": n_pure,
            "n_raw_bridges": len(raw_set),
            "n_strict_bridges": len(strict_set),
            "elbow_theta": theta,
            "recovery_of_canonical_raw": round(rec_raw, 4),
            "recovery_of_canonical_strict": round(rec_strict, 4),
        })
        print(f"  clusters={len(set(membership))} pure={n_pure} raw={len(raw_set)} "
              f"strict={len(strict_set)} theta={theta} "
              f"rec_raw={rec_raw:.2f} rec_strict={rec_strict:.2f}")

    names = list(algos.keys())
    baseline = "leiden_modularity"

    # Jaccard vs the canonical Leiden baseline, and full pairwise.
    vs_baseline = {
        n: {
            "jaccard_raw": round(jaccard(raw_sets[n], raw_sets[baseline]), 4),
            "jaccard_strict": round(jaccard(strict_sets[n], strict_sets[baseline]), 4),
        }
        for n in names if n != baseline
    }
    pairwise = []
    for a, b in combinations(names, 2):
        pairwise.append({
            "a": a, "b": b,
            "jaccard_raw": round(jaccard(raw_sets[a], raw_sets[b]), 4),
            "jaccard_strict": round(jaccard(strict_sets[a], strict_sets[b]), 4),
        })

    # Cross-ALGORITHM core: flagged by ALL algorithms.
    core_raw = set.intersection(*raw_sets.values())
    core_strict = (set.intersection(*strict_sets.values())
                   if all(strict_sets.values()) else set())
    union_raw = set().union(*raw_sets.values())

    from collections import Counter
    freq = Counter()
    for s in raw_sets.values():
        for t in s:
            freq[t] += 1
    freq_hist = Counter(freq.values())  # #theorems flagged in exactly k algorithms

    nraw = [d["n_raw_bridges"] for d in per_algo]
    nstrict = [d["n_strict_bridges"] for d in per_algo]
    jr = [p["jaccard_raw"] for p in pairwise]
    js = [p["jaccard_strict"] for p in pairwise]

    summary = {
        "n_algorithms": len(names),
        "algorithms": names,
        "canonical_raw": len(canon_raw),
        "canonical_strict": len(canon_strict),
        "n_raw_mean": round(statistics.mean(nraw), 1),
        "n_raw_range": [min(nraw), max(nraw)],
        "n_strict_mean": round(statistics.mean(nstrict), 1),
        "n_strict_range": [min(nstrict), max(nstrict)],
        "pairwise_jaccard_raw_mean": round(statistics.mean(jr), 4),
        "pairwise_jaccard_raw_range": [min(jr), max(jr)],
        "pairwise_jaccard_strict_mean": round(statistics.mean(js), 4),
        "pairwise_jaccard_strict_range": [min(js), max(js)],
        "core_raw_in_all_algos": len(core_raw),
        "core_strict_in_all_algos": len(core_strict),
        "union_raw_any_algo": len(union_raw),
        "raw_algo_frequency_hist": {str(k): freq_hist[k] for k in sorted(freq_hist)},
    }

    out = {
        "summary": summary,
        "per_algo": per_algo,
        "vs_baseline": vs_baseline,
        "pairwise": pairwise,
        "core_raw_sample": sorted(core_raw)[:40],
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
