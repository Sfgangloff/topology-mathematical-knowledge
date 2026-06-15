"""
Seed-robustness of bridge DETECTION (not just domain recovery).

a-0044 (clustering_stability, e-0044) showed the aggregate domain-recovery claim is
seed-robust (module-NMI 0.2847 +/- 0.0025 across 7 seeds) but flagged a real caveat:
the EXACT partition is only moderately reproducible (cluster count 93-223, pairwise
ARI ~0.46), so "cluster-IDENTITY-dependent downstream claims -- the specific 34 bridge
theorems, the exact 138 clusters -- rest on one particular partition that would shift
across seeds." e-0044 measured partition stability but NEVER re-ran bridge detection.

This experiment closes that gap directly. For each of the SAME 7 seeds e-0044 used, it
re-runs Leiden on the EXACT LCC object cluster.py uses, then re-runs the CANONICAL
detection pipeline UNCHANGED on each partition:

  * label_clusters.label_clusters  -> pure clusters (dom module >= 50%)
  * bridge_theorems.find_bridge_theorems -> RAW operational bridges (premises span >=2 pure clusters)
  * depth_filter.indegree_sweep + find_elbow + filter_bridges -> STRICT bridges (elbow theta)

and measures the stability of the DETECTED THEOREM SETS:

  * per-seed raw / strict bridge counts and the auto-selected elbow theta,
  * pairwise Jaccard of the raw and strict bridge THEOREM sets across seeds,
  * recovery of the canonical stored sets (results/bridge_theorems.json = 280 raw;
    results/bridge_theorems_filtered.json = 34 strict) by each fresh partition,
  * a stable "core" of theorems flagged as a bridge in ALL 7 seeds.

The honest validity question a-0044 raised: is "operational bridges exist and are a
scarce, meaningful object" (the q-0006/q-0007 claim) seed-robust, even if the exact
membership list is not? CPU-only, no new data; reuses the canonical detection code.

Run:  python3.12 -m src.bridge_detection_stability
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
OUT_FILE = RESULTS_DIR / "bridge_detection_stability.json"

SEEDS = [0, 1, 7, 42, 123, 2024, 31337]
N_ITERATIONS = 10  # matches cluster.py / e-0044
BIG = 10 ** 9      # top_k large enough to keep ALL raw bridges (no truncation)


def _nx_to_igraph(G_nx: nx.Graph):
    nodes = list(G_nx.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_nx.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges)
    return G_ig, nodes


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def detect_for_partition(G, cluster_map):
    """Run the canonical detection pipeline on one partition; return (raw_set, strict_set, theta)."""
    labels = label_clusters(cluster_map, G)
    raw = find_bridge_theorems(G, cluster_map, labels, top_k=BIG)
    raw_set = {b["theorem"] for b in raw}

    thresholds, counts, theorem_bridges = indegree_sweep(G, cluster_map, labels)
    if not theorem_bridges:
        return raw_set, set(), None
    theta = find_elbow(thresholds, counts)
    strict = filter_bridges(theorem_bridges, theta, labels)
    strict_set = {b["theorem"] for b in strict}
    return raw_set, strict_set, theta


def main():
    print("Loading graph…")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)

    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    G_lcc = G_und.subgraph(lcc).copy()
    print(f"LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")

    G_ig, nodes = _nx_to_igraph(G_lcc)

    # Canonical stored detected sets (the objects a-0044 worried about).
    canon_raw = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems.json"))}
    canon_strict = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems_filtered.json"))}
    print(f"Canonical stored: {len(canon_raw)} raw, {len(canon_strict)} strict")

    # Detect on each seed's fresh partition.
    raw_sets = {}
    strict_sets = {}
    per_seed = []
    for s in SEEDS:
        part = leidenalg.find_partition(
            G_ig, leidenalg.ModularityVertexPartition, n_iterations=N_ITERATIONS, seed=s
        )
        m = list(part.membership)
        cluster_map = {nodes[i]: m[i] for i in range(len(nodes))}
        raw_set, strict_set, theta = detect_for_partition(G, cluster_map)
        raw_sets[s] = raw_set
        strict_sets[s] = strict_set
        rec_raw = len(raw_set & canon_raw) / len(canon_raw) if canon_raw else 0.0
        rec_strict = len(strict_set & canon_strict) / len(canon_strict) if canon_strict else 0.0
        per_seed.append({
            "seed": s,
            "n_clusters": len(set(m)),
            "n_raw_bridges": len(raw_set),
            "n_strict_bridges": len(strict_set),
            "elbow_theta": theta,
            "recovery_of_canonical_raw": round(rec_raw, 4),
            "recovery_of_canonical_strict": round(rec_strict, 4),
        })
        print(f"  seed={s:>6}: clusters={len(set(m)):>4} raw={len(raw_set):>4} "
              f"strict={len(strict_set):>4} theta={theta} "
              f"rec_raw={rec_raw:.2f} rec_strict={rec_strict:.2f}")

    # Pairwise Jaccard across fresh seeds (set-level stability of the detected object).
    pairwise = []
    for a, b in combinations(SEEDS, 2):
        pairwise.append({
            "a": a, "b": b,
            "jaccard_raw": round(jaccard(raw_sets[a], raw_sets[b]), 4),
            "jaccard_strict": round(jaccard(strict_sets[a], strict_sets[b]), 4),
        })

    # Stable core: theorems flagged in ALL seeds; and frequency distribution.
    all_raw = set().union(*raw_sets.values())
    all_strict = set().union(*strict_sets.values())
    core_raw = set.intersection(*raw_sets.values())
    core_strict = set.intersection(*strict_sets.values()) if all(strict_sets.values()) else set()

    def freq_hist(sets):
        from collections import Counter
        c = Counter()
        for s in sets.values():
            for t in s:
                c[t] += 1
        hist = Counter(c.values())  # how many theorems appear in exactly k seeds
        return {str(k): hist[k] for k in sorted(hist)}

    jr = [p["jaccard_raw"] for p in pairwise]
    js = [p["jaccard_strict"] for p in pairwise]
    rec_raw_all = [d["recovery_of_canonical_raw"] for d in per_seed]
    rec_strict_all = [d["recovery_of_canonical_strict"] for d in per_seed]
    nraw = [d["n_raw_bridges"] for d in per_seed]
    nstrict = [d["n_strict_bridges"] for d in per_seed]

    summary = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
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
        "recovery_canonical_raw_mean": round(statistics.mean(rec_raw_all), 4),
        "recovery_canonical_strict_mean": round(statistics.mean(rec_strict_all), 4),
        "core_raw_in_all_seeds": len(core_raw),
        "core_strict_in_all_seeds": len(core_strict),
        "union_raw_any_seed": len(all_raw),
        "union_strict_any_seed": len(all_strict),
        "raw_seed_frequency_hist": freq_hist(raw_sets),
        "strict_seed_frequency_hist": freq_hist(strict_sets),
    }

    out = {"summary": summary, "per_seed": per_seed, "pairwise": pairwise}
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
