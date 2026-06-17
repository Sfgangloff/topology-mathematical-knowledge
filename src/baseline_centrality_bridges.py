"""
q-0014: Do simpler graph baselines recover the same bridge theorems as the
depth-filtered cross-cluster detector?

Baselines compared against the 34 depth-filtered bridge theorems
(results/bridge_theorems_filtered.json):

  1. Betweenness centrality (sampled, k=2000 pivots) — top-K nodes.
  2. Cross-module out-degree: # of distinct top-level Mathlib modules a node's
     out-neighbours (premises) live in. A node "spans modules" if its proof cites
     premises from >=2 distinct modules — the directory/import-style baseline.
  3. Cross-cluster out-degree (NO depth filter): # distinct PURE clusters reached
     by a node's premises (the raw loose bridge signal, before in-degree filtering).

For each baseline we report overlap (precision@K / recall) against the 34-bridge set,
and rank-statistics of the 34 bridges within each baseline ordering.
"""
import json, pickle
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"


def top_module(module: str) -> str:
    # module like 'Mathlib.Topology.Basic' or '.lake/.../Init/Prelude'
    parts = module.split(".")
    if "Mathlib" in parts:
        i = parts.index("Mathlib")
        return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    # non-mathlib: take last meaningful segment group
    for key in ("Init", "Std", "Batteries", "Lean", "Mathlib"):
        if key in module:
            return key
    return parts[-1] if parts else "Unknown"


def main():
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    cluster_map = json.load(open(DATA / "clusters.json"))
    labels = json.load(open(DATA / "cluster_labels.json"))
    bridges = json.load(open(RES / "bridge_theorems_filtered.json"))
    bridge_set = {b["theorem"] for b in bridges}
    K = len(bridge_set)  # 34

    pure_clusters = {
        cid for cid, info in labels.items()
        if not info["is_mixed"] and info["dominant_fraction"] >= 0.50
    }

    # ---- module per node ----
    node_mod = {n: top_module(G.nodes[n].get("module", "")) for n in G.nodes()}

    # ---- Baseline 1: sampled betweenness ----
    print("computing sampled betweenness (k=2000)...")
    bc = nx.betweenness_centrality(G, k=2000, seed=42, normalized=True)
    bc_rank = sorted(bc, key=lambda n: bc[n], reverse=True)

    # ---- Baseline 2: cross-module out-degree ----
    cross_mod = {}
    for n in G.nodes():
        mods = {node_mod[p] for p in G.successors(n) if p in node_mod}
        cross_mod[n] = len(mods)
    crossmod_rank = sorted(cross_mod, key=lambda n: cross_mod[n], reverse=True)

    # ---- Baseline 3: cross-cluster out-degree (loose, no depth filter) ----
    cross_clu = {}
    for n in G.nodes():
        cid_self = cluster_map.get(n)
        clusters = {cluster_map[p] for p in G.successors(n)
                    if p in cluster_map and cluster_map[p] in pure_clusters
                    and cluster_map[p] != cid_self}
        cross_clu[n] = len(clusters)
    crossclu_rank = sorted(cross_clu, key=lambda n: cross_clu[n], reverse=True)

    def overlap_stats(ranking, score):
        topK = set(ranking[:K])
        inter = topK & bridge_set
        # ranks of the 34 bridges in this ranking
        pos = {n: i for i, n in enumerate(ranking)}
        bridge_ranks = sorted(pos[b] for b in bridge_set if b in pos)
        med = bridge_ranks[len(bridge_ranks) // 2] if bridge_ranks else None
        # how far down to reach 50% / 100% recall
        N = len(ranking)
        return {
            "precision_at_K": round(len(inter) / K, 3),
            "recall_at_K": round(len(inter) / K, 3),
            "n_overlap_topK": len(inter),
            "bridge_median_rank": med,
            "bridge_best_rank": bridge_ranks[0] if bridge_ranks else None,
            "bridge_worst_rank": bridge_ranks[-1] if bridge_ranks else None,
            "N_nodes": N,
            "bridge_score_range": [round(min(score[b] for b in bridge_set), 4),
                                    round(max(score[b] for b in bridge_set), 4)],
        }

    out = {
        "bridge_set_size": K,
        "n_pure_clusters": len(pure_clusters),
        "betweenness_sampled_k2000": overlap_stats(bc_rank, bc),
        "cross_module_outdegree": overlap_stats(crossmod_rank, cross_mod),
        "cross_cluster_outdegree_loose": overlap_stats(crossclu_rank, cross_clu),
    }

    # Spearman-style: do bridges score high on each baseline relative to all nodes?
    import statistics
    allnodes = list(G.nodes())
    for name, sc in [("betweenness", bc), ("cross_module", cross_mod),
                     ("cross_cluster_loose", cross_clu)]:
        vals = [sc[n] for n in allnodes]
        bvals = [sc[b] for b in bridge_set if b in sc]
        # percentile of median bridge score
        med_b = statistics.median(bvals)
        pct = sum(1 for v in vals if v <= med_b) / len(vals)
        out[name + "_median_bridge_percentile"] = round(pct, 3)

    json.dump(out, open(RES / "baseline_centrality_bridges.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
