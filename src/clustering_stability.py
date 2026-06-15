"""
Clustering stability / seed-robustness of the foundational domain recovery (q-0002).

The whole paper rests on ONE Leiden run (cluster.py, ModularityVertexPartition,
10 iterations, seed=42). a-0001/a-0002 claim that unsupervised clustering recovers
Mathlib's named domains, but no experiment checks whether that recovery is stable or
an artifact of the single random seed. Leiden is a stochastic local optimiser, so the
honest validity question is: do independent reruns agree with each other, with the
canonical stored partition, and with the Mathlib module ground truth?

This re-runs Leiden on the EXACT same object cluster.py uses (undirected LCC of the
theorem dependency graph) under several seeds and measures:

  * #clusters and modularity per seed (stability of the coarse structure),
  * ARI / NMI of each seed vs the canonical stored partition (data/clusters.json, seed 42),
  * pairwise ARI / NMI across the fresh seeds (run-to-run agreement),
  * NMI of each seed vs the top-level Mathlib module labels -- the "domain recovery"
    metric, anchoring to a-0002's reported NMI(top_cluster ; 1-level module)=0.307.

CPU-only, no new data; replicates cluster.cluster() with a swept seed.

Run:  python3.12 -m src.clustering_stability
"""

import json
import pickle
import statistics
from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "clustering_stability.json"

SEEDS = [0, 1, 7, 42, 123, 2024, 31337]
N_ITERATIONS = 10  # matches cluster.py


def _top_module(file_path: str) -> str:
    """Top-level Mathlib sub-module, identical convention to label_clusters._top_module."""
    if not file_path:
        return "Unknown"
    parts = Path(file_path).with_suffix("").parts
    try:
        idx = next(i for i, s in enumerate(parts) if s == "Mathlib")
        return parts[idx + 1] if idx + 1 < len(parts) else "Mathlib"
    except StopIteration:
        for p in parts:
            if p in ("Init", "Batteries", "Std", "lake"):
                return p
        return "External"


def _nx_to_igraph(G_nx: nx.Graph):
    nodes = list(G_nx.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_nx.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges)
    return G_ig, nodes


def run_leiden(G_ig, seed):
    part = leidenalg.find_partition(
        G_ig,
        leidenalg.ModularityVertexPartition,
        n_iterations=N_ITERATIONS,
        seed=seed,
    )
    return part


def main():
    print("Loading graph…")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)

    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    G_lcc = G_und.subgraph(lcc).copy()
    print(f"LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")

    G_ig, nodes = _nx_to_igraph(G_lcc)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Module ground-truth label per node (top-level Mathlib module).
    module_labels = [
        _top_module(G.nodes[n].get("file_path", "")) if G.has_node(n) else "Unknown"
        for n in nodes
    ]

    # Canonical stored partition (seed 42, from cluster.py).
    stored = json.load(open(DATA_DIR / "clusters.json"))
    stored_vec = [stored.get(n, -1) for n in nodes]  # -1 if absent (should be none)
    n_missing = stored_vec.count(-1)
    print(f"Stored partition: {len(set(stored.values()))} clusters, {n_missing} LCC nodes missing")

    # Run Leiden under each seed.
    memberships = {}
    per_seed = []
    for s in SEEDS:
        part = run_leiden(G_ig, s)
        m = list(part.membership)
        memberships[s] = m
        nclus = len(set(m))
        mod = part.modularity
        # vs stored canonical partition
        ari_stored = adjusted_rand_score(stored_vec, m)
        nmi_stored = normalized_mutual_info_score(stored_vec, m)
        # vs module ground truth
        nmi_mod = normalized_mutual_info_score(module_labels, m)
        per_seed.append({
            "seed": s,
            "n_clusters": nclus,
            "modularity": round(mod, 5),
            "ari_vs_stored": round(ari_stored, 4),
            "nmi_vs_stored": round(nmi_stored, 4),
            "nmi_vs_module": round(nmi_mod, 4),
        })
        print(f"  seed={s:>6}: clusters={nclus:>4} mod={mod:.4f} "
              f"ARI/NMI vs stored={ari_stored:.3f}/{nmi_stored:.3f} NMI vs module={nmi_mod:.3f}")

    # Pairwise agreement across fresh seeds.
    pairwise = []
    for a, b in combinations(SEEDS, 2):
        ari = adjusted_rand_score(memberships[a], memberships[b])
        nmi = normalized_mutual_info_score(memberships[a], memberships[b])
        pairwise.append({"a": a, "b": b, "ari": round(ari, 4), "nmi": round(nmi, 4)})

    aris = [p["ari"] for p in pairwise]
    nmis = [p["nmi"] for p in pairwise]
    nclus = [d["n_clusters"] for d in per_seed]
    nmi_mod_all = [d["nmi_vs_module"] for d in per_seed]

    summary = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "n_clusters_mean": round(statistics.mean(nclus), 2),
        "n_clusters_std": round(statistics.pstdev(nclus), 2),
        "n_clusters_range": [min(nclus), max(nclus)],
        "pairwise_ari_mean": round(statistics.mean(aris), 4),
        "pairwise_ari_min": round(min(aris), 4),
        "pairwise_nmi_mean": round(statistics.mean(nmis), 4),
        "pairwise_nmi_min": round(min(nmis), 4),
        "nmi_vs_module_mean": round(statistics.mean(nmi_mod_all), 4),
        "nmi_vs_module_std": round(statistics.pstdev(nmi_mod_all), 4),
        "anchor_a0002_nmi_top_module": 0.307,
    }

    out = {"summary": summary, "per_seed": per_seed, "pairwise": pairwise}
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
