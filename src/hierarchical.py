"""
Hierarchical re-clustering: run Leiden inside each top-level cluster
to discover sub-domain structure.

Output: data/sub_clusters.json  {theorem_name: [top_cluster_id, sub_cluster_id]}
        data/sub_cluster_labels.json
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

DATA_DIR    = Path(__file__).parent.parent / "data"
SUB_FILE    = DATA_DIR / "sub_clusters.json"
SUB_LABELS  = DATA_DIR / "sub_cluster_labels.json"


def _nx_to_igraph(G_nx: nx.Graph) -> tuple[ig.Graph, list]:
    nodes = list(G_nx.nodes())
    idx   = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G_nx.edges()]
    return ig.Graph(n=len(nodes), edges=edges), nodes


def _top_module(file_path: str) -> str:
    from pathlib import Path as P
    parts = P(file_path).with_suffix("").parts
    try:
        i = next(j for j, s in enumerate(parts) if s == "Mathlib")
        return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    except StopIteration:
        return "External"


def recluster(
    G: nx.DiGraph,
    cluster_map: dict[str, int],
    top_n_clusters: int = 15,
    min_cluster_size: int = 100,
) -> dict[str, list[int]]:
    """Re-run Leiden clustering within each of the top-N largest clusters.

    This reveals sub-domain structure: e.g. the LinearAlgebra cluster splits into
    42 sub-clusters corresponding to matrix theory, module theory, spectral theory, etc.

    Returns {theorem_name: [top_cluster_id, sub_cluster_id]}. Nodes in the LCC of
    their top-level cluster get a sub_cluster_id ≥ 0; nodes outside the LCC get -1.
    """
    G_und = G.to_undirected()

    # Count sizes
    sizes: Counter = Counter(cluster_map.values())
    target_clusters = [
        cid for cid, sz in sizes.most_common(top_n_clusters)
        if sz >= min_cluster_size
    ]

    sub_map: dict[str, list[int]] = {}

    for cid in target_clusters:
        members = [n for n, c in cluster_map.items() if c == cid and G_und.has_node(n)]
        sub_G = G_und.subgraph(members).copy()

        # Keep only the largest connected component inside this cluster
        comps = sorted(nx.connected_components(sub_G), key=len, reverse=True)
        if not comps:
            continue
        sub_G = sub_G.subgraph(comps[0]).copy()
        nodes_in = list(sub_G.nodes())

        if len(nodes_in) < 10:
            for n in members:
                sub_map[n] = [cid, 0]
            continue

        G_ig, nodes = _nx_to_igraph(sub_G)
        partition = leidenalg.find_partition(
            G_ig, leidenalg.ModularityVertexPartition,
            n_iterations=10, seed=42,
        )
        for i, n in enumerate(nodes):
            sub_map[n] = [cid, partition.membership[i]]

        # Nodes not in the LCC
        lcc_set = set(nodes_in)
        for n in members:
            if n not in lcc_set:
                sub_map[n] = [cid, -1]

        n_sub = len(set(partition.membership))
        print(f"  Cluster {cid:>3} ({len(members):>5} nodes) → {n_sub} sub-clusters")

    return sub_map


def label_sub_clusters(sub_map: dict[str, list[int]], G: nx.DiGraph) -> dict:
    """Label each sub-cluster by its dominant Mathlib module, same logic as label_clusters."""
    from label_clusters import _entropy
    import math

    # Build {(top, sub): Counter of modules}
    group_modules: dict[tuple, Counter] = defaultdict(Counter)
    for node, (top, sub) in sub_map.items():
        fp = G.nodes[node].get("file_path", "") if G.has_node(node) else ""
        mod = _top_module(fp) if fp else "Unknown"
        group_modules[(top, sub)][mod] += 1

    labels = {}
    for (top, sub), mod_counts in group_modules.items():
        total = sum(mod_counts.values())
        best  = mod_counts.most_common(3)
        dom_frac = best[0][1] / total if total else 0
        ent   = _entropy(list(mod_counts.values()))
        max_e = math.log2(len(mod_counts)) if len(mod_counts) > 1 else 1.0
        norm_e = ent / max_e if max_e > 0 else 0.0
        labels[f"{top}_{sub}"] = {
            "top_cluster": top,
            "sub_cluster": sub,
            "label": best[0][0] if dom_frac >= 0.4 else " + ".join(m for m, _ in best[:3]),
            "dominant_fraction": round(dom_frac, 3),
            "norm_entropy": round(norm_e, 3),
            "size": total,
            "top_modules": [(m, c) for m, c in best],
        }
    return labels


def save(sub_map: dict, labels: dict) -> None:
    with open(SUB_FILE, "w") as f:
        json.dump(sub_map, f)
    with open(SUB_LABELS, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Sub-cluster map  → {SUB_FILE}")
    print(f"Sub-cluster labels → {SUB_LABELS}")


if __name__ == "__main__":
    from cluster import load as load_clusters

    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cluster_map = load_clusters()

    print("Re-clustering inside top clusters …")
    sub_map = recluster(G, cluster_map)
    labels  = label_sub_clusters(sub_map, G)
    save(sub_map, labels)

    print(f"\nSub-cluster label sample:")
    for key, info in sorted(labels.items(), key=lambda x: -x[1]["size"])[:20]:
        print(f"  {key:>8}  {info['label']:<30}  size={info['size']:>5}  "
              f"dom={info['dominant_fraction']:.0%}  H={info['norm_entropy']:.2f}")
