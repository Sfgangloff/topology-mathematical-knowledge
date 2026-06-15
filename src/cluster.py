"""
Cluster the theorem dependency graph using the Leiden algorithm
(better resolution and stability than Louvain).

Returns a dict: theorem_name → cluster_id
Also saves a summary of cluster sizes and their dominant Mathlib modules.
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

DATA_DIR = Path(__file__).parent.parent / "data"
CLUSTER_FILE = DATA_DIR / "clusters.json"
CLUSTER_SUMMARY_FILE = DATA_DIR / "cluster_summary.json"


def _nx_to_igraph(G_nx: nx.Graph) -> tuple[ig.Graph, list]:
    """Convert an undirected NetworkX graph to igraph, return (G_ig, node_list)."""
    nodes = list(G_nx.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_nx.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges)
    return G_ig, nodes


def cluster(G: nx.DiGraph, resolution: float = 1.0) -> dict[str, int]:
    """Run Leiden community detection on the undirected theorem dependency graph.

    Only the largest connected component (LCC) is clustered; isolated nodes
    are omitted. The undirected view is used because edge direction (A proves B)
    is less informative than co-citation structure for finding mathematical domains.

    Returns:
        cluster_map: {theorem_name: cluster_id} for all nodes in the LCC.
    """
    # Work on the undirected version (dependency direction not needed for topology)
    G_und = G.to_undirected()

    # Keep only the largest connected component to avoid noise
    lcc = max(nx.connected_components(G_und), key=len)
    print(f"Largest connected component: {len(lcc):,} / {G_und.number_of_nodes():,} nodes")
    G_lcc = G_und.subgraph(lcc).copy()

    G_ig, nodes = _nx_to_igraph(G_lcc)

    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.ModularityVertexPartition,
        n_iterations=10,
        seed=42,
    )

    cluster_map = {nodes[i]: partition.membership[i] for i in range(len(nodes))}
    print(f"Found {len(set(cluster_map.values()))} clusters.")
    return cluster_map


def summarise(cluster_map: dict[str, int], G: nx.DiGraph) -> dict:
    """Build a size-and-module summary for each cluster.

    Returns a dict {cluster_id: {size, top_modules}} where top_modules is the
    five most frequent top-level Mathlib sub-modules (e.g. 'Topology', 'Algebra')
    among the cluster's members.
    """
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for name, cid in cluster_map.items():
        cluster_members[cid].append(name)

    summary = {}
    for cid, members in sorted(cluster_members.items(), key=lambda x: -len(x[1])):
        modules = [G.nodes[m].get("module", "") for m in members if G.has_node(m)]
        # Top-level Mathlib module = second component after "Mathlib."
        top_modules = []
        for mod in modules:
            parts = mod.split(".")
            if len(parts) >= 2:
                top_modules.append(parts[1])
        top = Counter(top_modules).most_common(5)
        summary[cid] = {
            "size": len(members),
            "top_modules": top,
        }
    return summary


def save(cluster_map: dict, summary: dict):
    """Persist the cluster assignment and summary dicts to data/ as JSON."""
    with open(CLUSTER_FILE, "w") as f:
        json.dump({str(k): v for k, v in cluster_map.items()}, f)
    with open(CLUSTER_SUMMARY_FILE, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f)
    print(f"Clusters saved to {CLUSTER_FILE}")


def load() -> dict[str, int]:
    """Load the cluster assignment from the data/ cache. Returns {theorem_name: cluster_id}."""
    with open(CLUSTER_FILE) as f:
        return {k: int(v) for k, v in json.load(f).items()}


if __name__ == "__main__":
    from build_graph import load as load_graph
    G, _ = load_graph()
    cluster_map = cluster(G)
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G_full = pickle.load(f)
    summary = summarise(cluster_map, G_full)
    save(cluster_map, summary)

    print("\nTop clusters:")
    for cid, info in sorted(summary.items(), key=lambda x: -x[1]["size"])[:10]:
        print(f"  Cluster {cid:>3}: size={info['size']:>5}  top modules={info['top_modules']}")
