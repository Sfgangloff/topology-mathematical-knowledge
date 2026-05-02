"""
Visualizations:

1. Cluster map — nodes=clusters, sized by cluster size, coloured by label,
   edges between clusters that share bridge n-grams (edge weight = bridge score sum).

2. N-gram sweep — already produced by ngram_analysis.py; re-plotted here for
   consistency.

3. Hierarchical cluster map — same layout but nodes split into sub-clusters,
   grouped by top-level cluster.

4. Entropy vs. dominant-fraction scatter — helps understand which clusters are
   "pure" domain clusters vs. mixed.

All saved to results/*.png
"""

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Colour palette for top Mathlib modules
MODULE_COLORS = {
    "Algebra":          "#4C72B0",
    "Analysis":         "#DD8452",
    "CategoryTheory":   "#55A868",
    "Combinatorics":    "#C44E52",
    "Data":             "#8172B3",
    "FieldTheory":      "#937860",
    "Geometry":         "#DA8BC3",
    "GroupTheory":      "#8C8C8C",
    "LinearAlgebra":    "#CCB974",
    "MeasureTheory":    "#64B5CD",
    "ModelTheory":      "#e377c2",
    "NumberTheory":     "#7f7f7f",
    "Order":            "#bcbd22",
    "RingTheory":       "#17becf",
    "Topology":         "#aec7e8",
    "AlgebraicGeometry":"#ffbb78",
    "AlgebraicTopology":"#98df8a",
    "Computability":    "#ff9896",
    "External":         "#c5b0d5",
    "Unknown":          "#c7c7c7",
}
DEFAULT_COLOR = "#c7c7c7"


def _module_color(label: str) -> str:
    """Return the canonical color for a cluster label (uses the first-listed module for mixed labels)."""
    # label may be "Topology" or "Topology + Algebra + ..."
    first = label.split(" + ")[0].strip()
    return MODULE_COLORS.get(first, DEFAULT_COLOR)


# ── 1. Cluster bridge graph ───────────────────────────────────────────────────

def plot_cluster_graph(labels: dict[int, dict], bridges: list[dict]) -> None:
    """Save a spring-layout graph of clusters connected by bridge n-gram activity.

    Nodes are sized by cluster size (√size) and colored by dominant Mathlib module.
    Edge weight = sum of bridge scores for n-grams shared between the two clusters.
    Only clusters with a label entry are shown; cluster labels are shown for clusters
    with size > 300. Saved to results/cluster_graph.png.
    """
    # Build a weighted graph between clusters based on bridges
    G = nx.Graph()
    for cid, info in labels.items():
        G.add_node(cid, **info)

    for b in bridges:
        cluster_ids = [int(k) for k in b["cluster_counts"]]
        w = b["bridge_score"]
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                u, v = cluster_ids[i], cluster_ids[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)

    # Keep only clusters that appear in the cluster_map (have label info)
    G = G.subgraph([n for n in G.nodes if n in labels]).copy()

    # Layout
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5)

    node_sizes  = [math.sqrt(labels[n]["size"]) * 8 for n in G.nodes]
    node_colors = [_module_color(labels[n]["label"]) for n in G.nodes]
    edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    if len(edge_weights):
        ew_norm = 0.3 + 2.5 * (edge_weights - edge_weights.min()) / (edge_weights.ptp() + 1e-12)
    else:
        ew_norm = []

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, ax=ax, width=list(ew_norm), alpha=0.35, edge_color="#888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85)

    # Label only clusters large enough to be readable
    big_nodes = {n: labels[n]["label"] for n in G.nodes if labels[n]["size"] > 300}
    nx.draw_networkx_labels(G, pos, labels=big_nodes, ax=ax, font_size=6)

    # Legend
    seen_mods = {}
    for n in G.nodes:
        first = labels[n]["label"].split(" + ")[0].strip()
        seen_mods[first] = _module_color(labels[n]["label"])
    patches = [mpatches.Patch(color=c, label=m) for m, c in sorted(seen_mods.items())]
    ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2,
              title="Dominant module", title_fontsize=8)

    ax.set_title("Cluster graph — nodes sized by cluster size, edges by bridge score")
    ax.axis("off")
    fig.tight_layout()
    out = RESULTS_DIR / "cluster_graph.png"
    fig.savefig(out, dpi=150)
    print(f"Cluster graph → {out}")
    plt.close(fig)


# ── 2. Entropy scatter ────────────────────────────────────────────────────────

def plot_entropy_scatter(labels: dict[int, dict]) -> None:
    """Save a scatter of dominant_fraction vs. normalised_entropy for all clusters.

    Reveals the size-purity anticorrelation: large clusters are mixed (high entropy),
    small clusters are pure (low entropy). The red dashed line at x=0.4 marks the
    mixed/pure boundary. Saved to results/cluster_purity.png.
    """
    dom_fracs  = [info["dominant_fraction"] for info in labels.values()]
    norm_ents  = [info["norm_entropy"] for info in labels.values()]
    sizes      = [info["size"] for info in labels.values()]
    colors     = [_module_color(info["label"]) for info in labels.values()]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(dom_fracs, norm_ents,
                    s=[math.sqrt(s) * 5 for s in sizes],
                    c=colors, alpha=0.7, edgecolors="k", linewidths=0.3)

    # Annotate large clusters
    for cid, info in labels.items():
        if info["size"] > 1000:
            ax.annotate(info["label"].split(" + ")[0], (info["dominant_fraction"], info["norm_entropy"]),
                        fontsize=6, ha="center")

    ax.axvline(0.4, color="red", linestyle="--", alpha=0.5, label="40% threshold (mixed/pure)")
    ax.set_xlabel("Dominant module fraction")
    ax.set_ylabel("Normalised entropy")
    ax.set_title("Cluster purity: dominant fraction vs. module entropy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "cluster_purity.png"
    fig.savefig(out, dpi=150)
    print(f"Cluster purity scatter → {out}")
    plt.close(fig)


# ── 3. Hierarchical map ───────────────────────────────────────────────────────

def plot_hierarchical(sub_labels: dict) -> None:
    """Save a stacked bar chart of sub-cluster module composition for the top 15 clusters.

    Each row is one top-level cluster; each segment is a sub-cluster coloured by its
    dominant Mathlib module. Width = fraction of the top-level cluster's nodes.
    Saved to results/hierarchical_clusters.png.
    """
    from collections import defaultdict

    top_groups: dict[int, list] = defaultdict(list)
    for key, info in sub_labels.items():
        if info["sub_cluster"] < 0:
            continue
        top_groups[info["top_cluster"]].append(info)

    # Sort top clusters by total size
    top_order = sorted(top_groups.keys(),
                       key=lambda c: sum(i["size"] for i in top_groups[c]),
                       reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(14, 8))
    y = 0
    yticks, ylabels = [], []
    bar_height = 0.7

    for tc in top_order:
        subs = sorted(top_groups[tc], key=lambda x: -x["size"])
        total = sum(s["size"] for s in subs)
        x = 0
        for sub in subs:
            frac = sub["size"] / total
            color = _module_color(sub["label"])
            ax.barh(y, frac, left=x, height=bar_height, color=color,
                    edgecolor="white", linewidth=0.4)
            if frac > 0.08:
                ax.text(x + frac / 2, y, sub["label"].split(" + ")[0],
                        ha="center", va="center", fontsize=5.5, color="white",
                        fontweight="bold")
            x += frac
        yticks.append(y)
        ylabels.append(f"C{tc}  ({total})")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Fraction of cluster")
    ax.set_title("Hierarchical sub-clustering of top-level clusters")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = RESULTS_DIR / "hierarchical_clusters.png"
    fig.savefig(out, dpi=150)
    print(f"Hierarchical map → {out}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    with open(DATA_DIR / "cluster_labels.json") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    with open(RESULTS_DIR / "bridges.json") as f:
        bridges = json.load(f)

    plot_cluster_graph(labels, bridges)
    plot_entropy_scatter(labels)

    sub_label_path = DATA_DIR / "sub_cluster_labels.json"
    if sub_label_path.exists():
        with open(sub_label_path) as f:
            sub_labels = json.load(f)
        plot_hierarchical(sub_labels)
    else:
        print("Sub-cluster labels not found — skipping hierarchical plot (run hierarchical.py first).")


if __name__ == "__main__":
    main()
