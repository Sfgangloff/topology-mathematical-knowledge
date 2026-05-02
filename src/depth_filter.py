"""
Depth filter for bridge theorems.

A bridge theorem T uses premises from ≥2 clusters.  But some of those cross-cluster
premises are just widely-used utility lemmas (high in-degree in the dependency graph),
not genuine mathematical results from another domain.

IDEA: require that at least one cross-cluster premise has in-degree ≤ θ
(i.e., is a "specialised" node, not a utility hub).

We find the threshold θ automatically by sweeping and looking for the elbow in the
"number of surviving bridge theorems" vs θ curve.

Outputs:
  results/indegree_sweep.json + results/indegree_sweep.png
  results/bridge_theorems_filtered.json
  results/bridge_theorems_filtered_report.txt
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── 1. In-degree distribution & threshold sweep ───────────────────────────────

def indegree_sweep(G, cluster_map: dict, labels: dict, pure_threshold: float = 0.50):
    """Count how many bridge theorems survive at each in-degree threshold θ.

    A theorem survives at threshold θ if at least one of its cross-cluster premises
    has in-degree ≤ θ, meaning it is a "specialized" node — not a widely-used utility
    lemma. Sweeping θ from 0 to the 90th percentile of all in-degrees produces the
    survival curve used to find the elbow (automatic threshold selection).

    Returns: (thresholds, counts, theorem_bridges) where theorem_bridges is the full
    list of all raw bridge candidates before any threshold is applied.
    """
    pure_clusters = {
        cid for cid, info in labels.items()
        if not info["is_mixed"] and info["dominant_fraction"] >= pure_threshold
    }

    # Precompute: for each node, which cluster it is in and its in-degree
    in_deg = dict(G.in_degree())

    # For each theorem, find its cross-cluster premises with their in-degrees
    theorem_bridges = []
    for node in G.nodes():
        cid_self = cluster_map.get(node)
        if cid_self is None:
            continue
        premises = list(G.successors(node))
        if not premises:
            continue
        cross = [
            (p, cluster_map[p], in_deg.get(p, 0))
            for p in premises
            if p in cluster_map and cluster_map[p] in pure_clusters
               and cluster_map[p] != cid_self
        ]
        if not cross:
            continue
        # Distinct pure clusters reachable via any premise
        clusters_reached = {c for _, c, _ in cross}
        if len(clusters_reached) < 2:
            continue
        min_indeg_among_cross = min(d for _, _, d in cross)
        theorem_bridges.append({
            "theorem": node,
            "n_clusters": len(clusters_reached),
            "min_cross_indegree": min_indeg_among_cross,
            "cross_premises": [(p, c, d) for p, c, d in cross],
        })

    if not theorem_bridges:
        return [], []

    # Sweep θ from 0 to 90th percentile of in-degrees
    all_indeg = sorted(in_deg.values())
    max_theta = int(np.percentile(all_indeg, 90))
    thresholds = list(range(0, max_theta + 1, max(1, max_theta // 100)))

    counts = []
    for theta in thresholds:
        n = sum(1 for b in theorem_bridges if b["min_cross_indegree"] <= theta)
        counts.append(n)

    return thresholds, counts, theorem_bridges


def find_elbow(thresholds: list[int], counts: list[int]) -> int:
    """
    Find the elbow of the threshold-count curve using the maximum curvature method.
    Returns the threshold at the elbow point.
    """
    if len(thresholds) < 3:
        return thresholds[-1] if thresholds else 0

    xs = np.array(thresholds, dtype=float)
    ys = np.array(counts, dtype=float)

    # Normalise to [0,1]
    xs_n = (xs - xs.min()) / (xs.ptp() + 1e-9)
    ys_n = (ys - ys.min()) / (ys.ptp() + 1e-9)

    # Distance from each point to the line connecting first and last point
    p1 = np.array([xs_n[0], ys_n[0]])
    p2 = np.array([xs_n[-1], ys_n[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)

    dists = []
    for i in range(len(thresholds)):
        pt = np.array([xs_n[i], ys_n[i]])
        d = np.abs(np.cross(line, p1 - pt)) / (line_len + 1e-9)
        dists.append(d)

    elbow_idx = int(np.argmax(dists))
    return int(thresholds[elbow_idx])


def plot_sweep(thresholds: list[int], counts: list[int], elbow: int) -> None:
    """Save a two-panel plot: bridge count vs. theta, and log-log in-degree distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(thresholds, counts, "o-", markersize=3)
    ax.axvline(elbow, color="red", linestyle="--", label=f"Elbow θ={elbow}")
    ax.set_xlabel("Max in-degree of cross-cluster premise (θ)")
    ax.set_ylabel("# Bridge theorems surviving")
    ax.set_title("Bridge theorem count vs. premise depth threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # In-degree distribution (log-log)
    import pickle
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G_tmp = pickle.load(f)
    in_degs = sorted(G_tmp.in_degree(), key=lambda x: -x[1])
    deg_vals = [d for _, d in in_degs if d > 0]
    from collections import Counter
    deg_count = Counter(deg_vals)
    xs = sorted(deg_count.keys())
    ys = [deg_count[x] for x in xs]

    ax = axes[1]
    ax.loglog(xs, ys, ".", markersize=3, alpha=0.6)
    ax.axvline(elbow, color="red", linestyle="--", label=f"θ={elbow}")
    ax.set_xlabel("In-degree (log scale)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("In-degree distribution of theorem graph")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = RESULTS_DIR / "indegree_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"Sweep plot → {out}")
    plt.close(fig)


# ── 2. Filtered bridge theorems ───────────────────────────────────────────────

def filter_bridges(theorem_bridges: list[dict], theta: int, labels: dict) -> list[dict]:
    """Keep only bridge theorems whose minimum cross-cluster premise in-degree is ≤ theta.

    Low in-degree premises are "specialized" nodes introduced for a specific purpose —
    filtering them out removes bridges where the cross-cluster dependency is just a
    widely-used utility lemma (high in-degree = used everywhere).
    """
    filtered = [
        b for b in theorem_bridges
        if b["min_cross_indegree"] <= theta
    ]
    filtered.sort(key=lambda x: (x["n_clusters"], -x["min_cross_indegree"]), reverse=True)
    return filtered


def report_filtered(filtered: list[dict], labels: dict, G, top: int = 40) -> None:
    print(f"\nFiltered bridge theorems (min cross-cluster premise in-degree ≤ θ):\n")
    print(f"{'Rk':>4}  {'Theorem':<55}  {'#Cl':>4}  {'minInDeg':>8}  {'Bridges to'}")
    print("-" * 120)
    for i, b in enumerate(filtered[:top]):
        name = b["theorem"]
        if len(name) > 54:
            name = name[:51] + "..."
        bridge_labels = sorted(set(
            labels.get(c, {}).get("label", str(c))
            for _, c, _ in b["cross_premises"]
        ))
        bridges_str = ", ".join(bridge_labels)
        if len(bridges_str) > 40:
            bridges_str = bridges_str[:37] + "..."
        print(f"{i+1:>4}  {name:<55}  {b['n_clusters']:>4}  {b['min_cross_indegree']:>8}  {bridges_str}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from cluster import load as load_clusters
    from label_clusters import load as load_labels

    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cluster_map = load_clusters()
    labels = load_labels()

    print("Computing in-degree sweep …")
    thresholds, counts, theorem_bridges = indegree_sweep(G, cluster_map, labels)

    print(f"Total bridge theorems before filter: {len(theorem_bridges)}")
    print(f"In-degree range of cross-cluster premises: "
          f"0 – {max(b['min_cross_indegree'] for b in theorem_bridges)}")

    theta = find_elbow(thresholds, counts)
    print(f"Elbow threshold θ = {theta}")

    # Save sweep
    sweep_data = [{"theta": t, "count": c} for t, c in zip(thresholds, counts)]
    with open(RESULTS_DIR / "indegree_sweep.json", "w") as f:
        json.dump(sweep_data, f)

    plot_sweep(thresholds, counts, theta)

    filtered = filter_bridges(theorem_bridges, theta, labels)
    print(f"Bridge theorems after filter (θ={theta}): {len(filtered)}")

    report_filtered(filtered, labels, G, top=40)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    save_data = []
    for b in filtered[:200]:
        save_data.append({
            "theorem": b["theorem"],
            "n_clusters": b["n_clusters"],
            "min_cross_indegree": b["min_cross_indegree"],
            "cross_cluster_premises": [
                {"premise": p,
                 "cluster_label": labels.get(c, {}).get("label", str(c)),
                 "in_degree": d}
                for p, c, d in b["cross_premises"]
                if d <= theta
            ][:5],
        })
    with open(RESULTS_DIR / "bridge_theorems_filtered.json", "w") as f:
        json.dump(save_data, f, indent=2)

    with open(RESULTS_DIR / "bridge_theorems_filtered_report.txt", "w") as f:
        for b in filtered[:200]:
            f.write(f"{'=' * 80}\n")
            f.write(f"{b['theorem']}\n")
            f.write(f"  #clusters={b['n_clusters']}  min_indegree={b['min_cross_indegree']}\n")
            f.write(f"  Non-trivial cross-cluster premises (in_deg ≤ {theta}):\n")
            for p, c, d in sorted(b["cross_premises"], key=lambda x: x[2]):
                if d <= theta:
                    lab = labels.get(c, {}).get("label", str(c))
                    f.write(f"    [{lab}] {p}  (in_deg={d})\n")
            f.write("\n")
    print(f"Reports saved to results/bridge_theorems_filtered.json / _report.txt")
