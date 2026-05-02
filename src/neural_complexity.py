"""
Tononi-Edelman Neural Complexity (CN) on the cluster interaction graph.

Reference: Tononi & Edelman (1998) "Consciousness and Complexity", Science 282.

CN(X) = Σ_{k=1}^{n-1} [ <H(X_k)> - H(X | X \ X_k) ]

where X_k is a subset of k clusters, H is entropy, and <·> is the average
over all subsets of size k.  This is expensive for large n; we use an
approximation based on the cluster interaction weight matrix.

APPROXIMATION used here (tractable for n ≤ ~50 clusters):
  - Build a weighted adjacency matrix W where W[i,j] = number of bridge theorems
    linking cluster i and cluster j (from step 1 results)
  - Treat each row of the normalised W as a "probability distribution" over interactions
  - Integration of cluster i: I_i = H(row_i) — entropy of how cluster i distributes
    its connections across other clusters (high = connects broadly)
  - Differentiation of cluster i: D_i = 1 - H(col_i)/log(n) — how specialised the
    cluster's INCOMING connections are
  - Neural complexity proxy: CN_i = I_i * D_i for each cluster
    (high CN = broad outgoing connections AND specialised incoming = true bridge role)

Also computes the global CN as the average over all pairs:
  CN_global = Σ_{i<j} W[i,j] * |H(X_i) - H(X_j)| / Z
  (how much information is shared across pairs, weighted by connection strength)

Outputs:
  results/neural_complexity.json
  results/neural_complexity.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def compute_neural_complexity(
    bridge_theorems: list[dict],
    labels: dict[int, dict],
    cluster_map: dict[str, int],
) -> dict:
    """Compute a Tononi-Edelman neural complexity proxy for each cluster.

    Builds a weighted adjacency matrix W where W[i,j] = number of bridge theorems
    linking clusters i and j. Then for each cluster:

      integration  = H(outgoing connection distribution)  — high → connects to many clusters
      differentiation = 1 - H_norm(module distribution)  — high → specialised proof identity
      neural_complexity = integration × differentiation   — high → broad AND specialised

    A high-CN cluster is both a well-connected hub AND maintains its own distinct
    proof culture (e.g., SetTheory). A low-CN cluster either connects nowhere (isolated
    specialist) or connects everywhere but looks the same as its neighbours (generic hub).

    Also computes a global CN as weighted average of |H_i - H_j| across bridge pairs.

    Returns: cluster_metrics dict, global_cn float, weight matrix W, cluster_ids list.
    """
    # Build cluster interaction weight matrix from bridge theorem cross-cluster pairs
    cluster_ids = sorted(labels.keys())
    n = len(cluster_ids)
    idx = {c: i for i, c in enumerate(cluster_ids)}

    # W[i,j] = number of bridge theorems connecting cluster i and cluster j
    W = np.zeros((n, n))
    for b in bridge_theorems:
        cids = sorted(set(
            c for _, c, _ in b.get("cross_premises", [])
        ))
        self_cid = cluster_map.get(b["theorem"])
        if self_cid is not None and self_cid in idx:
            for c in cids:
                if c in idx and c != self_cid:
                    i, j = idx[self_cid], idx[c]
                    W[i, j] += 1
                    W[j, i] += 1

    # Normalise rows (for entropy calculation)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums

    # Per-cluster metrics
    cluster_metrics = {}
    for cid in cluster_ids:
        i = idx[cid]
        info = labels[cid]

        # Integration: entropy of outgoing connection distribution
        integration = _entropy(W_norm[i])

        # Differentiation: how specialised is this cluster's tactic vocabulary?
        # Proxy: 1 - normalised entropy of the cluster's module distribution
        top_mods = info.get("top_modules", [])
        if top_mods:
            counts = np.array([c for _, c in top_mods], dtype=float)
            counts /= counts.sum()
            diff = 1.0 - _entropy(counts) / math.log2(len(counts) + 1)
        else:
            diff = 0.0

        # Bridge activity: total connections
        total_connections = int(W[i].sum())

        # Neural complexity proxy for this cluster
        cn = integration * diff

        cluster_metrics[cid] = {
            "label": info["label"],
            "size": info["size"],
            "integration": round(integration, 4),
            "differentiation": round(diff, 4),
            "neural_complexity": round(cn, 4),
            "total_bridge_connections": total_connections,
            "is_mixed": info["is_mixed"],
        }

    # Global CN: average mutual information across all connected pairs
    # Approximation: Σ_{i<j} W[i,j] * |H_i - H_j| / total_weight
    total_weight = W.sum() / 2
    H_vec = np.array([_entropy(W_norm[i]) for i in range(n)])
    global_cn = 0.0
    if total_weight > 0:
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0:
                    global_cn += W[i, j] * abs(H_vec[i] - H_vec[j])
        global_cn /= total_weight

    return cluster_metrics, float(global_cn), W, cluster_ids


def plot(cluster_metrics: dict, cluster_ids: list, W: np.ndarray, labels: dict) -> None:
    """Save a two-panel plot: integration vs. differentiation scatter, and top-20 CN bar chart."""
    # Sort by neural complexity
    sorted_metrics = sorted(cluster_metrics.items(), key=lambda x: -x[1]["neural_complexity"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter — integration vs differentiation, sized by bridge connections
    ax = axes[0]
    xs = [m["integration"] for _, m in sorted_metrics]
    ys = [m["differentiation"] for _, m in sorted_metrics]
    sz = [max(20, m["total_bridge_connections"] * 5) for _, m in sorted_metrics]
    colors = ["#e74c3c" if m["is_mixed"] else "#3498db" for _, m in sorted_metrics]
    ax.scatter(xs, ys, s=sz, c=colors, alpha=0.7, edgecolors="k", linewidths=0.3)

    # Label top-CN clusters
    for cid, m in sorted_metrics[:12]:
        ax.annotate(m["label"].split(" + ")[0], (m["integration"], m["differentiation"]),
                    fontsize=6, ha="center")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="Mixed cluster"),
                        Patch(color="#3498db", label="Pure cluster")],
              fontsize=8)
    ax.set_xlabel("Integration (entropy of bridge connections)")
    ax.set_ylabel("Differentiation (1 - normalised module entropy)")
    ax.set_title("Neural complexity: integration × differentiation")
    ax.grid(True, alpha=0.3)

    # Right: top-20 clusters by neural complexity
    ax = axes[1]
    top20 = sorted_metrics[:20]
    names = [m["label"].split(" + ")[0][:18] for _, m in top20]
    cn_vals = [m["neural_complexity"] for _, m in top20]
    colors_bar = ["#e74c3c" if m["is_mixed"] else "#3498db" for _, m in top20]
    bars = ax.barh(range(len(top20)), cn_vals, color=colors_bar, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Neural complexity (integration × differentiation)")
    ax.set_title("Top-20 clusters by neural complexity")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    out = RESULTS_DIR / "neural_complexity.png"
    fig.savefig(out, dpi=150)
    print(f"Neural complexity plot → {out}")
    plt.close(fig)


def report(cluster_metrics: dict, global_cn: float, top: int = 20) -> None:
    print(f"\nGlobal neural complexity CN = {global_cn:.4f}")
    print(f"\nTop {top} clusters by neural complexity:\n")
    print(f"{'CID':>4}  {'Label':<35}  {'Int':>6}  {'Diff':>6}  {'CN':>7}  {'BridgeCon':>9}")
    print("-" * 75)
    sorted_m = sorted(cluster_metrics.items(), key=lambda x: -x[1]["neural_complexity"])
    for cid, m in sorted_m[:top]:
        print(f"{cid:>4}  {m['label']:<35}  "
              f"{m['integration']:>6.3f}  {m['differentiation']:>6.3f}  "
              f"{m['neural_complexity']:>7.4f}  {m['total_bridge_connections']:>9}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from cluster import load as load_clusters
    from label_clusters import load as load_labels

    cluster_map = load_clusters()
    labels = load_labels()

    # Load bridge theorems (from step 1)
    bt_path = RESULTS_DIR / "bridge_theorems_filtered.json"
    if not bt_path.exists():
        bt_path = RESULTS_DIR / "bridge_theorems.json"
    with open(bt_path) as f:
        bridge_theorems_raw = json.load(f)

    # Reconstruct cross_premises from saved data
    bridge_theorems = []
    for b in bridge_theorems_raw:
        cross = [(item["premise"],
                  next((c for c, info in labels.items() if info["label"] == item.get("cluster_label", "")), -1),
                  item.get("in_degree", 0))
                 for item in b.get("cross_cluster_premises", [])]
        bridge_theorems.append({
            "theorem": b["theorem"],
            "n_clusters": b["n_clusters"],
            "cross_premises": cross,
        })

    cluster_metrics, global_cn, W, cluster_ids = compute_neural_complexity(
        bridge_theorems, labels, cluster_map
    )

    report(cluster_metrics, global_cn)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "neural_complexity.json", "w") as f:
        json.dump({
            "global_cn": global_cn,
            "clusters": {str(k): v for k, v in cluster_metrics.items()},
        }, f, indent=2)

    plot(cluster_metrics, cluster_ids, W, labels)
    print(f"\nResults saved to results/neural_complexity.json")
