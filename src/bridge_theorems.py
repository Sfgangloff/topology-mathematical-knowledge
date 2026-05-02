"""
Step 1: Bridge theorems via cross-cluster premises.

A theorem T is a "bridge theorem" if its proof uses premises from ≥ 2 distinct
"pure" clusters (clusters where a single Mathlib module dominates ≥ PURITY_THRESHOLD).

Bridge score of T = number of distinct pure clusters spanned by T's premises.

Outputs:
  results/bridge_theorems.json  — ranked list of bridge theorems
  results/bridge_theorem_report.txt — human-readable report
"""

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

PURITY_THRESHOLD = 0.50   # cluster is "pure" if dominant module covers ≥ 50 %


def find_bridge_theorems(
    G,                            # nx.DiGraph — theorem dependency graph
    cluster_map: dict[str, int],
    labels: dict[int, dict],
    min_pure_clusters: int = 2,
    top_k: int = 500,
) -> list[dict]:
    """Find theorems whose proofs draw on premises from multiple distinct pure clusters.

    A "pure" cluster has a dominant Mathlib module covering ≥ PURITY_THRESHOLD of its
    nodes. A theorem is a "bridge" if it directly depends (via graph edges) on premises
    from ≥ min_pure_clusters distinct pure clusters, meaning its proof literally imports
    results from multiple mathematical domains.

    This detects "operational bridges" — theorems that are mechanically cross-domain —
    as opposed to "conceptual bridges" (structurally similar proofs without shared premises).

    Returns a list of result dicts sorted descending by n_premise_clusters, truncated to top_k.
    """
    # Identify pure clusters
    pure_clusters = {
        cid for cid, info in labels.items()
        if not info["is_mixed"] and info["dominant_fraction"] >= PURITY_THRESHOLD
    }
    print(f"Pure clusters (dom ≥ {PURITY_THRESHOLD:.0%}): {len(pure_clusters)} / {len(labels)}")

    # For each theorem, look at its direct successors (premises) in the dependency graph
    results = []
    for node in G.nodes():
        cid_self = cluster_map.get(node)
        if cid_self is None:
            continue

        # Premises = nodes that this theorem directly depends on
        premises = list(G.successors(node))
        if not premises:
            continue

        # Cluster of each premise
        premise_clusters = {
            cluster_map[p] for p in premises
            if p in cluster_map and cluster_map[p] in pure_clusters
        }

        if len(premise_clusters) < min_pure_clusters:
            continue

        # Label each spanned cluster
        spanned_labels = sorted(
            labels[c]["label"] for c in premise_clusters if c in labels
        )

        results.append({
            "theorem": node,
            "self_cluster": cid_self,
            "self_label": labels.get(cid_self, {}).get("label", "?"),
            "n_premise_clusters": len(premise_clusters),
            "premise_cluster_ids": sorted(premise_clusters),
            "premise_cluster_labels": spanned_labels,
            "n_premises": len(premises),
            "module": G.nodes[node].get("module", ""),
        })

    results.sort(key=lambda x: -x["n_premise_clusters"])
    return results[:top_k]


def save(bridge_theorems: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bridge_theorems.json", "w") as f:
        json.dump(bridge_theorems, f, indent=2)


def report(bridge_theorems: list[dict], labels: dict, top: int = 40) -> None:
    print(f"\nTop {top} bridge theorems (by #pure clusters spanned by premises):\n")
    print(f"{'Rank':>4}  {'Theorem':<55}  {'#Cl':>4}  {'Bridges to'}")
    print("-" * 110)
    for i, b in enumerate(bridge_theorems[:top]):
        name = b["theorem"]
        if len(name) > 54:
            name = name[:51] + "..."
        bridges_to = ", ".join(b["premise_cluster_labels"])
        if len(bridges_to) > 50:
            bridges_to = bridges_to[:47] + "..."
        print(f"{i+1:>4}  {name:<55}  {b['n_premise_clusters']:>4}  {bridges_to}")

    # Summary by cluster-pair
    print(f"\nMost common bridge cluster-pairs:")
    pair_counts: dict[tuple, int] = defaultdict(int)
    for b in bridge_theorems:
        ids = b["premise_cluster_ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (ids[i], ids[j])
                pair_counts[pair] += 1
    for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
        l1 = labels.get(pair[0], {}).get("label", str(pair[0]))
        l2 = labels.get(pair[1], {}).get("label", str(pair[1]))
        print(f"  {cnt:>4}  {l1}  ↔  {l2}")


if __name__ == "__main__":
    import pickle
    from cluster import load as load_clusters
    from label_clusters import load as load_labels

    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cluster_map = load_clusters()
    labels = load_labels()

    bridge_theorems = find_bridge_theorems(G, cluster_map, labels)
    print(f"\nFound {len(bridge_theorems)} bridge theorems")
    save(bridge_theorems)
    report(bridge_theorems, labels)

    # Write a text report too
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bridge_theorem_report.txt", "w") as f:
        for b in bridge_theorems[:200]:
            f.write(f"{b['n_premise_clusters']:>3}  {b['theorem']}\n")
            f.write(f"     bridges: {', '.join(b['premise_cluster_labels'])}\n\n")
    print(f"Report written → results/bridge_theorem_report.txt")
