"""
Tag each cluster with a semantic label derived from the dominant Mathlib module.
Clusters where no module is dominant are flagged as "mixed" with entropy score.

Output: data/cluster_labels.json  {cluster_id: {label, entropy, top_modules, is_mixed}}
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
LABELS_FILE = DATA_DIR / "cluster_labels.json"


def _entropy(counts: list[int]) -> float:
    """Shannon entropy (bits) of a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def _top_module(file_path: str) -> str:
    """Extract top-level Mathlib sub-module (e.g. 'Topology', 'Algebra')."""
    parts = Path(file_path).with_suffix("").parts
    try:
        idx = next(i for i, s in enumerate(parts) if s == "Mathlib")
        return parts[idx + 1] if idx + 1 < len(parts) else "Mathlib"
    except StopIteration:
        # Non-Mathlib (Init, Batteries, lake packages…)
        for p in parts:
            if p in ("Init", "Batteries", "Std", "lake"):
                return p
        return "External"


def label_clusters(cluster_map: dict[str, int], G) -> dict[int, dict]:
    """Assign a semantic label to each cluster based on its dominant Mathlib module.

    A cluster is labelled "mixed" if no single module accounts for ≥ 40% of its
    members. Mixed clusters are candidate bridge zones — they sit at intersections
    of multiple mathematical domains.

    Returns a dict {cluster_id: {label, dominant_module, dominant_fraction,
    entropy, norm_entropy, is_mixed, size, top_modules}}.
    """
    # Aggregate module distribution per cluster
    cluster_modules: dict[int, Counter] = defaultdict(Counter)
    for node, cid in cluster_map.items():
        fp = G.nodes[node].get("file_path", "") if G.has_node(node) else ""
        mod = _top_module(fp) if fp else "Unknown"
        cluster_modules[cid][mod] += 1

    labels = {}
    for cid, mod_counts in cluster_modules.items():
        total = sum(mod_counts.values())
        top = mod_counts.most_common(5)
        dom_count = top[0][1] if top else 0
        dom_frac = dom_count / total if total else 0
        ent = _entropy(list(mod_counts.values()))
        max_ent = math.log2(len(mod_counts)) if len(mod_counts) > 1 else 1.0
        norm_ent = ent / max_ent if max_ent > 0 else 0.0

        is_mixed = dom_frac < 0.40  # dominant module < 40 % → mixed

        if is_mixed:
            label = " + ".join(m for m, _ in top[:3])
        else:
            label = top[0][0] if top else "Unknown"

        labels[cid] = {
            "label": label,
            "dominant_module": top[0][0] if top else "Unknown",
            "dominant_fraction": round(dom_frac, 3),
            "entropy": round(ent, 3),
            "norm_entropy": round(norm_ent, 3),
            "is_mixed": is_mixed,
            "size": total,
            "top_modules": [(m, c) for m, c in top],
        }
    return labels


def save(labels: dict) -> None:
    with open(LABELS_FILE, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)
    print(f"Labels saved → {LABELS_FILE}")


def load() -> dict[int, dict]:
    with open(LABELS_FILE) as f:
        return {int(k): v for k, v in json.load(f).items()}


def report(labels: dict) -> None:
    """Print a human-readable summary table of cluster labels, sorted by size."""
    mixed = [x for x in labels.values() if x["is_mixed"]]
    clean = [x for x in labels.values() if not x["is_mixed"]]
    print(f"\nCluster label report  ({len(labels)} total)")
    print(f"  Cleanly labelled : {len(clean)}")
    print(f"  Mixed/ambiguous  : {len(mixed)}")
    print(f"\n{'CID':>4}  {'Label':<38}  {'Dom%':>5}  {'Ent':>5}  {'Size':>6}  Mixed?")
    print("-" * 75)
    for cid, info in sorted(labels.items(), key=lambda x: -x[1]["size"])[:30]:
        marker = "  *" if info["is_mixed"] else ""
        print(
            f"{cid:>4}  {info['label']:<38}  "
            f"{info['dominant_fraction']:>5.1%}  "
            f"{info['norm_entropy']:>5.2f}  "
            f"{info['size']:>6}{marker}"
        )
    if mixed:
        print(f"\n  (* = mixed cluster — dominant module covers < 40 % of nodes)")


if __name__ == "__main__":
    import pickle
    from cluster import load as load_clusters

    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cluster_map = load_clusters()
    labels = label_clusters(cluster_map, G)
    save(labels)
    report(labels)
