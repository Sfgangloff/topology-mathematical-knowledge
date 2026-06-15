"""
Conceptual integrative hub: neural-complexity (CN) over the CONCEPTUAL
(proof-structure) bridge matrix, as a counterpart to the OPERATIONAL
(shared-premise) hub computed in src/neural_complexity.py (a-0010, SetTheory).

a-0010 found SetTheory is the operational integrative hub using the bridge
matrix W_op[i,j] = # of shared-premise bridge theorems linking clusters i,j.
a-0032 then showed operational and conceptual bridges are near-independent
signals. This asks the natural follow-up: under the CONCEPTUAL lens (proof-
structure similarity, src/edit_distance_bridges.py), which cluster is the
integrative hub? If it is NOT SetTheory, the two bridge classes also disagree
on which domain is most integrative -- a clean structural corollary of a-0032.

Method (identical CN proxy as neural_complexity.py, only the matrix changes):
  W_struct[i,j] = surprise(closest cross-cluster proof pair i,j)
                = max(0, baseline_ED - min_cross_ED)   (from edit_distance_bridges)
  integration_i    = H(row_i of normalized W_struct)        (broad structural reach)
  differentiation_i= 1 - H_norm(module distribution of i)   (specialized identity)
  CN_i             = integration_i * differentiation_i

Restricted to the 30 clusters the structural census scored (same scope as the
edit-distance analysis; this is a high-similarity sample, per a-0032's caveat).

Output: results/conceptual_complexity.json
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def compute(struct_pairs, labels):
    # Clusters scored by the structural census
    cluster_ids = sorted({p["cluster_a"] for p in struct_pairs} |
                         {p["cluster_b"] for p in struct_pairs})
    n = len(cluster_ids)
    idx = {c: i for i, c in enumerate(cluster_ids)}

    # Conceptual interaction matrix weighted by surprise (proof-structure similarity)
    W = np.zeros((n, n))
    for p in struct_pairs:
        a, b = idx[p["cluster_a"]], idx[p["cluster_b"]]
        w = max(0.0, float(p.get("surprise", 0.0)))
        W[a, b] += w
        W[b, a] += w

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums

    metrics = {}
    for cid in cluster_ids:
        i = idx[cid]
        info = labels[cid]
        integration = _entropy(W_norm[i])
        top_mods = info.get("top_modules", [])
        if top_mods:
            counts = np.array([c for _, c in top_mods], dtype=float)
            counts /= counts.sum()
            diff = 1.0 - _entropy(counts) / math.log2(len(counts) + 1)
        else:
            diff = 0.0
        cn = integration * diff
        metrics[cid] = {
            "label": info["label"],
            "size": info["size"],
            "integration": round(integration, 4),
            "differentiation": round(diff, 4),
            "neural_complexity": round(cn, 4),
            "total_struct_connections": int((W[i] > 0).sum()),
            "struct_weight": round(float(W[i].sum()), 4),
            "is_mixed": info["is_mixed"],
        }

    # Global CN: surprise-weighted average |H_i - H_j| over connected pairs
    total_w = W.sum() / 2
    H = np.array([_entropy(W_norm[i]) for i in range(n)])
    global_cn = 0.0
    if total_w > 0:
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0:
                    global_cn += W[i, j] * abs(H[i] - H[j])
        global_cn /= total_w
    return metrics, float(global_cn)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from label_clusters import load as load_labels

    labels = load_labels()
    struct_pairs = json.load(open(RESULTS_DIR / "edit_distance_bridges.json"))

    metrics, global_cn = compute(struct_pairs, labels)

    # Operational hub for comparison
    op = json.load(open(RESULTS_DIR / "neural_complexity.json"))
    op_top = sorted(op["clusters"].items(),
                    key=lambda x: -x[1]["neural_complexity"])[0]

    sorted_m = sorted(metrics.items(), key=lambda x: -x[1]["neural_complexity"])
    print(f"Conceptual global CN = {global_cn:.4f}   (operational = {op['global_cn']:.4f})")
    print(f"\nTop conceptual integrative hubs ({len(metrics)} clusters scored):")
    print(f"{'CID':>4}  {'Label':<35}  {'Int':>6}  {'Diff':>6}  {'CN':>7}  {'#conn':>5}")
    print("-" * 78)
    for cid, m in sorted_m[:10]:
        print(f"{cid:>4}  {m['label'][:35]:<35}  {m['integration']:>6.3f}  "
              f"{m['differentiation']:>6.3f}  {m['neural_complexity']:>7.4f}  "
              f"{m['total_struct_connections']:>5}")

    # SetTheory rank under conceptual lens
    setth = [(r, cid, m) for r, (cid, m) in enumerate(sorted_m, 1)
             if m["label"].startswith("SetTheory")]
    print(f"\nOperational hub: {op_top[1]['label']} (CN={op_top[1]['neural_complexity']})")
    conc_hub = sorted_m[0]
    print(f"Conceptual hub : {conc_hub[1]['label']} (CN={conc_hub[1]['neural_complexity']})")
    if setth:
        r, cid, m = setth[0]
        print(f"SetTheory under conceptual lens: rank {r}/{len(sorted_m)} "
              f"(CN={m['neural_complexity']}, int={m['integration']}, diff={m['differentiation']})")

    out = {
        "global_cn": global_cn,
        "operational_global_cn": op["global_cn"],
        "operational_hub": {"cid": op_top[0], "label": op_top[1]["label"],
                            "cn": op_top[1]["neural_complexity"]},
        "conceptual_hub": {"cid": conc_hub[0], "label": conc_hub[1]["label"],
                           "cn": conc_hub[1]["neural_complexity"]},
        "setth_conceptual_rank": (setth[0][0] if setth else None),
        "n_clusters_scored": len(metrics),
        "clusters": {str(k): v for k, v in metrics.items()},
    }
    with open(RESULTS_DIR / "conceptual_complexity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/conceptual_complexity.json")
