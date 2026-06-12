"""
Validate the hierarchical sub-clustering (a-0003 / e-0003) against Mathlib's
OWN module hierarchy.

a-0003 was inconclusive: re-clustering inside each top-level cluster produced
447 "plausible sub-domains", but they were never checked against Mathlib's
second-level module path (e.g. LinearAlgebra/Matrix, Topology/Algebra), the
natural ground truth for sub-domains.

This script supplies that check. Ground truth = the 2-level Mathlib module path
from each declaration's file_path. We ask: does splitting a top-level cluster
into sub-clusters recover FINER module structure than the top cluster already
captures? Metric: normalized mutual information (NMI), the same family e-0002
used at the top level (Li et al. NMI=0.34), plus sub-cluster purity at the
2-level granularity and a within-top-cluster permutation null.

Inputs : data/corpus.jsonl, data/sub_clusters.json
Output : results/subcluster_validation.json
CPU-only, no external ML deps (NMI implemented directly).
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _module(file_path: str, depth: int) -> str:
    """Mathlib module path truncated to `depth` components below 'Mathlib'.

    depth=1 -> 'LinearAlgebra'   (top-level, matches label_clusters._top_module)
    depth=2 -> 'LinearAlgebra/Matrix'
    Non-Mathlib declarations collapse to a coarse external bucket.
    """
    parts = Path(file_path).with_suffix("").parts
    try:
        idx = next(i for i, s in enumerate(parts) if s == "Mathlib")
    except StopIteration:
        for p in parts:
            if p in ("Init", "Batteries", "Std", "lake"):
                return p
        return "External"
    sub = parts[idx + 1 : idx + 1 + depth]
    return "/".join(sub) if sub else "Mathlib"


def _nmi(labels_a, labels_b) -> float:
    """Normalized mutual information (arithmetic-mean normalization) in [0,1]."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    ca, cb = Counter(labels_a), Counter(labels_b)
    joint = Counter(zip(labels_a, labels_b))

    def entropy(counter):
        return -sum((c / n) * math.log(c / n) for c in counter.values() if c > 0)

    h_a, h_b = entropy(ca), entropy(cb)
    mi = 0.0
    for (a, b), c in joint.items():
        p = c / n
        mi += p * math.log(p / ((ca[a] / n) * (cb[b] / n)))
    denom = 0.5 * (h_a + h_b)
    return mi / denom if denom > 0 else 0.0


def _deterministic_shuffle(values, seed):
    """In-place-style deterministic permutation (no Math.random / global RNG)."""
    vals = list(values)
    # Fisher-Yates with a simple LCG so the null is reproducible across runs.
    state = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    for i in range(len(vals) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        vals[i], vals[j] = vals[j], vals[i]
    return vals


def main():
    sub = json.load(open(DATA_DIR / "sub_clusters.json"))  # name -> [top, sub]
    names = set(sub)

    # full_name -> file_path for the sub-clustered declarations only
    fp = {}
    with open(DATA_DIR / "corpus.jsonl") as f:
        for line in f:
            o = json.loads(line)
            nm = o.get("full_name")
            if nm in names:
                fp[nm] = o.get("file_path", "")

    # Assemble aligned label vectors over declarations we can resolve.
    top, hier, mod1, mod2 = [], [], [], []
    resolved = 0
    for nm, (t, s) in sub.items():
        path = fp.get(nm)
        if not path:
            continue
        resolved += 1
        top.append(t)
        hier.append((t, s))
        mod1.append(_module(path, 1))
        mod2.append(_module(path, 2))

    # --- Global NMI against the module hierarchy --------------------------
    nmi_top_mod1 = _nmi(top, mod1)     # sanity check vs e-0002 (top-level)
    nmi_top_mod2 = _nmi(top, mod2)     # what the top split already captures of 2-level
    nmi_hier_mod2 = _nmi(hier, mod2)   # what the FULL hierarchy captures of 2-level
    nmi_hier_mod1 = _nmi(hier, mod1)

    # --- Within-top-cluster recovery: does the sub-split add 2-level structure?
    by_top = defaultdict(lambda: {"sub": [], "mod2": []})
    for t, h, m2 in zip(top, hier, mod2):
        by_top[t]["sub"].append(h[1])
        by_top[t]["mod2"].append(m2)

    per_cluster = []
    w_real, w_null, w_total = 0.0, 0.0, 0
    for t, d in by_top.items():
        size = len(d["sub"])
        if size < 50:
            continue
        real = _nmi(d["sub"], d["mod2"])
        # permutation null: shuffle sub labels within this cluster
        null = _nmi(_deterministic_shuffle(d["sub"], seed=t + 1), d["mod2"])
        per_cluster.append(
            {"top_cluster": t, "size": size,
             "n_subclusters": len(set(d["sub"])),
             "n_mod2": len(set(d["mod2"])),
             "within_nmi_sub_vs_mod2": round(real, 4),
             "within_nmi_null": round(null, 4)}
        )
        w_real += real * size
        w_null += null * size
        w_total += size

    per_cluster.sort(key=lambda r: -r["within_nmi_sub_vs_mod2"])

    # --- Sub-cluster purity at the 2-level granularity --------------------
    subcluster_mod2 = defaultdict(Counter)
    for h, m2 in zip(hier, mod2):
        subcluster_mod2[h][m2] += 1
    purities = []
    for h, counts in subcluster_mod2.items():
        tot = sum(counts.values())
        purities.append(counts.most_common(1)[0][1] / tot)
    mean_purity_mod2 = sum(purities) / len(purities)
    frac_pure_50 = sum(p >= 0.5 for p in purities) / len(purities)

    # top-level cluster purity at mod2 for contrast
    top_mod2 = defaultdict(Counter)
    for t, m2 in zip(top, mod2):
        top_mod2[t][m2] += 1
    top_pur = [c.most_common(1)[0][1] / sum(c.values()) for c in top_mod2.values()]
    mean_top_purity_mod2 = sum(top_pur) / len(top_pur)

    out = {
        "resolved_declarations": resolved,
        "total_subclustered": len(sub),
        "n_top_clusters": len(set(top)),
        "n_subclusters": len(set(hier)),
        "n_mod1_groups": len(set(mod1)),
        "n_mod2_groups": len(set(mod2)),
        "global_nmi": {
            "top_vs_mod1": round(nmi_top_mod1, 4),
            "top_vs_mod2": round(nmi_top_mod2, 4),
            "hier_vs_mod1": round(nmi_hier_mod1, 4),
            "hier_vs_mod2": round(nmi_hier_mod2, 4),
            "hierarchy_gain_on_mod2": round(nmi_hier_mod2 - nmi_top_mod2, 4),
        },
        "within_cluster_recovery": {
            "weighted_mean_nmi_sub_vs_mod2": round(w_real / w_total, 4),
            "weighted_mean_nmi_null": round(w_null / w_total, 4),
            "n_clusters_scored": len(per_cluster),
        },
        "purity_mod2": {
            "mean_subcluster_purity": round(mean_purity_mod2, 4),
            "frac_subclusters_purity_ge_0.5": round(frac_pure_50, 4),
            "mean_top_cluster_purity": round(mean_top_purity_mod2, 4),
        },
        "per_cluster": per_cluster,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS_DIR / "subcluster_validation.json", "w"), indent=2)

    g = out["global_nmi"]
    print(f"resolved {resolved}/{len(sub)} declarations; "
          f"{out['n_subclusters']} sub-clusters; {out['n_mod2_groups']} 2-level modules")
    print(f"NMI top_vs_mod1 (e-0002 analog) = {g['top_vs_mod1']}")
    print(f"NMI top_vs_mod2  = {g['top_vs_mod2']}")
    print(f"NMI hier_vs_mod2 = {g['hier_vs_mod2']}  (gain {g['hierarchy_gain_on_mod2']})")
    wc = out["within_cluster_recovery"]
    print(f"within-cluster NMI(sub;mod2) = {wc['weighted_mean_nmi_sub_vs_mod2']} "
          f"vs null {wc['weighted_mean_nmi_null']}")
    pm = out["purity_mod2"]
    print(f"mean sub-cluster purity@mod2 = {pm['mean_subcluster_purity']} "
          f"(>=0.5: {pm['frac_subclusters_purity_ge_0.5']}); "
          f"top-cluster purity@mod2 = {pm['mean_top_cluster_purity']}")


if __name__ == "__main__":
    main()
