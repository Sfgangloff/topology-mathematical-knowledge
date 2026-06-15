"""
Operational vs conceptual bridge COMPLEMENTARITY (q-0009 / central open problem).

The capstone (a-0001) names the operational-vs-conceptual bridge gap as the
central open problem: operational bridges (shared PREMISES, detected by
bridge_theorems.py) vs conceptual bridges (shared proof STRUCTURE, surfaced by
edit_distance_bridges.py). a-0009 asserts these are a "second, distinct class"
the premise detector misses -- but only qualitatively (two named examples).

This script quantifies it directly over existing results, with NO new heavy
compute: do the two detectors flag the SAME cluster pairs, or DISJOINT ones?
If overlap is at/below chance, the two bridge types are genuinely complementary
-- structural similarity connects domain pairs that share no specialized premises.

Inputs (already on disk):
  results/bridge_theorems.json        280 operational bridges (premise-based)
  results/edit_distance_bridges.json  200 structural bridges  (tactic-sequence)

Output:
  results/bridge_complementarity.json
"""

import json
from itertools import combinations
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def upair(a, b):
    return tuple(sorted((int(a), int(b))))


def main():
    op = json.loads((RESULTS / "bridge_theorems.json").read_text())
    ed = json.loads((RESULTS / "edit_distance_bridges.json").read_text())

    # --- Cluster universes each method actually evaluated ---------------------
    # Structural method only scored its TOP-30 clusters; operational scored all
    # pure clusters. A cluster pair is JOINTLY EVALUABLE only if both clusters
    # sit in BOTH universes -- restrict the comparison there so neither method is
    # penalized for clusters the other never saw.
    ed_clusters = set()
    for e in ed:
        ed_clusters.add(int(e["cluster_a"]))
        ed_clusters.add(int(e["cluster_b"]))
    op_clusters = set()
    for b in op:
        op_clusters.add(int(b["self_cluster"]))
        op_clusters.update(int(c) for c in b["premise_cluster_ids"])

    U = ed_clusters & op_clusters  # jointly-evaluable cluster universe
    n_possible = comb(len(U), 2)

    def in_U(p):
        return p[0] in U and p[1] in U

    # --- Cluster-pair sets flagged by each detector (restricted to U) ---------
    op_pairs = set()
    for b in op:
        s = int(b["self_cluster"])
        for c in b["premise_cluster_ids"]:
            p = upair(s, c)
            if p[0] != p[1] and in_U(p):
                op_pairs.add(p)

    # Structural bridges: all 200 reported pairs are below the within-cluster
    # baseline (~1.0); also report a STRICT subset (closest quartile by distance)
    # to show robustness to the "what counts as a structural bridge" threshold.
    ed_all = [e for e in ed if in_U(upair(e["cluster_a"], e["cluster_b"]))]
    ed_pairs_all = {upair(e["cluster_a"], e["cluster_b"]) for e in ed_all}

    dists = sorted(e["norm_edit_dist"] for e in ed_all)
    q1 = dists[len(dists) // 4] if dists else 0.0  # closest-quartile cutoff
    ed_pairs_strict = {
        upair(e["cluster_a"], e["cluster_b"])
        for e in ed_all
        if e["norm_edit_dist"] <= q1
    }

    def overlap_stats(opp, edp, label):
        inter = opp & edp
        union = opp | edp
        jacc = len(inter) / len(union) if union else 0.0
        # Expected overlap under independence (hypergeometric mean) over the
        # n_possible jointly-evaluable pairs.
        exp_rand = (len(opp) * len(edp) / n_possible) if n_possible else 0.0
        return {
            "label": label,
            "n_operational_pairs": len(opp),
            "n_structural_pairs": len(edp),
            "n_overlap": len(inter),
            "n_structural_only": len(edp - opp),
            "n_operational_only": len(opp - edp),
            "jaccard": round(jacc, 4),
            "expected_overlap_if_independent": round(exp_rand, 3),
            "overlap_vs_chance_ratio": round(len(inter) / exp_rand, 3) if exp_rand else None,
            "structural_only_pairs": sorted(edp - opp),
        }

    stats_all = overlap_stats(op_pairs, ed_pairs_all, "all_structural")
    stats_strict = overlap_stats(op_pairs, ed_pairs_strict, "strict_closest_quartile")

    # --- Theorem-name level: do the two detectors surface the SAME theorems? --
    op_theorems = {b["theorem"] for b in op}
    ed_theorems = set()
    for e in ed:
        ed_theorems.add(e["theorem_a"])
        ed_theorems.add(e["theorem_b"])
    thm_overlap = op_theorems & ed_theorems

    # --- Label the conceptual-only domain pairs (structural, no shared premise)
    id2label = {}
    for e in ed:
        id2label[int(e["cluster_a"])] = e["label_a"]
        id2label[int(e["cluster_b"])] = e["label_b"]
    structural_only_labeled = [
        {"pair": list(p), "label_a": id2label.get(p[0]), "label_b": id2label.get(p[1])}
        for p in stats_all["structural_only_pairs"]
    ]

    out = {
        "jointly_evaluable_clusters": sorted(U),
        "n_jointly_evaluable_clusters": len(U),
        "n_possible_pairs_in_U": n_possible,
        "structural_strict_cutoff_norm_edit_dist": round(q1, 4),
        "cluster_pair_overlap": {
            "all_structural": stats_all,
            "strict_closest_quartile": stats_strict,
        },
        "theorem_level": {
            "n_operational_theorems": len(op_theorems),
            "n_structural_theorems": len(ed_theorems),
            "n_shared_theorems": len(thm_overlap),
            "shared_theorems": sorted(thm_overlap),
        },
        "structural_only_pairs_labeled": structural_only_labeled,
    }

    (RESULTS / "bridge_complementarity.json").write_text(json.dumps(out, indent=2))

    # --- Console summary ------------------------------------------------------
    print(f"Jointly-evaluable clusters: {len(U)}  ({n_possible} possible pairs)")
    print()
    for key in ("all_structural", "strict_closest_quartile"):
        s = out["cluster_pair_overlap"][key]
        print(f"[{key}] structural cutoff dist<= {q1 if key!='all_structural' else 0.78}")
        print(f"  operational pairs={s['n_operational_pairs']}  "
              f"structural pairs={s['n_structural_pairs']}  overlap={s['n_overlap']}")
        print(f"  structural-only={s['n_structural_only']}  operational-only={s['n_operational_only']}")
        print(f"  Jaccard={s['jaccard']}  expected-overlap-if-independent="
              f"{s['expected_overlap_if_independent']}  ratio={s['overlap_vs_chance_ratio']}")
        print()
    print(f"Theorem-level: operational={len(op_theorems)} structural={len(ed_theorems)} "
          f"shared={len(thm_overlap)}")
    print(f"Wrote {RESULTS / 'bridge_complementarity.json'}")


if __name__ == "__main__":
    main()
