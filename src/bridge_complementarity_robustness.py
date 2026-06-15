"""
Robustness of the operational-vs-conceptual bridge INDEPENDENCE finding (a-0032)
to the structural-similarity AGGREGATION statistic (q-0009 / central open problem).

a-0032 (supported) quantified that operational (shared-PREMISE) and conceptual
(shared proof-STRUCTURE) bridge detectors flag near-disjoint cluster pairs:
Jaccard 0.176, overlap-vs-chance ratio 0.76 (i.e. independent-to-mildly-
anti-correlated).  But it carries an explicit caveat: the structural detector
(edit_distance_bridges.py) records, per cluster pair, ONLY the single CLOSEST
representative theorem pair -- the MIN normalized edit distance over the
top-30 x top-30 cross-cluster proofs.  That is an extreme-tail / high-similarity
sample, not the typical cross-cluster proof similarity.  If the independence
result is an artifact of using the MIN, then using a DISTRIBUTION-level statistic
(MEAN / MEDIAN cross-cluster proof similarity) should make structural-bridge
cluster pairs correlate with operational ones.  If independence survives MIN,
MEAN and MEDIAN, a-0032 is robust to the aggregation choice.

This script reproduces the exact edit_distance_bridges.py selection (top-30
clusters, top-30 longest proofs each, >=5 tactics, 80-token cap, normalized
Levenshtein) but records the FULL per-cluster-pair distribution (min / mean /
median).  It then re-runs the bridge_complementarity.py overlap analysis once
per aggregation statistic, restricted to the same jointly-evaluable cluster
universe, and reports Jaccard + overlap-vs-chance for each.

Pure stdlib (no numpy); CPU-only; reuses on-disk JSON data + the operational
bridge list already on disk.

Inputs:
  data/theorem_tactics.json           proof tactic sequences per theorem
  data/clusters.json                  theorem -> cluster id
  data/cluster_labels.json            cluster id -> label
  results/bridge_theorems.json        280 operational (premise) bridges
Output:
  results/bridge_complementarity_robustness.json
"""

import json
import statistics
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

from normalize_tactics import normalize_sequence

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# --- match edit_distance_bridges.py exactly ---------------------------------
MIN_TACTICS = 5
TOP_PER_CLUSTER = 30
MAX_CLUSTERS = 30


def _normalised_ed(a, b):
    """Token-level Levenshtein normalized by the longer length; 80-token cap."""
    if len(a) > 80 or len(b) > 80:
        a, b = a[:80], b[:80]
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ai == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n] / max(m, n, 1)


def upair(a, b):
    return tuple(sorted((int(a), int(b))))


def build_pair_stats():
    theorem_tactics = json.loads((DATA / "theorem_tactics.json").read_text())
    cluster_map = {k: int(v) for k, v in json.loads((DATA / "clusters.json").read_text()).items()}
    labels = json.loads((DATA / "cluster_labels.json").read_text())

    # normalise + length filter (identical to find_edit_bridges)
    normed = {}
    for thm, tacs in theorem_tactics.items():
        seq = normalize_sequence([t.strip() for t in tacs])
        if len(seq) >= MIN_TACTICS:
            normed[thm] = seq

    by_cluster = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))

    target = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    for cid in target:
        by_cluster[cid] = sorted(by_cluster[cid], key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]

    # within-cluster baseline (median over first <=10 proofs per cluster) -- as original
    within = []
    for cid in target:
        proofs = by_cluster[cid]
        k = min(10, len(proofs))
        for i in range(k):
            for j in range(i + 1, k):
                within.append(_normalised_ed(proofs[i][1], proofs[j][1]))
    baseline = statistics.median(within) if within else 1.0

    # cross-cluster: record full distribution stats per cluster pair
    pair_stats = {}
    pairs = list(combinations(target, 2))
    for idx, (ca, cb) in enumerate(pairs):
        eds = [_normalised_ed(sa, sb) for _, sa in by_cluster[ca] for _, sb in by_cluster[cb]]
        if not eds:
            continue
        pair_stats[upair(ca, cb)] = {
            "min": min(eds),
            "mean": statistics.fmean(eds),
            "median": statistics.median(eds),
        }
        if (idx + 1) % 50 == 0:
            print(f"  ...{idx + 1}/{len(pairs)} cluster pairs")

    id2label = {int(k): v.get("label", k) for k, v in labels.items()}
    return pair_stats, baseline, set(target), id2label


def overlap_stats(opp, edp, n_possible):
    inter = opp & edp
    union = opp | edp
    exp_rand = (len(opp) * len(edp) / n_possible) if n_possible else 0.0
    return {
        "n_operational_pairs": len(opp),
        "n_structural_pairs": len(edp),
        "n_overlap": len(inter),
        "n_structural_only": len(edp - opp),
        "n_operational_only": len(opp - edp),
        "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "expected_overlap_if_independent": round(exp_rand, 3),
        "overlap_vs_chance_ratio": round(len(inter) / exp_rand, 3) if exp_rand else None,
    }


def main():
    print("Computing per-cluster-pair edit-distance distributions (min/mean/median)...")
    pair_stats, baseline, target_clusters, id2label = build_pair_stats()
    print(f"Baseline (within-cluster median ED) = {baseline:.4f}; "
          f"{len(pair_stats)} cluster pairs scored.")

    op = json.loads((RESULTS / "bridge_theorems.json").read_text())

    # jointly-evaluable cluster universe (same construction as bridge_complementarity)
    op_clusters = set()
    for b in op:
        op_clusters.add(int(b["self_cluster"]))
        op_clusters.update(int(c) for c in b["premise_cluster_ids"])
    U = target_clusters & op_clusters
    n_possible = comb(len(U), 2)

    def in_U(p):
        return p[0] in U and p[1] in U

    op_pairs = set()
    for b in op:
        s = int(b["self_cluster"])
        for c in b["premise_cluster_ids"]:
            p = upair(s, c)
            if p[0] != p[1] and in_U(p):
                op_pairs.add(p)

    base_rate = len(op_pairs) / n_possible  # operational density over jointly-evaluable pairs

    # Threshold-free comparison: take the K CLOSEST cluster pairs by each statistic
    # (smallest agg value = most structurally similar) at several K, and measure how
    # enriched they are for operational bridges vs the base rate. Robust to the
    # degenerate within-cluster baseline (=1.0) and to median ties at 1.0.
    K_GRID = [20, 40, 58, 80]  # 58 ~= closest quartile of the 231 in-U pairs
    results = {}
    for agg in ("min", "mean", "median"):
        scored = sorted(((st[agg], p) for p, st in pair_stats.items() if in_U(p)))
        topk = {}
        for K in K_GRID:
            sel = {p for _, p in scored[:K]}
            ov = len(sel & op_pairs)
            exp = len(sel) * base_rate
            topk[str(K)] = {
                "n_selected": len(sel),
                "n_operational_in_selection": ov,
                "frac_operational_in_selection": round(ov / len(sel), 4) if sel else 0.0,
                "expected_if_independent": round(exp, 3),
                "enrichment_vs_chance": round(ov / exp, 3) if exp else None,
            }
        results[agg] = {"closest_topK": topk}

    out = {
        "description": "Robustness of a-0032 operational/conceptual independence to the "
                       "structural-similarity aggregation statistic (min vs mean vs median). "
                       "Threshold-free: enrichment of the K structurally-closest cluster pairs "
                       "for operational bridges, relative to the base rate.",
        "n_jointly_evaluable_clusters": len(U),
        "n_possible_pairs_in_U": n_possible,
        "n_operational_pairs_in_U": len(op_pairs),
        "operational_base_rate": round(base_rate, 4),
        "within_cluster_baseline_median_ed": round(baseline, 4),
        "by_aggregation": results,
    }
    (RESULTS / "bridge_complementarity_robustness.json").write_text(json.dumps(out, indent=2))

    print(f"\nJointly-evaluable clusters: {len(U)} ({n_possible} possible pairs)")
    print(f"Operational pairs in U: {len(op_pairs)}  (base rate {base_rate:.3f})\n")
    print("Enrichment-vs-chance of the K structurally-CLOSEST cluster pairs "
          "(1.0 = independent, >1 = co-located, <1 = anti):\n")
    header = "  agg   |" + "".join(f"  K={K:<3}" for K in K_GRID)
    print(header)
    print("-" * len(header))
    for agg in ("min", "mean", "median"):
        row = f"{agg:>7} |"
        for K in K_GRID:
            row += f"  {results[agg]['closest_topK'][str(K)]['enrichment_vs_chance']:<5}"
        print(row)
    print(f"\nWrote {RESULTS / 'bridge_complementarity_robustness.json'}")


if __name__ == "__main__":
    main()
