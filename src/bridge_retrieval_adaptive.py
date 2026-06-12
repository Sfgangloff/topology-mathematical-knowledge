"""
Home-cluster-adaptive router breadth vs the global breadth dial. (q-0011, follow-on to
e-0019/a-0018)

e-0019 swept a SINGLE global router breadth K over all cross-domain theorems and found the
complete-coverage CEILING climbs with K but at steeply rising, ultimately whole-library
candidate cost (a-0018: "a precision-cheap vs coverage-complete dial, not a free recovery of
the hard third").  But it applied the SAME K to every theorem.  A prover does not know a
theorem's premises before proving, yet it DOES know the one cheap feature that conditions
hardness: the theorem's home cluster.  If clusters are heterogeneous in how wide a router
their cross-domain theorems need -- some reliably hub/bridge-served at top-5, others
chronically needing far-flung premises -- then setting the breadth PER HOME CLUSTER should
reach a target complete-coverage at lower mean candidate cost than any single global K.

This script tests that lever, leakage-controlled.  For each home cluster it estimates, on the
TRAIN split only, the complete-coverage achievable at each breadth K; the adaptive policy at
target t routes each TEST theorem to the MINIMAL breadth whose train coverage for that home
cluster reaches t (whole-library fallback if none does).  Held identical to e-0019: same data,
same cross-domain TEST theorems, same train-only bridge weights W, same complete-coverage
metric (ENTIRE clustered premise set must land in the routed pool -- exact, ranking-free).
ONLY the routing changes: global K  ->  per-home-cluster K(t).

It reports, on TEST:
  * the per-cluster K=5 complete-coverage distribution (is hardness heterogeneous by home?)
  * train->test correlation of per-cluster K=5 coverage (does train hardness predict test?)
  * the adaptive (mean candidate cost -> complete-coverage) frontier swept over target t
  * the global (mean candidate cost -> complete-coverage) frontier (recomputed, e-0019 logic)
  * at matched complete-coverage, the candidate-cost ratio adaptive / global

Output: results/bridge_retrieval_adaptive.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_adaptive.json"

# breadth grid the adaptive policy may choose from per cluster; None == whole-library fallback
K_GRID = [0, 1, 2, 3, 5, 10, 20, 50, None]
# targets t for the adaptive policy and the matched-coverage comparison points
TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def load():
    cluster = json.load(open(CLUSTERS))
    premises = collections.defaultdict(set)
    split = {}
    for line in open(STEPS):
        d = json.loads(line)
        fn = d["full_name"]
        split[fn] = d["split"]
        for p in d.get("premises", []):
            if p != fn:
                premises[fn].add(p)
    return cluster, premises, split


def build_bridge_weights(cluster, premises, split):
    """W[c][d] = #train theorems in cluster c that cite a premise in cluster d (d!=c)."""
    W = collections.defaultdict(lambda: collections.Counter())
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        seen = set()
        for p in ps:
            d = cluster.get(p)
            if d is None or d == c or d in seen:
                continue
            seen.add(d)
            W[c][d] += 1
    return W


def cross_domain_theorems(cluster, premises, split, which):
    """Yield (home_cluster, frozenset(premise_clusters)) for cross-domain theorems in `which` split."""
    out = []
    for fn, ps in premises.items():
        if split.get(fn) != which:
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem_clusters = {cluster[p] for p in ps if p in cluster}
        if not prem_clusters or prem_clusters <= {c}:
            continue
        out.append((c, frozenset(prem_clusters)))
    return out


def routed_set(c, neigh, K):
    """The set of clusters routed for home c at breadth K (None -> sentinel 'all')."""
    if K is None:
        return None
    return {c} | set(neigh.get(c, [])[:K])


def main():
    cluster, premises, split = load()
    W = build_bridge_weights(cluster, premises, split)
    neigh = {c: [d for d, _ in W[c].most_common()] for c in W}
    cluster_size = collections.Counter(cluster.values())
    all_clustered = sum(cluster_size.values())

    train = cross_domain_theorems(cluster, premises, split, "train")
    test = cross_domain_theorems(cluster, premises, split, "test")

    def pool_size(c, K):
        r = routed_set(c, neigh, K)
        return all_clustered if r is None else sum(cluster_size[cid] for cid in r)

    def covered(c, prem_clusters, K):
        r = routed_set(c, neigh, K)
        return True if r is None else (prem_clusters <= r)

    # --- per-cluster train complete-coverage at each finite breadth K ---
    by_home_train = collections.defaultdict(list)
    for c, pc in train:
        by_home_train[c].append(pc)
    # train_cov[c][K] = fraction of cluster c's train cross-domain theorems fully covered at K
    train_cov = {}
    for c, items in by_home_train.items():
        train_cov[c] = {}
        for K in K_GRID:
            if K is None:
                train_cov[c][K] = 1.0
            else:
                hit = sum(covered(c, pc, K) for pc in items)
                train_cov[c][K] = hit / len(items)

    # aggregate (global) train coverage at each K -> fallback breadth for clusters unseen in train
    def global_train_cov(K):
        if not train:
            return 1.0
        return sum(covered(c, pc, K) for c, pc in train) / len(train)
    agg_train_cov = {K: global_train_cov(K) for K in K_GRID}

    def min_K_for(cov_at_K, t):
        """smallest breadth in K_GRID whose coverage>=t (None/all is last and always 1.0)."""
        for K in K_GRID:
            if cov_at_K[K] >= t:
                return K
        return None

    # --- descriptive: per-cluster K=5 coverage heterogeneity, train vs test ---
    by_home_test = collections.defaultdict(list)
    for c, pc in test:
        by_home_test[c].append(pc)
    per_cluster_k5 = []
    common = [c for c in by_home_test if c in train_cov and len(by_home_test[c]) >= 5]
    for c in common:
        tr = train_cov[c][5]
        te = sum(covered(c, pc, 5) for pc in by_home_test[c]) / len(by_home_test[c])
        per_cluster_k5.append({"cluster": c, "n_test": len(by_home_test[c]),
                               "train_cov_k5": round(tr, 3), "test_cov_k5": round(te, 3)})

    def pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        return round(cov / (vx * vy), 4) if vx > 0 and vy > 0 else None

    corr = pearson([r["train_cov_k5"] for r in per_cluster_k5],
                   [r["test_cov_k5"] for r in per_cluster_k5])
    k5_vals = sorted(r["train_cov_k5"] for r in per_cluster_k5)
    het = {
        "n_clusters_ge5_test": len(per_cluster_k5),
        "train_cov_k5_min": k5_vals[0] if k5_vals else None,
        "train_cov_k5_median": k5_vals[len(k5_vals) // 2] if k5_vals else None,
        "train_cov_k5_max": k5_vals[-1] if k5_vals else None,
        "frac_clusters_train_k5_below_0.5": round(
            sum(v < 0.5 for v in k5_vals) / len(k5_vals), 3) if k5_vals else None,
        "frac_clusters_train_k5_above_0.9": round(
            sum(v > 0.9 for v in k5_vals) / len(k5_vals), 3) if k5_vals else None,
        "train_test_k5_pearson": corr,
    }

    n = len(test)

    # --- adaptive frontier: per-home-cluster minimal breadth to reach train target t ---
    adaptive = []
    for t in TARGETS:
        # precompute per-cluster chosen K
        cl_K = {}
        for c in set(c for c, _ in test):
            cov_at_K = train_cov.get(c, agg_train_cov)  # fallback to global if cluster unseen
            cl_K[c] = min_K_for(cov_at_K, t)
        cov_hits, cost_sum = 0, 0
        for c, pc in test:
            K = cl_K[c]
            cost_sum += pool_size(c, K)
            if covered(c, pc, K):
                cov_hits += 1
        adaptive.append({"target_t": t,
                         "test_complete_coverage": round(cov_hits / n, 4),
                         "mean_candidate_cost": round(cost_sum / n, 1)})

    # --- global frontier: single breadth K for all (e-0019 logic, recomputed here) ---
    global_frontier = []
    for K in K_GRID:
        cov_hits, cost_sum = 0, 0
        for c, pc in test:
            cost_sum += pool_size(c, K)
            if covered(c, pc, K):
                cov_hits += 1
        global_frontier.append({"k_route": "all" if K is None else K,
                                "test_complete_coverage": round(cov_hits / n, 4),
                                "mean_candidate_cost": round(cost_sum / n, 1)})

    # --- matched-coverage comparison: for each adaptive point, the cost of the cheapest
    #     global breadth reaching at least the same complete-coverage ---
    def cheapest_global_at_least(cov):
        cand = [g for g in global_frontier if g["test_complete_coverage"] >= cov - 1e-9]
        return min(cand, key=lambda g: g["mean_candidate_cost"]) if cand else None

    matched = []
    for a in adaptive:
        g = cheapest_global_at_least(a["test_complete_coverage"])
        if g is None:
            continue
        matched.append({
            "complete_coverage": a["test_complete_coverage"],
            "adaptive_mean_cost": a["mean_candidate_cost"],
            "cheapest_global_k": g["k_route"],
            "global_mean_cost": g["mean_candidate_cost"],
            "adaptive_over_global_cost_ratio": round(
                a["mean_candidate_cost"] / g["mean_candidate_cost"], 3),
            "candidates_saved_per_theorem": round(
                g["mean_candidate_cost"] - a["mean_candidate_cost"], 1),
        })

    res = {
        "metric": "complete-coverage rate (fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed pool); cost = mean routed-pool size (candidate decls) per theorem",
        "n_cross_domain_test_theorems": n,
        "n_cross_domain_train_theorems": len(train),
        "total_clustered_decls": all_clustered,
        "router": "home cluster + top-K bridge-neighbour clusters; adaptive policy picks K per home cluster as the minimal breadth reaching TRAIN complete-coverage>=t (whole-library fallback)",
        "leakage_control": "bridge weights W and per-cluster breadth choice estimated on TRAIN split only; complete-coverage and cost measured on held-out TEST theorems",
        "k_grid": ["all" if k is None else k for k in K_GRID],
        "per_home_cluster_k5_heterogeneity": het,
        "adaptive_frontier": adaptive,
        "global_frontier": global_frontier,
        "matched_coverage_comparison": matched,
    }
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"n_cross_domain_test = {n}, train = {len(train)}, total_clustered = {all_clustered}")
    print("\nper-home-cluster K=5 coverage heterogeneity:")
    for k, v in het.items():
        print(f"  {k}: {v}")
    print("\nmatched-coverage: adaptive (per-home K) vs cheapest global K at >= same coverage")
    print(f"  {'coverage':>9} {'adapt_cost':>11} {'glob_k':>7} {'glob_cost':>10} {'ratio':>6} {'saved':>9}")
    for m in matched:
        print(f"  {m['complete_coverage']:>9} {m['adaptive_mean_cost']:>11} "
              f"{str(m['cheapest_global_k']):>7} {m['global_mean_cost']:>10} "
              f"{m['adaptive_over_global_cost_ratio']:>6} {m['candidates_saved_per_theorem']:>9}")


if __name__ == "__main__":
    main()
