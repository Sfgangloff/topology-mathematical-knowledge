"""
Hybrid UNION router: are the static bridge signal and the goal name-token signal
COMPLEMENTARY? (q-0011, follow-on to e-0030/a-0027, e-0031/a-0028, e-0032/a-0029).

The goal-router sub-chain established:
  * e-0030 (a-0027): a GOAL-specific name-token router (home + top-m predicted foreign
    clusters) beats the STATIC top-K bridge router (home + top-K W-neighbours) at matched
    complete-coverage breadth -- but leaves a ~3-4x gap to the leaky oracle goal-router.
  * e-0031 (a-0028): de-biasing the predictor by the GLOBAL demand prior P(d) gives only a
    mild lift; pushing it (full PMI/lift) COLLAPSES.
  * e-0032 (a-0029): fusing the HOME-specific bridge prior W[home][.] INTO the predictor
    score never helps and degrades top-1 accuracy.

Both fusion attempts (e-0031, e-0032) mixed the structural prior INTO the predictor's
SCORE for a single ranked list.  They never tested the simplest combination: take the
SET UNION of the two routers' chosen clusters.  Score-fusion conflates the two signals;
a union keeps them separate -- the static arm contributes the home cluster's globally
heaviest bridge neighbours (the common hub escapes), the goal arm contributes the
goal-specific specialized clusters its name tokens point to.  If the two routing signals
are COMPLEMENTARY, a hybrid that spends part of its breadth on each should reach a given
complete-coverage ceiling at a SMALLER candidate pool than either pure arm; if they are
REDUNDANT (the goal arm already subsumes the static hubs, or vice versa) the hybrid lies
on the better pure arm's curve and buys nothing.

Arms (only the routing policy changes; SAME 432 held-out cross-domain TEST theorems, SAME
train-only bridge weights W, SAME name-token predictor and leakage control as e-0030):
  * static(K)      : home + top-K bridge neighbours by train weight W[home][.].
  * goal(m)        : home + top-m foreign clusters predicted from the goal's NAME tokens.
  * hybrid(a,b)    : home + top-a static bridge neighbours + top-b goal-predicted clusters
                     (set union; overlaps counted once, so pool can be < a+b clusters).

Metric (identical to e-0030): complete-coverage CEILING = fraction of cross-domain TEST
theorems whose ENTIRE clustered premise set lands inside the routed cluster pool (exact,
ranking-free, upper bound above any intra-cluster ranker).  Candidate cost = mean sum of
routed cluster sizes.

Analysis: build the Pareto efficiency frontier (mean pool size -> max ceiling) over ALL
hybrid (a,b) combos and over the pure static / goal arms, then report, at each target
ceiling, the cheapest pool each FAMILY needs -- so we can see whether the hybrid frontier
strictly dominates both pure arms (complementary) or merely matches the better one
(redundant).

Leakage control: W, IDF, token x foreign-cluster co-occurrence counts and the global
foreign-cluster demand ranking are ALL estimated on TRAIN only; coverage on held-out TEST;
the only query signal is the test theorem's own NAME.

Output: results/bridge_retrieval_hybrid_router.json
"""

import json
import collections
from pathlib import Path

# Reuse the exact machinery of the goal-router experiment (same loader, W, IDF, predictor).
from bridge_retrieval_goalrouter import (
    load, build_bridge_weights, build_idf, cross_domain_set,
    train_foreign_predictor, predict_foreign, K_SWEEP, M_SWEEP,
)

RESULTS = Path(__file__).parent.parent / "results"
OUT = RESULTS / "bridge_retrieval_hybrid_router.json"

# Breadth grids for the hybrid: a static neighbours + b goal-predicted clusters.
A_GRID = [0, 1, 2, 3, 5, 10]
B_GRID = [0, 1, 2, 3, 5, 10]


def ceiling_and_pool(test, cluster_size, route_fn, breadth):
    """Return (complete-coverage ceiling, mean pool size, median pool size)."""
    n = len(test)
    covered = 0
    pool_sizes = []
    for fn, c, prem_clusters in test:
        routed = route_fn(fn, c, prem_clusters, breadth)
        pool_sizes.append(sum(cluster_size[cid] for cid in routed))
        if prem_clusters <= routed:
            covered += 1
    pool_sizes.sort()
    return covered / n, sum(pool_sizes) / n, pool_sizes[len(pool_sizes) // 2]


def pareto_frontier(points):
    """points: list of dicts with 'mean_pool' and 'ceiling'. Keep only non-dominated
    points (a point dominates another if it has >= ceiling at <= pool)."""
    pts = sorted(points, key=lambda p: (p["mean_pool"], -p["ceiling"]))
    front = []
    best_ceil = -1.0
    for p in pts:
        if p["ceiling"] > best_ceil + 1e-12:
            front.append(p)
            best_ceil = p["ceiling"]
    return front


def cheapest_to_reach(rows, target):
    """Cheapest mean_pool among rows whose ceiling >= target."""
    best = None
    for r in rows:
        if r["ceiling"] >= target - 1e-9:
            if best is None or r["mean_pool"] < best["mean_pool"]:
                best = r
    return best


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    max_static = max(max(A_GRID), max(K_SWEEP))
    top_static = {c: [d for d, _ in W[c].most_common(max_static)] for c in W}

    idf, stop = build_idf(cluster)
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    tok_cluster, tok_total, global_demand = train_foreign_predictor(train, idf, stop)

    # Per-theorem goal prediction (exclude only the home cluster), cached.
    pred_cache = {fn: predict_foreign(fn, tok_cluster, tok_total, idf, stop, {c})
                  for fn, c, _ in test}

    def route_static(fn, c, prem_clusters, K):
        return {c} | set(top_static.get(c, [])[:K])

    def route_goal(fn, c, prem_clusters, m):
        return {c} | set(pred_cache[fn][:m])

    def route_hybrid(fn, c, prem_clusters, ab):
        a, b = ab
        return ({c} | set(top_static.get(c, [])[:a]) | set(pred_cache[fn][:b]))

    # ---- pure arms ----
    static_rows = []
    for K in K_SWEEP:
        ceil, mp, md = ceiling_and_pool(test, cluster_size, route_static, K)
        static_rows.append({"arm": "static", "K": K, "ceiling": round(ceil, 4),
                            "mean_pool": round(mp, 1), "median_pool": md})
    goal_rows = []
    for m in M_SWEEP:
        ceil, mp, md = ceiling_and_pool(test, cluster_size, route_goal, m)
        goal_rows.append({"arm": "goal", "m": m, "ceiling": round(ceil, 4),
                          "mean_pool": round(mp, 1), "median_pool": md})

    # ---- hybrid grid ----
    hybrid_rows = []
    for a in A_GRID:
        for b in B_GRID:
            if a == 0 and b == 0:
                continue  # home-only, trivially ceiling 0 on cross-domain
            ceil, mp, md = ceiling_and_pool(test, cluster_size, route_hybrid, (a, b))
            hybrid_rows.append({"arm": "hybrid", "a": a, "b": b,
                                "ceiling": round(ceil, 4),
                                "mean_pool": round(mp, 1), "median_pool": md})

    # ---- Pareto frontiers (pool -> ceiling) per family ----
    def front(rows):
        return pareto_frontier([{"mean_pool": r["mean_pool"], "ceiling": r["ceiling"],
                                 **{k: r[k] for k in r if k not in ("mean_pool", "ceiling")}}
                                for r in rows])

    static_front = front(static_rows)
    goal_front = front(goal_rows)
    hybrid_front = front(hybrid_rows)
    all_front = front(static_rows + goal_rows + hybrid_rows)

    # ---- head-to-head: cheapest pool each family needs to reach a target ceiling ----
    targets = [0.50, 0.60, 0.674, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    head2head = []
    for t in targets:
        s = cheapest_to_reach(static_rows, t)
        g = cheapest_to_reach(goal_rows, t)
        h = cheapest_to_reach(hybrid_rows, t)
        row = {
            "target_ceiling": t,
            "static": None if s is None else {"K": s["K"], "pool": s["mean_pool"], "ceiling": s["ceiling"]},
            "goal": None if g is None else {"m": g["m"], "pool": g["mean_pool"], "ceiling": g["ceiling"]},
            "hybrid": None if h is None else {"a": h["a"], "b": h["b"], "pool": h["mean_pool"], "ceiling": h["ceiling"]},
        }
        # is the hybrid strictly cheaper than the best pure arm?
        pure = [x for x in (row["static"], row["goal"]) if x is not None]
        if h is not None and pure:
            best_pure = min(x["pool"] for x in pure)
            row["hybrid_pool_ratio_vs_best_pure"] = round(h["mean_pool"] / best_pure, 3)
            row["hybrid_strictly_cheaper"] = bool(h["mean_pool"] < best_pure - 1e-9)
        head2head.append(row)

    # Does the hybrid contribute ANY point to the combined Pareto frontier that no pure arm matches?
    hybrid_on_combined = [p for p in all_front if p.get("arm") == "hybrid"]

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free, upper bound above any intra-cluster ranker)",
        "question": "Are the static bridge router and the goal name-token router COMPLEMENTARY? Does their set-UNION (hybrid) reach a target complete-coverage ceiling at a smaller candidate pool than either pure arm?",
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "total_clustered_decls": sum(cluster_size.values()),
        "leakage_control": "bridge weights W, IDF, token x foreign-cluster co-occurrence counts, and global foreign-cluster demand ALL estimated on TRAIN only; coverage measured on held-out TEST theorems; only query signal is the test theorem's own NAME.",
        "static_sweep": static_rows,
        "goal_sweep": goal_rows,
        "hybrid_grid": hybrid_rows,
        "static_pareto": static_front,
        "goal_pareto": goal_front,
        "hybrid_pareto": hybrid_front,
        "combined_pareto": all_front,
        "hybrid_points_on_combined_pareto": hybrid_on_combined,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"train_cross = {res['n_train_cross_domain']}")
    print("\nstatic sweep:   K  ceiling  mean_pool")
    for r in res["static_sweep"]:
        print(f"   {r['K']:>3} {r['ceiling']:>8} {r['mean_pool']:>11}")
    print("goal sweep:     m  ceiling  mean_pool")
    for r in res["goal_sweep"]:
        print(f"   {r['m']:>3} {r['ceiling']:>8} {r['mean_pool']:>11}")
    print("\nhybrid grid (a static + b goal):  a  b  ceiling  mean_pool")
    for r in res["hybrid_grid"]:
        print(f"   a={r['a']:>2} b={r['b']:>2}  {r['ceiling']:>8} {r['mean_pool']:>11}")
    print("\ncombined Pareto frontier (pool -> ceiling):")
    for p in res["combined_pareto"]:
        tag = p.get("arm")
        extra = (f"K={p.get('K')}" if tag == "static" else
                 f"m={p.get('m')}" if tag == "goal" else
                 f"a={p.get('a')},b={p.get('b')}")
        print(f"   pool={p['mean_pool']:>9}  ceiling={p['ceiling']:<7} [{tag} {extra}]")
    print("\nhead-to-head cheapest pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x, keys):
            if x is None:
                return "unreachable"
            ks = " ".join(f"{k}={x[k]}" for k in keys)
            return f"{ks} pool={x['pool']}"
        ratio = h.get("hybrid_pool_ratio_vs_best_pure")
        rtxt = f"  hybrid/best_pure={ratio}" if ratio is not None else ""
        print(f"   ceiling>={h['target_ceiling']}: "
              f"static[{fmt(h['static'], ['K'])}]  goal[{fmt(h['goal'], ['m'])}]  "
              f"hybrid[{fmt(h['hybrid'], ['a', 'b'])}]{rtxt}")


if __name__ == "__main__":
    main()
