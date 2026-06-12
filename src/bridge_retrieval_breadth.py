"""
Router-breadth vs the complete-coverage CEILING: can a WIDER bridge router buy back the
'hard third' the prover-relevant bar left unreachable?  (q-0011, follow-on to e-0017/a-0017)

e-0017 fixed the bridge router at home + top-5 bridge-neighbour clusters and, under the
prover-relevant COMPLETE-COVERAGE metric (a cross-domain step is only possible if EVERY
premise it needs is retrieved), found a hard ceiling of 0.674: ~1/3 of cross-domain test
theorems need at least one premise OUTSIDE the top-5 router and so are unreachable at ANY
candidate budget.  a-0017 named the obvious untested lever:
  "a hard third need a broader router (more neighbour clusters or whole-library fallback)".

This script measures that lever directly and cheaply.  The complete-coverage CEILING at a
given router breadth depends ONLY on whether a theorem's full premise set lands in the
routed POOL -- it needs no intra-cluster ranking, so it is exact and fast.  Holding
everything else identical to e-0017 (same data, same 432 held-out cross-domain TEST
theorems, same train-only bridge weights W, same leakage control), it sweeps the router
breadth K_ROUTE over {0(home-only),1,2,3,5,10,20,50,all} and reports per breadth:

  * complete-coverage CEILING  = fraction whose ENTIRE clustered premise set is in the pool
                                 (the max complete-coverage rate achievable at any budget)
  * mean / median / max routed-POOL size = the candidate-cost floor of that breadth
  * marginal ceiling gain and pool-size cost of each widening step

The K=all arm (route to every cluster = no routing / whole-library fallback) has ceiling
1.0 by construction at the cost of pooling all ~92.6k clustered decls.  The question is the
SHAPE of the tradeoff between: does the ceiling climb fast (cheap to buy the hard third) or
does the pool blow up far faster than the ceiling rises (the hard third is intrinsically
expensive, vindicating a-0017's "starker tradeoff under the prover bar")?

Output: results/bridge_retrieval_breadth.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_breadth.json"

K_ROUTES = [0, 1, 2, 3, 5, 10, 20, 50, None]  # None == all clusters (whole-library fallback)


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


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    all_clustered = sum(cluster_size.values())
    neigh_sorted = {c: [d for d, _ in W[c].most_common()] for c in W}

    # cross-domain test theorems, each carrying its needed cluster set
    tests = []
    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem_clusters = {cluster[p] for p in ps if p in cluster}
        if not prem_clusters:
            continue
        if prem_clusters <= {c}:          # all premises in home cluster -> not cross-domain
            continue
        tests.append((c, prem_clusters))
    n = len(tests)

    rows = []
    for K in K_ROUTES:
        covered = 0
        pool_sizes = []
        for c, prem_clusters in tests:
            if K is None:
                routed = None  # all clusters
            else:
                routed = {c} | set(neigh_sorted.get(c, [])[:K])
            # pool size = sum of sizes of routed clusters (home always counted)
            if routed is None:
                psize = all_clustered
                full = True  # every clustered premise is in the whole library
            else:
                psize = sum(cluster_size[cid] for cid in routed)
                full = prem_clusters <= routed
            pool_sizes.append(psize)
            if full:
                covered += 1
        pool_sizes.sort()
        rows.append({
            "k_route": "all" if K is None else K,
            "complete_coverage_ceiling": round(covered / n, 4),
            "mean_pool_size": round(sum(pool_sizes) / n, 1),
            "median_pool_size": pool_sizes[len(pool_sizes) // 2],
            "max_pool_size": pool_sizes[-1],
        })

    # marginal step analysis (ceiling gained per added candidate, between consecutive K)
    steps = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        d_cov = b["complete_coverage_ceiling"] - a["complete_coverage_ceiling"]
        d_pool = b["mean_pool_size"] - a["mean_pool_size"]
        steps.append({
            "from_k": a["k_route"], "to_k": b["k_route"],
            "ceiling_gain": round(d_cov, 4),
            "mean_pool_added": round(d_pool, 1),
            "extra_candidates_per_extra_covered_theorem":
                (round(d_pool / d_cov, 1) if d_cov > 0 else None),
        })

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain test theorems whose ENTIRE clustered premise set is inside the routed pool (max complete-coverage rate achievable at ANY candidate budget for that router breadth)",
        "n_cross_domain_test_theorems": n,
        "total_clustered_decls": all_clustered,
        "router": "home cluster + top-K bridge-neighbour clusters (train-estimated W); K='all' == route to every cluster (whole-library fallback)",
        "leakage_control": "bridge weights W estimated on TRAIN split only; coverage measured on held-out TEST theorems",
        "baseline_e0017": {"k_route": 5, "complete_coverage_ceiling": 0.674,
                           "note": "this script reproduces the K=5 ceiling and extends the sweep"},
        "by_breadth": rows,
        "marginal_steps": steps,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}, "
          f"total_clustered = {res['total_clustered_decls']}")
    print(f"{'K':>4} {'ceiling':>9} {'mean_pool':>10} {'median_pool':>12} {'max_pool':>9}")
    for r in res["by_breadth"]:
        print(f"{str(r['k_route']):>4} {r['complete_coverage_ceiling']:>9} "
              f"{r['mean_pool_size']:>10} {r['median_pool_size']:>12} {r['max_pool_size']:>9}")
    print("\nmarginal steps (extra candidates per extra fully-covered theorem):")
    for s in res["marginal_steps"]:
        print(f"  {str(s['from_k']):>3}->{str(s['to_k']):<3} "
              f"+{s['ceiling_gain']:.4f} ceiling, +{s['mean_pool_added']:.0f} mean pool, "
              f"{s['extra_candidates_per_extra_covered_theorem']} cand/theorem")


if __name__ == "__main__":
    main()
