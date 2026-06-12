"""
Is the 'hard third' CONCENTRATED?  A sparse TARGETED escape-cluster whitelist vs uniform
router widening.  (q-0011, follow-on to e-0018/a-0018 and e-0020/a-0019)

e-0018 (a-0018) established that under the prover-relevant COMPLETE-COVERAGE bar, the
home + top-5 bridge router caps at a 0.674 ceiling: ~1/3 of cross-domain test theorems need
at least one premise OUTSIDE the top-5 router.  Widening the router UNIFORMLY (top-K) buys
that hard third back only at steeply rising, ultimately whole-library candidate cost, and
e-0020 (a-0019) showed conditioning the breadth on the home cluster helps only modestly.

But uniform widening adds the next bridge-neighbour clusters as a PREFIX of the W-ranking.
It never asks the sharper question this script tests:

  When a cross-domain theorem ESCAPES its top-5 router, WHERE do the escaped premises live?
  Are those escape-target clusters concentrated in a SMALL, train-predictable set (so a
  fixed sparse whitelist recovers the hard third cheaply), or diffuse across the library
  (so a-0018's whole-library verdict is fundamental)?

Design (held identical to e-0018: same data, same 432 held-out cross-domain TEST theorems,
same train-only bridge weights W, same exact ranking-free complete-coverage ceiling metric;
ONLY the routing policy changes):

  1. Descriptive concentration of the hard third.  For each TEST theorem that escapes its
     home+top-5 router, record its escaped target clusters E = prem_clusters - (home+top5).
     Report |E| distribution and the global concentration curve: how many distinct clusters
     the union of escapes spans, and what fraction of (theorem, escaped-cluster) demand the
     N most-frequent escape-target clusters cover.

  2. Leakage-safe TARGETED router.  Estimate, on the TRAIN split only, a GLOBAL ranking of
     escape-target clusters by how often a train cross-domain theorem's premises escape its
     own home+top-5 router into that cluster.  The targeted policy adds the top-M of this
     fixed whitelist to EVERY theorem's home+top-5 pool, and we sweep M.  Because it is a
     fixed global whitelist (not a per-home prefix), it directly tests concentration: if the
     hard third concentrates, a small M lifts the ceiling at a small fixed pool cost.

  3. Head-to-head.  Put the targeted sweep (ceiling vs mean pool) next to the uniform
     breadth sweep from e-0018 and report, at matched complete-coverage ceiling, which
     reaches it with fewer candidate decls.

Output: results/bridge_retrieval_targeted.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_targeted.json"

K_BASE = 5                               # fixed bridge router breadth (the e-0018 baseline)
M_WHITELIST = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50]   # extra TARGETED escape-clusters added


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


def cross_domain_set(premises, split, cluster, which):
    """Return list of (home, prem_clusters) for cross-domain theorems in the given split."""
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
        out.append((c, prem_clusters))
    return out


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    all_clustered = sum(cluster_size.values())
    top5 = {c: set([d for d, _ in W[c].most_common(K_BASE)]) for c in W}

    def router_base(c):
        return {c} | top5.get(c, set())

    # ---- 1. estimate global escape-target whitelist on TRAIN ----
    train = cross_domain_set(premises, split, cluster, "train")
    escape_freq = collections.Counter()   # how many TRAIN theorems escape into cluster d
    for c, prem_clusters in train:
        esc = prem_clusters - router_base(c)
        for d in esc:
            escape_freq[d] += 1
    whitelist = [d for d, _ in escape_freq.most_common()]   # fixed, global, train-only

    # ---- 2/3. TEST evaluation ----
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    # descriptive concentration of the hard third on TEST (escape relative to home+top5)
    hard = []                      # (home, escaped_target_clusters)
    test_escape_freq = collections.Counter()
    esc_sizes = []
    for c, prem_clusters in test:
        esc = prem_clusters - router_base(c)
        if esc:
            hard.append((c, esc))
            esc_sizes.append(len(esc))
            for d in esc:
                test_escape_freq[d] += 1
    n_hard = len(hard)
    total_demand = sum(test_escape_freq.values())   # (theorem, escaped-cluster) pairs
    distinct_escape_clusters = len(test_escape_freq)
    # concentration curve: cumulative demand covered by the N most-frequent escape clusters
    ranked = [cnt for _, cnt in test_escape_freq.most_common()]
    cum = 0
    conc_curve = []
    for i, cnt in enumerate(ranked, 1):
        cum += cnt
        if i in (1, 3, 5, 10, 20, 30, 50) or i == len(ranked):
            conc_curve.append({"top_n_escape_clusters": i,
                               "frac_escape_demand_covered": round(cum / total_demand, 4)})

    # targeted router sweep: home + top5 + top-M of the TRAIN whitelist (fixed for all)
    targeted_rows = []
    for M in M_WHITELIST:
        wl = set(whitelist[:M])
        covered = 0
        pool_sizes = []
        for c, prem_clusters in test:
            routed = router_base(c) | wl
            pool_sizes.append(sum(cluster_size[cid] for cid in routed))
            if prem_clusters <= routed:
                covered += 1
        pool_sizes.sort()
        targeted_rows.append({
            "m_whitelist": M,
            "complete_coverage_ceiling": round(covered / n, 4),
            "mean_pool_size": round(sum(pool_sizes) / n, 1),
            "median_pool_size": pool_sizes[len(pool_sizes) // 2],
            "max_pool_size": pool_sizes[-1],
        })

    # ---- uniform breadth reference (same metric, prefix of W per home cluster) ----
    K_ROUTES = [5, 10, 20, 50, None]
    neigh_sorted = {c: [d for d, _ in W[c].most_common()] for c in W}
    uniform_rows = []
    for K in K_ROUTES:
        covered = 0
        pool_sizes = []
        for c, prem_clusters in test:
            if K is None:
                pool_sizes.append(all_clustered)
                covered += 1
            else:
                routed = {c} | set(neigh_sorted.get(c, [])[:K])
                pool_sizes.append(sum(cluster_size[cid] for cid in routed))
                if prem_clusters <= routed:
                    covered += 1
        pool_sizes.sort()
        uniform_rows.append({
            "k_route": "all" if K is None else K,
            "complete_coverage_ceiling": round(covered / n, 4),
            "mean_pool_size": round(sum(pool_sizes) / n, 1),
        })

    # head-to-head: cheapest mean pool to reach a target ceiling, targeted vs uniform
    def cheapest(rows, key, target):
        best = None
        for r in rows:
            if r["complete_coverage_ceiling"] >= target:
                if best is None or r["mean_pool_size"] < best["mean_pool_size"]:
                    best = r
        return None if best is None else {key: best[key],
                                          "ceiling": best["complete_coverage_ceiling"],
                                          "mean_pool_size": best["mean_pool_size"]}
    head2head = []
    for t in (0.70, 0.75, 0.80, 0.85, 0.90):
        head2head.append({
            "target_ceiling": t,
            "targeted": cheapest(targeted_rows, "m_whitelist", t),
            "uniform": cheapest(uniform_rows, "k_route", t),
        })

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed pool",
        "n_cross_domain_test_theorems": n,
        "total_clustered_decls": all_clustered,
        "k_base": K_BASE,
        "leakage_control": "bridge weights W and the escape-target whitelist both estimated on TRAIN only; coverage measured on held-out TEST theorems",
        "hard_third_concentration": {
            "note": "escape = premise clusters of a TEST theorem outside its home+top5 router",
            "n_hard_third_theorems": n_hard,
            "frac_hard_third": round(n_hard / n, 4),
            "escaped_clusters_per_theorem_mean": round(sum(esc_sizes) / max(1, len(esc_sizes)), 3),
            "escaped_clusters_per_theorem_median": (sorted(esc_sizes)[len(esc_sizes) // 2] if esc_sizes else 0),
            "distinct_escape_target_clusters": distinct_escape_clusters,
            "total_escape_demand_pairs": total_demand,
            "concentration_curve": conc_curve,
        },
        "targeted_sweep": targeted_rows,
        "uniform_breadth_reference": uniform_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    hc = res["hard_third_concentration"]
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"hard third = {hc['n_hard_third_theorems']} ({hc['frac_hard_third']})")
    print(f"escaped clusters/theorem: mean {hc['escaped_clusters_per_theorem_mean']} "
          f"median {hc['escaped_clusters_per_theorem_median']}; "
          f"{hc['distinct_escape_target_clusters']} distinct escape-target clusters")
    print("concentration (top-N escape clusters cover X of escape demand):")
    for p in hc["concentration_curve"]:
        print(f"   top-{p['top_n_escape_clusters']:>3}: {p['frac_escape_demand_covered']}")
    print(f"\n{'M':>4} {'ceiling':>9} {'mean_pool':>10}   (targeted: home+top5 + top-M whitelist)")
    for r in res["targeted_sweep"]:
        print(f"{r['m_whitelist']:>4} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>10}")
    print(f"\n{'K':>4} {'ceiling':>9} {'mean_pool':>10}   (uniform breadth reference)")
    for r in res["uniform_breadth_reference"]:
        print(f"{str(r['k_route']):>4} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>10}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        t = h["targeted"]; u = h["uniform"]
        ts = f"M={t['m_whitelist']} pool={t['mean_pool_size']}" if t else "unreachable"
        us = f"K={u['k_route']} pool={u['mean_pool_size']}" if u else "unreachable"
        print(f"   ceiling>={h['target_ceiling']}: targeted[{ts}]  uniform[{us}]")


if __name__ == "__main__":
    main()
