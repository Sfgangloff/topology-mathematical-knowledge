"""
Cross-domain premise BURDEN and router REACHABILITY: the structural ceiling on
bridge routing's complete-coverage payoff. (q-0011; explains WHY a-0025's oracle
gate gain saturates so low.)

The whole a-0013..a-0025 chain measured downstream coverage curves but never the
STRUCTURE of what a cross-domain proof actually needs.  a-0025's decisive bound was
that even a PERFECT cross-domain gate buys only complete coverage (cross-domain
B=1000: +0.048 over home_only), not recall.  This script asks why that ceiling is
where it is, with a pure distributional pass over the held-out cross-domain TEST set
-- no ranking, no windows, so it isolates the COMBINATORIAL ceiling that sits ABOVE
any ranker.

For each cross-domain TEST theorem (clustered premises leave its home cluster):
  - n_ooh        : # out-of-home clustered premises
  - n_foreign    : # DISTINCT foreign clusters those premises live in
  - The complete-coverage requirement is structural: a top-K bridge router can route
    a goal to home + its K most-frequent neighbour clusters (TRAIN-learned W, e-0011
    machinery).  Complete coverage of a cross-domain goal is POSSIBLE only if EVERY
    one of its foreign clusters is among the home's top-K routed neighbours.  If even
    one needed cluster is unroutable, NO window B and no ranker can complete it.

So the router's reachability rate is a HARD upper bound on bridge's complete-coverage
gain, independent of B and of the content ranker.  We report:
  - burden distribution (n_ooh, n_foreign): how concentrated is cross-domain need?
  - router_complete@K : frac of cross-domain goals whose ALL foreign clusters are in
                        the home's top-K neighbours  (the structural ceiling)
  - router_any@K      : frac with >=1 foreign cluster routed (partial reachability)
  - swept over K in {1,5,10,20,50} to show the breadth/ceiling trade (cf. a-0018)

Leakage note: home cluster is taken as given here (same as the rest of the chain;
a-0023 showed it is name-predictable and the leakage is modest).  W is TRAIN-only.
This is a premise-availability upper bound, not a measured prover success rate.

Output: results/bridge_retrieval_burden.json
"""

import json
import collections
import statistics
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_burden.json"

K_SWEEP = [1, 5, 10, 20, 50]


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


def build_router(cluster, premises, split):
    """home cluster -> Counter of neighbour cluster frequencies, TRAIN only (e-0011 W)."""
    W = collections.defaultdict(collections.Counter)
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        for p in ps:
            d = cluster.get(p)
            if d is None or d == c:
                continue
            W[c][d] += 1
    return W


def main():
    cluster, premises, split = load()
    W = build_router(cluster, premises, split)
    topk_cache = {k: {c: {d for d, _ in nb.most_common(k)} for c, nb in W.items()}
                  for k in K_SWEEP}

    n_test = 0
    n_cross = 0
    n_ooh_list = []
    n_foreign_list = []
    single_premise = 0   # exactly 1 out-of-home premise
    single_foreign = 0   # exactly 1 distinct foreign cluster
    # router reachability counters, per K
    router_complete = {k: 0 for k in K_SWEEP}
    router_any = {k: 0 for k in K_SWEEP}
    # routable foreign-cluster fraction (micro) per K
    foreign_total = 0
    foreign_routed = {k: 0 for k in K_SWEEP}

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        n_test += 1
        ooh = [p for p in prem if cluster[p] != c]
        if not ooh:
            continue  # within-domain
        n_cross += 1
        foreign = {cluster[p] for p in ooh}
        n_ooh_list.append(len(ooh))
        n_foreign_list.append(len(foreign))
        single_premise += int(len(ooh) == 1)
        single_foreign += int(len(foreign) == 1)
        foreign_total += len(foreign)
        for k in K_SWEEP:
            routed = topk_cache[k].get(c, set())
            hit = foreign & routed
            foreign_routed[k] += len(hit)
            if foreign <= routed:
                router_complete[k] += 1
            if hit:
                router_any[k] += 1

    def dist(xs):
        xs = sorted(xs)
        return {
            "mean": round(statistics.mean(xs), 3),
            "median": statistics.median(xs),
            "p90": xs[int(0.90 * (len(xs) - 1))],
            "max": xs[-1],
            "frac_eq1": round(sum(v == 1 for v in xs) / len(xs), 4),
            "frac_le2": round(sum(v <= 2 for v in xs) / len(xs), 4),
            "frac_le3": round(sum(v <= 3 for v in xs) / len(xs), 4),
        }

    res = {
        "metric_note": "structural cross-domain premise burden and top-K router "
                       "reachability over held-out cross-domain TEST theorems; "
                       "router_complete@K is a HARD ceiling on bridge complete-coverage "
                       "(independent of window B and content ranker)",
        "n_test": n_test,
        "n_cross_domain": n_cross,
        "frac_cross_domain": round(n_cross / n_test, 4),
        "out_of_home_premise_count": dist(n_ooh_list),
        "distinct_foreign_cluster_count": dist(n_foreign_list),
        "frac_single_ooh_premise": round(single_premise / n_cross, 4),
        "frac_single_foreign_cluster": round(single_foreign / n_cross, 4),
        "router_complete_at_k": {
            str(k): round(router_complete[k] / n_cross, 4) for k in K_SWEEP},
        "router_any_at_k": {
            str(k): round(router_any[k] / n_cross, 4) for k in K_SWEEP},
        "foreign_cluster_routed_micro_at_k": {
            str(k): round(foreign_routed[k] / foreign_total, 4) for k in K_SWEEP},
        "k_sweep": K_SWEEP,
        "interpretation": (
            "router_complete@K bounds the fraction of cross-domain goals for which "
            "bridge can EVER achieve complete coverage; the a-0025 oracle-gate "
            "complete-coverage gain cannot exceed this. Gap between router_any and "
            "router_complete = goals reachable only PARTIALLY (some needed cluster "
            "off-router)."),
    }
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
