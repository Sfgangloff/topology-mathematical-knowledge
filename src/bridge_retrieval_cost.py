"""
Candidate-budget control for the bridge-aware premise-retrieval simulation (q-0011).

e-0012 / e-0013 matched the bridge and popularity retrievers at the same *cluster
count* k: both add k extra clusters to the home cluster, and the popularity rule
nearly tied bridge in aggregate cross-domain cluster-recall (<=1.2pt, e-0012),
with bridge winning only on the specialist-needing stratum (e-0013).

But cluster count is NOT a retriever's real cost.  A premise retriever pays per
CANDIDATE PREMISE it has to rank, and the clusters differ enormously in size: the
globally most-cited (popularity) clusters are exactly the largest hubs in Mathlib
(ids 0,1,2,3,5 have ~6k-12k members each), while a theorem-specific bridge
neighbour is typically tiny (median cluster size = 9 nodes).  So "add the top-k
popularity hubs" and "add the top-k bridge neighbours" cost wildly different
numbers of candidate premises.  Matching at cluster-k flatters popularity.

This script re-runs the comparison at matched CANDIDATE BUDGET.  For each method we
walk k = 0,1,2,... adding clusters in the method's ranked order, and at each step
record BOTH:

    * cluster-recall  : fraction of the theorem's distinct premise clusters covered
    * candidate cost  : number of clustered declarations in the selected clusters
                        (= sum of selected cluster sizes; the pool a retriever ranks)

Averaging over held-out cross-domain test theorems gives, per method, a
(mean candidate cost -> mean recall) efficiency curve.  We then read off, by linear
interpolation on that curve, the candidate budget each method needs to reach target
recall levels.  If bridge reaches a given recall at a far SMALLER candidate budget
than popularity, the bridge graph is the cost-efficient router even where cluster-k
made it look like a tie.

Methods: home (k=0 only), bridge(k), popularity(k), random(k) (averaged over seeds).
Leakage control identical to e-0012: W, popularity and cluster sizes that drive the
ranking come from data available before test; recall is measured on TEST theorems.

Output: results/bridge_retrieval_cost.json
"""

import json
import random
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_cost.json"

KMAX = 15                      # walk k = 0..KMAX clusters added to home
RANDOM_TRIALS = 20
RANDOM_SEED = 42
TARGETS = [0.80, 0.90, 0.95]   # recall levels to report budget-to-reach for


def load():
    cluster = json.load(open(CLUSTERS))  # node name -> cluster id (int)
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


def build(cluster, premises, split):
    """W[c][d] and global popularity from TRAIN only; cluster sizes from the
    full clustering (a structural fact known before any test theorem)."""
    W = collections.defaultdict(lambda: collections.Counter())
    pop = collections.Counter()
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
            pop[d] += 1
    size = collections.Counter(cluster.values())
    return W, pop, size


def interp_budget(curve, target):
    """curve: list of (cost, recall) sorted by k (cost non-decreasing). Return the
    candidate cost at which recall first reaches `target`, linearly interpolated
    between the bracketing points; None if the curve never reaches target."""
    for i in range(1, len(curve)):
        c0, r0 = curve[i - 1]
        c1, r1 = curve[i]
        if r1 >= target:
            if r0 >= target:           # already there at the previous point
                return round(c0, 1)
            if r1 == r0 or c1 == c0:
                return round(c1, 1)
            frac = (target - r0) / (r1 - r0)
            return round(c0 + frac * (c1 - c0), 1)
    return None


def evaluate(cluster, premises, split, W, pop, size):
    pop_ranked = [d for d, _ in pop.most_common()]          # global popularity order
    all_clusters = list(size.keys())
    topk_bridge = {c: [d for d, _ in W[c].most_common(KMAX)] for c in W}

    rng = random.Random(RANDOM_SEED)
    random_orders = [rng.sample(all_clusters, len(all_clusters))
                     for _ in range(RANDOM_TRIALS)]

    # accumulators: per method, per k -> sums of recall and candidate cost
    methods = ["bridge", "pop", "rand"]
    rec_sum = {m: [0.0] * (KMAX + 1) for m in methods}
    cost_sum = {m: [0.0] * (KMAX + 1) for m in methods}
    n_cross = 0

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem_clusters = [cluster[p] for p in ps if p in cluster]
        if not prem_clusters:
            continue
        distinct = set(prem_clusters)
        if distinct <= {c}:            # within-domain only; skip
            continue
        n_cross += 1
        total = len(distinct)

        br = topk_bridge.get(c, [])
        pop_extra = [d for d in pop_ranked if d != c]

        def walk(order, acc_rec, acc_cost):
            sel = {c}
            cost = size[c]
            for k in range(KMAX + 1):
                if k > 0:
                    if k - 1 < len(order):
                        d = order[k - 1]
                        if d not in sel:
                            sel.add(d)
                            cost += size[d]
                    # if order exhausted, sel/cost stay flat (method capped out)
                acc_rec[k] += len(distinct & sel) / total
                acc_cost[k] += cost

        walk(br, rec_sum["bridge"], cost_sum["bridge"])
        walk(pop_extra, rec_sum["pop"], cost_sum["pop"])

        # random: average over trials at each k, then add once.
        for k in range(KMAX + 1):
            r_acc = co_acc = 0.0
            for order in random_orders:
                extra = [d for d in order if d != c][:k]
                sel = {c} | set(extra)
                r_acc += len(distinct & sel) / total
                co_acc += sum(size[d] for d in sel)
            rec_sum["rand"][k] += r_acc / RANDOM_TRIALS
            cost_sum["rand"][k] += co_acc / RANDOM_TRIALS

    curves = {}
    budgets = {}
    for m in methods:
        curve = [(round(cost_sum[m][k] / n_cross, 1),
                  round(rec_sum[m][k] / n_cross, 4)) for k in range(KMAX + 1)]
        curves[m] = curve
        budgets[m] = {f"recall>={t}": interp_budget(curve, t) for t in TARGETS}

    return {
        "n_cross_domain_test_theorems": n_cross,
        "note": "curve points are (mean_candidate_cost, mean_cluster_recall) for k=0..KMAX",
        "efficiency_curves": curves,
        "candidate_budget_to_reach_target_recall": budgets,
    }


def main():
    cluster, premises, split = load()
    W, pop, size = build(cluster, premises, split)
    res = evaluate(cluster, premises, split, W, pop, size)
    res["params"] = {"kmax": KMAX, "random_trials": RANDOM_TRIALS,
                     "random_seed": RANDOM_SEED, "targets": TARGETS,
                     "n_clusters": len(size),
                     "largest_cluster_sizes": size.most_common(5)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
