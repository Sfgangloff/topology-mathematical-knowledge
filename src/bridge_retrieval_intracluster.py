"""
Intra-cluster premise ranking: the "missing piece" from e-0014 (q-0011).

e-0011..e-0014 routed premise retrieval at CLUSTER granularity: a retriever picks
the home cluster plus some bridge-neighbour clusters, and the candidate pool is the
UNION OF WHOLE CLUSTERS.  e-0014 found that, accounted at true retrieval cost
(number of candidate declarations to rank), cluster-level routing is intrinsically
candidate-heavy: reaching 0.90 cross-domain cluster-recall needs ~60k of 92.6k
clustered declarations (~64% of Mathlib), because the bridge / popularity routers
both converge on the same huge hub clusters.  Its closing line:

    "...intra-cluster premise ranking is the missing piece."

This script tests exactly that.  We FIX the cluster routing (bridge: home + top-k
bridge neighbours, the e-0011 router) and then, INSIDE the selected clusters, instead
of dumping every member into the candidate pool we RANK individual declarations by a
leakage-safe train-citation popularity prior (how many distinct TRAIN theorems cite
each declaration as a premise) and add them one at a time.  We measure, per held-out
cross-domain TEST theorem, PREMISE recall (fraction of the theorem's actual premises
present in the candidate pool) as a function of candidate budget B = number of
individual declarations ranked.

Three arms, all over the SAME bridge-routed candidate set (so the only thing varied
is intra-cluster ordering):

    * whole-pool : take every declaration in the selected clusters (e-0014 accounting;
                   its candidate cost is the size of the union, its recall the ceiling).
    * popularity : add declarations in train-citation-popularity order (the prior).
    * random     : add declarations in random order (control: does the popularity
                   prior actually help, or is any ordering as good?).  A random prefix
                   of size B over a pool of N decls holding `hits` of the theorem's
                   recoverable premises contains, in expectation, hits*min(B,N)/N of
                   them, so the random curve is the exact hypergeometric mean
                   recall_rand(B) = ceiling * min(B, pool)/pool -- no sampling needed.

If popularity ranking reaches the whole-pool recall ceiling at a candidate budget far
SMALLER than the union size, intra-cluster ranking IS the cheap lever e-0014 was
missing, and the operational payoff of bridge structure stops being "visit the right
clusters sooner" and becomes "retrieve the right premises cheaply".  If instead the
theorems' premises live in the rarely-cited tail, popularity ranking won't help and
retrieval is intrinsically expensive even within clusters -- an equally honest result.

Leakage control: W (bridge weights) and the per-declaration citation prior come from
the TRAIN split only; recall is measured on TEST theorems.

Output: results/bridge_retrieval_intracluster.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_intracluster.json"

K_ROUTE = 5                       # bridge router: home + top-5 bridge-neighbour clusters
# candidate-budget grid (number of individual declarations the retriever ranks)
BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
TARGETS = [0.70, 0.80, 0.90]      # absolute premise-recall levels to report budget-to-reach


def load():
    cluster = json.load(open(CLUSTERS))            # decl name -> cluster id (int)
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
    """From TRAIN only: bridge weights W[c][d] and per-declaration citation prior
    cite[p] = number of distinct train theorems that cite p as a premise."""
    W = collections.defaultdict(lambda: collections.Counter())
    cite = collections.Counter()
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        c = cluster.get(fn)
        for p in ps:
            cite[p] += 1                            # popularity prior over declarations
        if c is None:
            continue
        seen = set()
        for p in ps:
            d = cluster.get(p)
            if d is None or d == c or d in seen:
                continue
            seen.add(d)
            W[c][d] += 1
    size = collections.Counter(cluster.values())
    return W, cite, size


def interp_budget(curve, target):
    """curve: list of (cost, recall) with cost non-decreasing. Candidate cost at which
    recall first reaches target (linear interpolation); None if never reached."""
    for i in range(1, len(curve)):
        c0, r0 = curve[i - 1]
        c1, r1 = curve[i]
        if r1 >= target:
            if r0 >= target:
                return round(c0, 1)
            if r1 == r0 or c1 == c0:
                return round(c1, 1)
            frac = (target - r0) / (r1 - r0)
            return round(c0 + frac * (c1 - c0), 1)
    return None


def evaluate(cluster, premises, split, W, cite, size):
    # group declarations by cluster, each cluster's members sorted by popularity prior
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0))
                  for cid, ms in members.items()}

    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    rec_pop = [0.0] * len(BUDGETS)
    rec_rand = [0.0] * len(BUDGETS)
    cost_pop = [0.0] * len(BUDGETS)        # min(B, pool size) averaged
    pool_size_sum = 0.0                    # whole-pool candidate cost (e-0014 accounting)
    ceiling_sum = 0.0                      # premise recall when whole pool taken
    n_cross = 0

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        prem_clusters = {cluster[p] for p in prem}
        if prem_clusters <= {c}:           # within-domain only; skip
            continue
        n_cross += 1
        total = len(prem)

        selected = {c} | set(topk_bridge.get(c, []))
        # popularity-ranked candidate list over the selected clusters
        ranked = []
        for cid in selected:
            ranked.extend(pop_sorted.get(cid, []))
        ranked.sort(key=lambda n: -cite.get(n, 0))
        pool = len(ranked)
        pool_size_sum += pool
        # ceiling: premises recoverable at all from this routing (whole pool taken)
        ceiling_sum += len(prem.intersection(ranked)) / total

        prem_set = prem
        ceiling_hits = len(prem_set.intersection(ranked))   # premises recoverable here
        for bi, B in enumerate(BUDGETS):
            topB = ranked[:B]
            rec_pop[bi] += len(prem_set.intersection(topB)) / total
            cost_pop[bi] += min(B, pool)
            # random arm: exact hypergeometric mean over a random prefix of size B
            frac = (min(B, pool) / pool) if pool else 0.0
            rec_rand[bi] += (ceiling_hits * frac) / total

    pop_curve = [(round(cost_pop[bi] / n_cross, 1), round(rec_pop[bi] / n_cross, 4))
                 for bi in range(len(BUDGETS))]
    rand_curve = [(round(cost_pop[bi] / n_cross, 1), round(rec_rand[bi] / n_cross, 4))
                  for bi in range(len(BUDGETS))]
    mean_pool = round(pool_size_sum / n_cross, 1)
    ceiling = round(ceiling_sum / n_cross, 4)

    return {
        "n_cross_domain_test_theorems": n_cross,
        "router": f"bridge: home + top-{K_ROUTE} bridge-neighbour clusters",
        "mean_whole_pool_candidate_cost": mean_pool,
        "whole_pool_premise_recall_ceiling": ceiling,
        "note": "curve points are (mean_candidate_cost, mean_premise_recall) for the budget grid",
        "budget_grid": BUDGETS,
        "popularity_curve": pop_curve,
        "random_curve": rand_curve,
        "popularity_budget_to_reach": {
            f"recall>={t}": interp_budget(pop_curve, t) for t in TARGETS},
        "random_budget_to_reach": {
            f"recall>={t}": interp_budget(rand_curve, t) for t in TARGETS},
    }


def main():
    cluster, premises, split = load()
    W, cite, size = build(cluster, premises, split)
    res = evaluate(cluster, premises, split, W, cite, size)
    res["params"] = {"k_route": K_ROUTE, "budget_grid": BUDGETS,
                     "targets": TARGETS, "n_clusters": len(size),
                     "n_distinct_cited_decls_train": len(cite)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
