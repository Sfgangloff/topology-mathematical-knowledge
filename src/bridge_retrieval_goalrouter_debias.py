"""
De-biasing the goal-specific router's target predictor (q-0011, follow-on to e-0030/a-0027).

e-0030 (a-0027) showed a GOAL-SPECIFIC router -- home + the top-m foreign clusters predicted
from the goal's NAME tokens -- beats the static top-K bridge router on the complete-coverage
ceiling, but only PARTIALLY realizes the concentrated-need advantage e-0029 found: an ORACLE
goal-specific router reaches 0.93 complete coverage at ~21.8k candidate decls (24% of the
library) while the realizable name-token predictor needs ~72k -- a ~4x gap.  e-0030 diagnosed
the cause and named the lever verbatim:

    "...the name-token predictor captures only part of that gap because it has a HUB-BIAS
     toward large token-associated clusters instead of the small specialized clusters a
     cross-domain goal needs ... a less hub-biased target predictor ... is the remaining lever."

The e-0030 predictor scores a foreign cluster d by  sum_t IDF(t) * P(d|t),  P(d|t)=n(t,d)/n(t).
P(d|t) is hub-biased: a globally-popular foreign cluster d (large global demand n(d)) co-occurs
with almost every token, so it scores high for many goals regardless of specificity.  This
experiment tests the named lever with NO new data: divide P(d|t) by the cluster's TRAIN demand
prior P(d) (a pointwise-mutual-information / lift normalization), so the router targets clusters
SPECIFICALLY associated with the goal's tokens rather than popular hubs.

    score_lift(d) = sum_t IDF(t) * P(d|t) / P(d)^alpha,   P(d) = n(d) / sum_d' n(d').

alpha=0 recovers the e-0030 baseline (hub-biased); alpha=1 is full lift (PMI numerator).
A pure log-PMI variant  sum_t IDF(t) * log(P(d|t)/P(d))  is also reported.

Metric (identical to e-0029/e-0030): complete-coverage CEILING = fraction of cross-domain TEST
theorems whose ENTIRE clustered premise set lands inside the routed cluster pool.  Exact,
ranking-free, an upper bound above any intra-cluster ranker.  Candidate cost = mean sum of
routed cluster sizes.  Same 432 held-out cross-domain TEST theorems, same train-only weights and
leakage control as the whole chain; the ONLY query signal is the test theorem's own NAME.

Arms:
  * goal_base(m)   : e-0030 predictor, score = sum_t IDF(t) * P(d|t)            (alpha=0).
  * goal_lift(m)   : score = sum_t IDF(t) * P(d|t) / P(d)^alpha, alpha in {0.5,1.0}.
  * goal_pmi(m)    : score = sum_t IDF(t) * log(P(d|t)/P(d))   (only over d with n(t,d)>0).
  * oracle(m)      : home + the goal's m most-demanded ACTUAL foreign clusters (leaky bound).

Output: results/bridge_retrieval_goalrouter_debias.json
"""

import json
import math
import collections
from pathlib import Path

from src.bridge_retrieval_goalrouter import (
    load, build_idf, cross_domain_set, train_foreign_predictor, tokenize, sweep,
)

RESULTS = Path(__file__).parent.parent / "results"
OUT = RESULTS / "bridge_retrieval_goalrouter_debias.json"

M_SWEEP = [1, 2, 3, 5, 10, 20]
ALPHAS = [0.0, 0.5, 1.0]


def predict_lift(fn, tok_cluster, tok_total, idf, stop, exclude, log_prior, alpha):
    """Rank foreign clusters by sum_t IDF(t) * P(d|t) / P(d)^alpha.

    log_prior[d] = log P(d) (train demand prior). alpha=0 -> e-0030 baseline.
    """
    score = collections.Counter()
    for t in (tokenize(fn) - stop) & idf.keys():
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for d, ntd in tok_cluster[t].items():
            if d in exclude:
                continue
            pdt = ntd / nt
            score[d] += w * pdt * math.exp(-alpha * log_prior[d])
    return [d for d, _ in score.most_common()]


def predict_pmi(fn, tok_cluster, tok_total, idf, stop, exclude, log_prior):
    """Rank foreign clusters by sum_t IDF(t) * log(P(d|t)/P(d)), only over observed (t,d)."""
    score = collections.Counter()
    for t in (tokenize(fn) - stop) & idf.keys():
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for d, ntd in tok_cluster[t].items():
            if d in exclude:
                continue
            pmi = math.log(ntd / nt) - log_prior[d]
            score[d] += w * pmi
    return [d for d, _ in score.most_common()]


def hit_stats(test, pred_cache):
    """top-k hits a foreign cluster / covers ALL foreign clusters."""
    n = len(test)
    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    full = {1: 0, 2: 0, 3: 0, 5: 0}
    for fn, c, prem_clusters in test:
        foreign = prem_clusters - {c}
        pred = pred_cache[fn]
        for k in hits:
            if set(pred[:k]) & foreign:
                hits[k] += 1
            if foreign <= set(pred[:k]):
                full[k] += 1
    return ({f"top{k}_hits_a_foreign_cluster": round(v / n, 4) for k, v in hits.items()},
            {f"top{k}_covers_ALL_foreign_clusters": round(v / n, 4) for k, v in full.items()})


def cheapest(rows, target):
    best = None
    for r in rows:
        if r["complete_coverage_ceiling"] >= target:
            if best is None or r["mean_pool_size"] < best["mean_pool_size"]:
                best = r
    return None if best is None else {"breadth": best["breadth"],
                                      "ceiling": best["complete_coverage_ceiling"],
                                      "mean_pool_size": best["mean_pool_size"]}


def evaluate(cluster, premises, split):
    cluster_size = collections.Counter(cluster.values())
    idf, stop = build_idf(cluster)
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    tok_cluster, tok_total, global_demand = train_foreign_predictor(train, idf, stop)
    total_demand = sum(global_demand.values())
    # log P(d) over the global foreign-cluster demand prior (train only).
    log_prior = {d: math.log(cnt / total_demand) for d, cnt in global_demand.items()}

    arms = {}          # arm name -> sweep rows
    pred_caches = {}   # arm name -> {fn: ranked foreign clusters}

    # baseline + lift arms
    for alpha in ALPHAS:
        name = "goal_base" if alpha == 0.0 else f"goal_lift_a{alpha}"
        cache = {fn: predict_lift(fn, tok_cluster, tok_total, idf, stop, {c}, log_prior, alpha)
                 for fn, c, _ in test}
        pred_caches[name] = cache
        arms[name] = sweep(test, cluster_size,
                           lambda fn, c, pc, m, _cache=cache: {c} | set(_cache[fn][:m]),
                           M_SWEEP)

    # log-PMI arm
    cache_pmi = {fn: predict_pmi(fn, tok_cluster, tok_total, idf, stop, {c}, log_prior)
                 for fn, c, _ in test}
    pred_caches["goal_pmi"] = cache_pmi
    arms["goal_pmi"] = sweep(test, cluster_size,
                             lambda fn, c, pc, m: {c} | set(cache_pmi[fn][:m]),
                             M_SWEEP)

    # oracle arm (leaky upper bound, same as e-0030)
    def route_oracle(fn, c, pc, m):
        foreign = sorted(pc - {c}, key=lambda d: -global_demand[d])
        return {c} | set(foreign[:m])
    arms["oracle"] = sweep(test, cluster_size, route_oracle, M_SWEEP)

    # predictor accuracy per arm
    pred_acc = {}
    for name, cache in pred_caches.items():
        h, f = hit_stats(test, cache)
        pred_acc[name] = {"hits": h, "full_coverage": f}

    # head-to-head: cheapest mean pool to reach each target, all goal arms vs oracle
    goal_arms = ["goal_base", "goal_lift_a0.5", "goal_lift_a1.0", "goal_pmi"]
    head2head = []
    for t in (0.50, 0.674, 0.70, 0.80, 0.90, 0.95):
        row = {"target_ceiling": t, "oracle": cheapest(arms["oracle"], t)}
        for ga in goal_arms:
            row[ga] = cheapest(arms[ga], t)
        # best realizable goal arm at this target
        cands = [(ga, row[ga]) for ga in goal_arms if row[ga]]
        if cands:
            best = min(cands, key=lambda x: x[1]["mean_pool_size"])
            row["best_realizable"] = {"arm": best[0], **best[1]}
            if row["oracle"]:
                row["best_realizable_over_oracle_pool_ratio"] = round(
                    best[1]["mean_pool_size"] / row["oracle"]["mean_pool_size"], 2)
        head2head.append(row)

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free)",
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "total_clustered_decls": sum(cluster_size.values()),
        "leakage_control": "IDF, token x foreign-cluster co-occurrence counts, and the demand prior P(d) ALL estimated on TRAIN only; coverage on held-out TEST; only query signal is the test theorem's own NAME.",
        "predictor_accuracy": pred_acc,
        "sweeps": arms,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"train_cross = {res['n_train_cross_domain']}")
    print("\npredictor top-k hits a foreign cluster:")
    for name, acc in res["predictor_accuracy"].items():
        print(f"  {name:>16}: {acc['hits']}")
    for arm, rows in res["sweeps"].items():
        print(f"\n{arm}:  {'b':>3} {'ceiling':>9} {'mean_pool':>11} {'med_pool':>9}")
        for r in rows:
            print(f"   {r['breadth']:>3} {r['complete_coverage_ceiling']:>9} "
                  f"{r['mean_pool_size']:>11} {r['median_pool_size']:>9}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"b={x['breadth']} pool={x['mean_pool_size']}" if x else "unreachable"
        br = h.get("best_realizable")
        brtxt = (f"  BEST={br['arm']}(b={br['breadth']},pool={br['mean_pool_size']})"
                 f" x{h.get('best_realizable_over_oracle_pool_ratio')} oracle" if br else "")
        print(f"   >= {h['target_ceiling']}: base[{fmt(h['goal_base'])}] "
              f"lift0.5[{fmt(h['goal_lift_a0.5'])}] lift1.0[{fmt(h['goal_lift_a1.0'])}] "
              f"pmi[{fmt(h['goal_pmi'])}] oracle[{fmt(h['oracle'])}]{brtxt}")


if __name__ == "__main__":
    main()
