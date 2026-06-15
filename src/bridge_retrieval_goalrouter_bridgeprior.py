"""
Fuse the goal-router's name-token target predictor with the HOME-SPECIFIC bridge
prior W[home][.]  (q-0011, follow-on to e-0030/a-0027 and e-0031/a-0028).

a-0027 (e-0030) found the pure goal-specific router -- which ranks foreign clusters
ONLY by name-token evidence  score(d) = sum_t IDF(t) * P(d|t)  -- has a HUB-BIAS: it
routes large, globally-popular token-associated clusters rather than the small
specialized cluster a given cross-domain goal actually needs, leaving a ~3-4x gap to
the oracle goal-router (oracle 0.93 ceiling at ~21.8k candidates vs realizable ~72k).

a-0028 (e-0031) tried to de-bias by dividing P(d|t) by the GLOBAL demand prior P(d)
(a PMI/lift normalization).  Mild deflation (alpha=0.5) bought a free ~6-10% pool cut,
but FULL lift / log-PMI COLLAPSED -- over-correcting toward tiny rarely-demanded
clusters.  a-0028 closed: "the dominant residual gap ... is the predictor's RANKING
quality (which specific small specialized cluster), not the popularity weighting."

This experiment tests the one structural prior neither pure arm used: the HOME-SPECIFIC
bridge weight W[home][d] (how often a TRAIN proof in the goal's home cluster cites a
premise in cluster d).  Unlike the global demand prior P(d) (which collapses, e-0031)
and unlike the global-popularity router (a-0027), W[home][.] is SPARSE and
HOME-CONDITIONED: it encodes which foreign clusters THIS home domain actually bridges
to.  Fusing it with the goal's name-token evidence should sharpen ranking toward
home-plausible foreign clusters without the global-hub bias.

    score_fused(d) = name_evidence(d) * (W[home][d] + s)^beta

  * beta = 0  ->  pure name predictor  (reproduces e-0030 goal arm exactly).
  * beta > 0  ->  multiply in the home-specific bridge prior; smoothing s keeps
                  never-bridged clusters reachable (they are not hard-zeroed, so the
                  router can still escape the home's historical neighbours).

Everything else is held IDENTICAL to e-0030/e-0031: same 432 held-out cross-domain
TEST theorems, same train-only W / IDF / token x foreign-cluster co-occurrence, same
exact ranking-free complete-coverage CEILING metric (every needed cluster must be
routed), same head-to-head cheapest-pool comparison, same leaky oracle reference.

Leakage control: W, IDF, the token x foreign-cluster co-occurrence counts and the
global demand ranking are ALL estimated on TRAIN only; coverage is measured on
held-out TEST theorems; the only query signal is the test theorem's own NAME.

Output: results/bridge_retrieval_goalrouter_bridgeprior.json
"""

import json
import collections
from pathlib import Path

from bridge_retrieval_goalrouter import (
    tokenize, load, build_bridge_weights, cross_domain_set,
    build_idf, train_foreign_predictor, sweep, M_SWEEP,
)

RESULTS = Path(__file__).parent.parent / "results"
OUT = RESULTS / "bridge_retrieval_goalrouter_bridgeprior.json"

BETA_SWEEP = [0.0, 0.5, 1.0, 2.0]   # 0.0 == pure name predictor (e-0030 reproduction)
SMOOTH = 0.5                         # additive smoothing on W[home][d] so never-bridged
                                     # clusters stay reachable (not hard-zeroed)


def name_evidence(fn, tok_cluster, tok_total, idf, stop, exclude):
    """Raw name-token score per candidate foreign cluster: sum_t IDF(t) * P(d|t)."""
    score = collections.Counter()
    for t in (tokenize(fn) - stop) & idf.keys():
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for d, ntd in tok_cluster[t].items():
            if d in exclude:
                continue
            score[d] += w * (ntd / nt)
    return score


def ranked_fused(ev, home, W, beta):
    """Rank candidate foreign clusters by name_evidence * (W[home][d]+s)^beta."""
    if beta == 0.0:
        return [d for d, _ in ev.most_common()]
    wh = W.get(home, {})
    fused = {}
    for d, s in ev.items():
        fused[d] = s * ((wh.get(d, 0) + SMOOTH) ** beta)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    idf, stop = build_idf(cluster)
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    tok_cluster, tok_total, global_demand = train_foreign_predictor(train, idf, stop)

    # name-token evidence (excludes only the home cluster) -- shared across all beta arms
    ev_cache = {fn: name_evidence(fn, tok_cluster, tok_total, idf, stop, {c})
                for fn, c, _ in test}

    arms = {}      # beta -> sweep rows
    pred_acc = {}  # beta -> top-k accuracy
    for beta in BETA_SWEEP:
        pred_cache = {fn: ranked_fused(ev_cache[fn], c, W, beta)
                      for fn, c, _ in test}

        def route_goal(fn, c, prem_clusters, m, _pc=pred_cache):
            return {c} | set(_pc[fn][:m])
        arms[beta] = sweep(test, cluster_size, route_goal, M_SWEEP)

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
        pred_acc[beta] = {
            "top_k_hits_a_foreign_cluster": {k: round(v / n, 4) for k, v in hits.items()},
            "top_k_covers_ALL_foreign_clusters": {k: round(v / n, 4) for k, v in full.items()},
        }

    # ---- oracle reference (home + the goal's m most-demanded ACTUAL foreign clusters) ----
    def route_oracle(fn, c, prem_clusters, m):
        foreign = sorted(prem_clusters - {c}, key=lambda d: -global_demand[d])
        return {c} | set(foreign[:m])
    oracle_rows = sweep(test, cluster_size, route_oracle, M_SWEEP)

    # ---- head-to-head: cheapest mean pool to reach each target ceiling ----
    def cheapest(rows, target):
        best = None
        for r in rows:
            if r["complete_coverage_ceiling"] >= target:
                if best is None or r["mean_pool_size"] < best["mean_pool_size"]:
                    best = r
        return None if best is None else {"breadth": best["breadth"],
                                          "ceiling": best["complete_coverage_ceiling"],
                                          "mean_pool_size": best["mean_pool_size"]}

    head2head = []
    base = arms[0.0]
    for t in (0.50, 0.60, 0.674, 0.70, 0.80, 0.90, 0.95):
        h = {"target_ceiling": t,
             "base_beta0": cheapest(base, t),
             "oracle": cheapest(oracle_rows, t)}
        for beta in BETA_SWEEP:
            if beta == 0.0:
                continue
            h[f"fused_beta{beta}"] = cheapest(arms[beta], t)
        bb = h["base_beta0"]
        # best fused pool at this target vs base
        fused_pools = [h[f"fused_beta{b}"]["mean_pool_size"]
                       for b in BETA_SWEEP if b != 0.0 and h.get(f"fused_beta{b}")]
        if bb and fused_pools:
            h["best_fused_pool_ratio_vs_base"] = round(min(fused_pools) / bb["mean_pool_size"], 3)
        head2head.append(h)

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free, upper bound above any intra-cluster ranker)",
        "fusion": "score_fused(d) = name_evidence(d) * (W[home][d] + s)^beta ; beta=0 reproduces e-0030 pure name predictor; W[home][.] is the TRAIN home-specific bridge prior",
        "smoothing_s": SMOOTH,
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "total_clustered_decls": sum(cluster_size.values()),
        "leakage_control": "W (incl. the home-specific prior W[home][.]), IDF, token x foreign-cluster co-occurrence, and global demand ALL estimated on TRAIN only; coverage on held-out TEST; only query signal is the test theorem's NAME.",
        "predictor_accuracy_by_beta": {str(b): pred_acc[b] for b in BETA_SWEEP},
        "fused_sweep_by_beta": {str(b): arms[b] for b in BETA_SWEEP},
        "oracle_sweep": oracle_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"train_cross = {res['n_train_cross_domain']}")
    print("\npredictor top-k hits a foreign cluster, by beta:")
    for b in BETA_SWEEP:
        acc = res["predictor_accuracy_by_beta"][str(b)]["top_k_hits_a_foreign_cluster"]
        full = res["predictor_accuracy_by_beta"][str(b)]["top_k_covers_ALL_foreign_clusters"]
        print(f"  beta={b}:  hits {acc}   coversALL {full}")
    for b in BETA_SWEEP:
        print(f"\nfused_sweep beta={b}:  {'m':>3} {'ceiling':>9} {'mean_pool':>11}")
        for r in res["fused_sweep_by_beta"][str(b)]:
            print(f"   {r['breadth']:>3} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>11}")
    print(f"\noracle:  {'m':>3} {'ceiling':>9} {'mean_pool':>11}")
    for r in res["oracle_sweep"]:
        print(f"   {r['breadth']:>3} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>11}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"pool={x['mean_pool_size']}@b{x['breadth']}" if x else "unreach"
        parts = [f"base[{fmt(h['base_beta0'])}]"]
        for b in BETA_SWEEP:
            if b == 0.0:
                continue
            parts.append(f"f{b}[{fmt(h.get(f'fused_beta{b}'))}]")
        parts.append(f"oracle[{fmt(h['oracle'])}]")
        ratio = h.get("best_fused_pool_ratio_vs_base")
        rtxt = f"  bestfused/base={ratio}" if ratio is not None else ""
        print(f"   >={h['target_ceiling']}: " + "  ".join(parts) + rtxt)


if __name__ == "__main__":
    main()
