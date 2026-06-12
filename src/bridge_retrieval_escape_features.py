"""
Richer escape-predictor features: the lever a-0021 named (q-0011, follow-on to e-0023/a-0021).

a-0021 (e-0023) showed a per-THEOREM escape predictor that scores candidate escape clusters
purely from the goal theorem's NAME tokens, score(d) = sum_t IDF(t) * P(d|t), beats the
a-0020 global whitelist -- but it captures only part of the oracle gain because the name
signal localizes escapes only MODERATELY (top-1 predicted cluster hits an actual escape
target 39.7%, top-3 66.0%, top-5 83.0%).  Its closing line named the remaining lever:

    "...the remaining lever is a STRONGER query->escape-cluster predictor (richer features
     than the goal's name tokens)."

The one strong leakage-free feature the name-only predictor THROWS AWAY is the theorem's
HOME CLUSTER.  a-0019 already established that cross-domain escape behaviour is conditioned
by the home cluster (per-cluster K=5 coverage train->test Pearson r=0.55, train coverage
0.57-0.80 across clusters) -- so WHERE a theorem lives strongly predicts WHERE it escapes,
independently of its name.  The name-only predictor never uses it: P(d|t) is pooled over all
homes.  This script adds the home cluster as a feature and asks whether it sharpens escape
localization and lowers the cost-to-ceiling.

Predictors compared (all leakage-safe, TRAIN-only counts; identical e-0018/e-0023 setup:
432 held-out cross-domain TEST theorems, home+top-5 bridge router base, exact ranking-free
complete-coverage CEILING metric; only the per-theorem escape RANKING changes):

  * name   : score(d) = sum_t IDF(t) * P(d|t)                       [a-0021 baseline]
  * home   : score(d) = P(d | home=c)  = n_c(d) / sum_d n_c(d)      [home prior alone]
             (per-home escape frequency from TRAIN escapes -- the feature 'name' ignores)
  * combo  : score(d) = P(d | home=c) * (1 + sum_t IDF(t) * P(d|t)) [home prior x name evidence]
             a naive-Bayes-flavoured product: the home prior restricts to clusters this home
             actually escapes to, the name evidence re-ranks among them.
  * oracle : the theorem's m most-demanded ACTUAL escape clusters (leaky upper bound; ref).

Question: does conditioning on the home cluster -- a feature the prover always knows -- raise
top-1 escape-localization accuracy above the name-only 39.7%, and reach each complete-coverage
ceiling at a smaller mean candidate pool?  If 'home'/'combo' beat 'name', richer features are
the lever a-0021 predicted; if they tie, the name signal already subsumes the home prior.

Output: results/bridge_retrieval_escape_features.json
"""

import json
import collections
from pathlib import Path

# Reuse the exact loaders, tokenizer, IDF, router and metric of the a-0021 experiment so the
# ONLY thing that changes between the two studies is the escape-ranking function.
from bridge_retrieval_escape_predictor import (
    load, build_bridge_weights, cross_domain_set, build_idf,
    train_escape_predictor, predict_escape, sweep, tokenize,
    K_BASE, M_SWEEP,
)

RESULTS = Path(__file__).parent.parent / "results"
OUT = RESULTS / "bridge_retrieval_escape_features.json"


def train_home_prior(train, router_base):
    """home cluster c x escape cluster d -> #train theorems in c that escape into d."""
    home_esc = collections.defaultdict(collections.Counter)   # n_c(d)
    for fn, c, prem_clusters in train:
        for d in prem_clusters - router_base(c):
            home_esc[c][d] += 1
    return home_esc


def rank_home(c, home_esc, exclude):
    """Candidate escape clusters ranked by P(d | home=c), descending."""
    return [d for d, _ in home_esc.get(c, collections.Counter()).most_common()
            if d not in exclude]


def rank_combo(fn, c, tok_cluster, tok_total, idf, stop, home_esc, exclude):
    """score(d) = P(d|home=c) * (1 + name_evidence(d)); ranks home's escape targets by name."""
    cc = home_esc.get(c)
    if not cc:
        return []
    tot = sum(cc.values())
    # name evidence per candidate cluster (only over clusters this home ever escapes to)
    name_ev = collections.Counter()
    for t in (tokenize(fn) - stop) & idf.keys():
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for d, ntd in tok_cluster[t].items():
            if d in cc and d not in exclude:
                name_ev[d] += w * (ntd / nt)
    scored = []
    for d, n in cc.items():
        if d in exclude:
            continue
        scored.append((d, (n / tot) * (1.0 + name_ev.get(d, 0.0))))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored]


def accuracy(test, router_base, rank_fn):
    """Of theorems that escape the base router, how often is a top-k pick an actual escape?"""
    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    n_hard = 0
    for fn, c, prem_clusters in test:
        esc = prem_clusters - router_base(c)
        if not esc:
            continue
        n_hard += 1
        pred = rank_fn(fn, c, router_base(c))
        for k in hits:
            if set(pred[:k]) & esc:
                hits[k] += 1
    return n_hard, {f"top{k}": round(v / max(1, n_hard), 4) for k, v in hits.items()}


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    top5 = {c: set(d for d, _ in W[c].most_common(K_BASE)) for c in W}

    def router_base(c):
        return {c} | top5.get(c, set())

    idf, stop = build_idf(cluster)
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    tok_cluster, tok_total, _ = train_escape_predictor(train, router_base, idf, stop)
    home_esc = train_home_prior(train, router_base)
    esc_demand = collections.Counter()
    for fn, c, prem_clusters in train:
        for d in prem_clusters - router_base(c):
            esc_demand[d] += 1

    # --- ranking functions (signature: fn, c, exclude -> ranked list of escape clusters) ---
    def r_name(fn, c, exclude):
        return predict_escape(fn, tok_cluster, tok_total, idf, stop, exclude)

    def r_home(fn, c, exclude):
        return rank_home(c, home_esc, exclude)

    def r_combo(fn, c, exclude):
        return rank_combo(fn, c, tok_cluster, tok_total, idf, stop, home_esc, exclude)

    rankers = {"name": r_name, "home": r_home, "combo": r_combo}

    # --- localization accuracy per ranker ---
    acc = {}
    for key, rf in rankers.items():
        n_hard, a = accuracy(test, router_base, rf)
        acc[key] = a
    acc["_n_hard_third"] = n_hard

    # --- complete-coverage ceiling vs mean pool sweep per ranker ---
    sweeps = {}
    for key, rf in rankers.items():
        def pick(fn, c, prem_clusters, base, m, _rf=rf):
            return [] if m == 0 else _rf(fn, c, base)[:m]
        sweeps[key] = sweep(test, cluster_size, router_base, pick, M_SWEEP)

    def pick_oracle(fn, c, prem_clusters, base, m):
        esc = [d for d in prem_clusters - base]
        esc.sort(key=lambda d: -esc_demand[d])
        return esc[:m]
    sweeps["oracle"] = sweep(test, cluster_size, router_base, pick_oracle, M_SWEEP)

    # --- head-to-head: cheapest mean pool to reach each target ceiling ---
    def cheapest(rows, target):
        best = None
        for r in rows:
            if r["complete_coverage_ceiling"] >= target:
                if best is None or r["mean_pool_size"] < best["mean_pool_size"]:
                    best = r
        return None if best is None else {"m": best["m"], "ceiling": best["complete_coverage_ceiling"],
                                          "mean_pool_size": best["mean_pool_size"]}
    head2head = []
    for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        row = {"target_ceiling": t}
        for key in ("name", "home", "combo", "oracle"):
            row[key] = cheapest(sweeps[key], t)
        head2head.append(row)

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed pool",
        "n_cross_domain_test_theorems": n,
        "n_hard_third_test_theorems": acc["_n_hard_third"],
        "total_clustered_decls": sum(cluster_size.values()),
        "k_base": K_BASE,
        "leakage_control": "bridge weights W, IDF, name-token co-occurrence n(t,d) and home-prior n_c(d) ALL estimated on TRAIN only; coverage on held-out TEST; query signals = test theorem's NAME tokens + its HOME cluster id (both always available to a prover).",
        "escape_localization_accuracy": {k: v for k, v in acc.items() if k != "_n_hard_third"},
        "sweeps": sweeps,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"hard third = {res['n_hard_third_test_theorems']}")
    print("\nescape-localization accuracy (top-k pick hits an actual escape target):")
    print(f"   {'ranker':>6} {'top1':>7} {'top2':>7} {'top3':>7} {'top5':>7}")
    for key in ("name", "home", "combo"):
        a = res["escape_localization_accuracy"][key]
        print(f"   {key:>6} {a['top1']:>7} {a['top2']:>7} {a['top3']:>7} {a['top5']:>7}")
    for key in ("name", "home", "combo", "oracle"):
        print(f"\n{key} sweep:  {'m':>3} {'ceiling':>9} {'mean_pool':>10}")
        for r in res["sweeps"][key]:
            print(f"   {r['m']:>3} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>10}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"m={x['m']} pool={int(x['mean_pool_size'])}" if x else "unreachable"
        print(f"   >={h['target_ceiling']}: name[{fmt(h['name'])}]  home[{fmt(h['home'])}]  "
              f"combo[{fmt(h['combo'])}]  oracle[{fmt(h['oracle'])}]")


if __name__ == "__main__":
    main()
