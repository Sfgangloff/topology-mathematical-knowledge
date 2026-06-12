"""
Does the q-0011 retrieval chain leak through the HOME CLUSTER? (q-0011, robustness control
for the whole a-0013..a-0022 chain).

Every experiment in the q-0011 chain (e-0012..e-0024) routes a held-out TEST theorem to
its home cluster + top-K bridge neighbours, and treats the home cluster `cluster[fn]` as a
known input.  But that home cluster comes from `data/clusters.json` = the Leiden clustering
of the FULL dependency graph (e-0002), which INCLUDES the test theorem's own outgoing premise
edges.  So the home-cluster label is itself derived from the very premises the retriever is
trying to recover -- a leakage a real prover does not enjoy: meeting a fresh goal, it has NO
premises yet (those are what it seeks), so it cannot read the home cluster off the graph.

This script asks the honest question: if the prover must PREDICT the home cluster from a
leakage-free signal it actually has (the goal's NAME tokens, same content signal as e-0016/
e-0023), how well can it, and does the bridge-routing complete-coverage of a-0017/a-0018
survive?

Setup is identical to e-0018/e-0023 (432 held-out cross-domain TEST theorems defined by the
TRUE home cluster, train-only bridge weights W, exact ranking-free complete-coverage CEILING:
fraction of test theorems whose ENTIRE clustered premise set lands in the routed pool).  The
ONLY change is which home cluster the router is given:

  * oracle-home  : route TRUE_home + top-K bridge neighbours          (the a-0013..a-0022 assumption)
  * predicted-home: predict home from name tokens (TRAIN-only IDF*P(c|t)), route
                    PRED_home + its top-K bridge neighbours, scored against the theorem's
                    TRUE premise clusters (a wrong home prediction misses the in-home premises).

Predictor (leakage-safe, TRAIN-only): co-occurrence n(t,c) of name token t with home cluster
c over TRAIN theorems; score(c) = sum_t IDF(t) * P(c|t), P(c|t)=n(t,c)/n(t).  Same machinery
as the e-0023 escape predictor, retargeted from escape clusters to the home cluster.

If predicted-home complete-coverage tracks oracle-home, the q-0011 chain is robust to the
home-label leakage.  If it collapses, the retrieval proxy overstates the bridge-aware payoff.

Output: results/bridge_retrieval_home_leakage.json
"""

import json
import re
import math
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_home_leakage.json"

K_BASE = 5                     # fixed bridge router breadth (e-0018 baseline)
DF_STOP_FRAC = 0.20            # drop name tokens in >20% of decls as stop-tokens

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    toks = set()
    for part in name.replace("_", ".").split("."):
        for m in _CAMEL.findall(part):
            if len(m) >= 2:
                toks.add(m.lower())
    return toks


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
        out.append((fn, c, prem_clusters))
    return out


def build_idf(cluster):
    df = collections.Counter()
    for name in cluster:
        for tok in tokenize(name):
            df[tok] += 1
    N = len(cluster)
    stop = {tok for tok, c in df.items() if c > DF_STOP_FRAC * N}
    idf = {tok: math.log(N / c) for tok, c in df.items() if tok not in stop}
    return idf, stop


def train_home_predictor(cluster, split, idf, stop):
    """token t x home cluster c -> co-occurrence counts, over ALL TRAIN theorems (not just
    cross-domain), since at test time the prover does not yet know if a goal is cross-domain."""
    tok_cluster = collections.defaultdict(collections.Counter)   # n(t,c)
    tok_total = collections.Counter()                            # n(t)
    for fn, c in cluster.items():
        if split.get(fn) != "train":
            continue
        toks = (tokenize(fn) - stop) & idf.keys()
        for t in toks:
            tok_total[t] += 1
            tok_cluster[t][c] += 1
    return tok_cluster, tok_total


def predict_home(fn, tok_cluster, tok_total, idf, stop):
    score = collections.Counter()
    for t in (tokenize(fn) - stop) & idf.keys():
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for c, ntc in tok_cluster[t].items():
            score[c] += w * (ntc / nt)
    return [c for c, _ in score.most_common()]


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    top5 = {c: set(d for d, _ in W[c].most_common(K_BASE)) for c in W}

    def router(home):
        return {home} | top5.get(home, set())

    idf, stop = build_idf(cluster)
    tok_cluster, tok_total = train_home_predictor(cluster, split, idf, stop)
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    # ---- home-cluster prediction accuracy on the cross-domain test set ----
    hits = {1: 0, 3: 0, 5: 0}
    for fn, c, _ in test:
        pred = predict_home(fn, tok_cluster, tok_total, idf, stop)
        for k in hits:
            if c in pred[:k]:
                hits[k] += 1
    home_acc = {f"home_top{k}_acc": round(v / n, 4) for k, v in hits.items()}

    # ---- complete coverage under oracle vs predicted home (top-1) ----
    cov_oracle = pool_oracle = 0
    cov_pred = pool_pred = 0
    pred_eq_true = 0
    pools_o, pools_p = [], []
    for fn, c, prem_clusters in test:
        # oracle home
        routed_o = router(c)
        pools_o.append(sum(cluster_size[cid] for cid in routed_o))
        if prem_clusters <= routed_o:
            cov_oracle += 1
        # predicted home (top-1)
        ph = predict_home(fn, tok_cluster, tok_total, idf, stop)
        ph1 = ph[0] if ph else c
        if ph1 == c:
            pred_eq_true += 1
        routed_p = router(ph1)
        pools_p.append(sum(cluster_size[cid] for cid in routed_p))
        if prem_clusters <= routed_p:
            cov_pred += 1
    pools_o.sort(); pools_p.sort()

    # ---- predicted-home with a small fallback set: union of top-h predicted home clusters'
    #      routers (hedge the home uncertainty by routing several candidate homes) ----
    hedge_rows = []
    for h in (1, 2, 3, 5):
        cov = 0
        pools = []
        for fn, c, prem_clusters in test:
            ph = predict_home(fn, tok_cluster, tok_total, idf, stop)[:h]
            if not ph:
                ph = [c]
            routed = set()
            for home in ph:
                routed |= router(home)
            pools.append(sum(cluster_size[cid] for cid in routed))
            if prem_clusters <= routed:
                cov += 1
        pools.sort()
        hedge_rows.append({
            "home_candidates": h,
            "complete_coverage_ceiling": round(cov / n, 4),
            "mean_pool_size": round(sum(pools) / n, 1),
            "median_pool_size": pools[len(pools) // 2],
        })

    return {
        "n_cross_domain_test": n,
        "k_base": K_BASE,
        "home_prediction_accuracy": home_acc,
        "predicted_home_eq_true_home": round(pred_eq_true / n, 4),
        "oracle_home": {
            "complete_coverage_ceiling": round(cov_oracle / n, 4),
            "mean_pool_size": round(sum(pools_o) / n, 1),
            "median_pool_size": pools_o[len(pools_o) // 2],
        },
        "predicted_home_top1": {
            "complete_coverage_ceiling": round(cov_pred / n, 4),
            "mean_pool_size": round(sum(pools_p) / n, 1),
            "median_pool_size": pools_p[len(pools_p) // 2],
        },
        "predicted_home_hedged": hedge_rows,
    }


def main():
    cluster, premises, split = load()
    result = evaluate(cluster, premises, split)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
