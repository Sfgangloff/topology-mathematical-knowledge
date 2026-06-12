"""
Goal-specific router vs the static top-K bridge router (q-0011, follow-on to e-0029/a-0026).

e-0029 (a-0026) isolated the structural cause of the low oracle-gate complete-coverage gain:
the STATIC top-K bridge router (home + the K globally-biggest bridge neighbours of the home
cluster, the SAME K clusters for every goal in a home cluster) cannot reach the foreign
cluster many cross-domain proofs need.  It also found that cross-domain premise need is
CONCENTRATED -- 59.5% of cross-domain goals need exactly ONE distinct foreign cluster,
median = 1 -- and named the lever explicitly:

    "...the right lever is a GOAL-SPECIFIC router (route the 1-2 clusters THIS goal needs)
     rather than a wider static one that dilutes the within-domain majority."

The escape-predictor chain (e-0023/e-0024) built a per-theorem name-token predictor but KEPT
the static top-5 router as a fixed BASE and only predicted the escapes BEYOND it, so it always
paid the static base's ~53.8k-decl pool.  This experiment tests the lever a-0026 actually
named: DROP the static base entirely and make the WHOLE router goal-specific -- route
home + the top-m foreign clusters PREDICTED from the goal's own name tokens -- then compare,
head-to-head AT MATCHED COMPLETE-COVERAGE, whether the goal-specific router reaches a given
ceiling at a SMALLER candidate pool than the static top-K router.

Metric (identical to e-0029): complete-coverage CEILING = fraction of cross-domain TEST
theorems whose ENTIRE clustered premise set lands inside the routed cluster pool.  It is
exact and ranking-free (depends only on whether every needed cluster is routed), so it is an
upper bound above any intra-cluster ranker.  Candidate cost = mean sum of routed cluster
sizes (the pool a retriever must rank).

Arms (only the routing policy changes; same 432 held-out cross-domain TEST theorems, same
train-only bridge weights W, same leakage control as the whole chain):
  * static(K)   : home + top-K bridge neighbours by train weight W[home][.]  (the e-0029 router).
  * goal(m)     : home + the top-m foreign clusters predicted from the goal's NAME tokens.
                  Predictor (TRAIN-only): from each train cross-domain theorem (home c,
                  foreign clusters F = prem_clusters - {c}) accumulate token t x foreign
                  cluster d co-occurrence n(t,d) and token freq n(t); at test score a foreign
                  cluster d by  sum_t IDF(t) * P(d|t),  P(d|t)=n(t,d)/n(t).
  * oracle(m)   : home + the goal's m most-demanded ACTUAL foreign clusters (leaky upper
                  bound = best any goal-specific router of breadth m could do).

Leakage control: W, IDF, the token x foreign-cluster co-occurrence counts and the global
foreign-cluster demand ranking are ALL estimated on TRAIN only; coverage is measured on
held-out TEST theorems; the only query signal is the test theorem's own NAME (always
available to a prover before it proves).

Output: results/bridge_retrieval_goalrouter.json
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
OUT = RESULTS / "bridge_retrieval_goalrouter.json"

K_SWEEP = [0, 1, 2, 3, 5, 10, 20]      # static top-K bridge-neighbour breadths
M_SWEEP = [1, 2, 3, 5, 10, 20]         # goal-specific / oracle predicted-cluster breadths
DF_STOP_FRAC = 0.20                     # drop name tokens in >20% of decls as stop-tokens

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    """full_name -> set of lowercased subtokens (split on '.', '_', camelCase, digits)."""
    toks = set()
    for part in name.replace("_", ".").split("."):
        for m in _CAMEL.findall(part):
            if len(m) >= 2:
                toks.add(m.lower())
    return toks


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
    """Return list of (full_name, home, prem_clusters) for cross-domain theorems in split."""
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
    """IDF over all clustered declaration names; drop stop-tokens (>DF_STOP_FRAC of decls)."""
    df = collections.Counter()
    for name in cluster:
        for tok in tokenize(name):
            df[tok] += 1
    N = len(cluster)
    stop = {tok for tok, c in df.items() if c > DF_STOP_FRAC * N}
    idf = {tok: math.log(N / c) for tok, c in df.items() if tok not in stop}
    return idf, stop


def train_foreign_predictor(train, idf, stop):
    """token t x FOREIGN cluster d -> co-occurrence counts, from ALL train cross-domain
    theorems (foreign = any cluster in prem_clusters other than the home cluster)."""
    tok_cluster = collections.defaultdict(collections.Counter)   # n(t,d)
    tok_total = collections.Counter()                            # n(t)
    global_demand = collections.Counter()                        # global foreign-cluster demand
    for fn, c, prem_clusters in train:
        foreign = prem_clusters - {c}
        if not foreign:
            continue
        toks = (tokenize(fn) - stop) & idf.keys()
        for t in toks:
            tok_total[t] += 1
            for d in foreign:
                tok_cluster[t][d] += 1
        for d in foreign:
            global_demand[d] += 1
    return tok_cluster, tok_total, global_demand


def predict_foreign(fn, tok_cluster, tok_total, idf, stop, exclude):
    """Rank candidate foreign clusters for a query theorem by sum_t IDF(t) * P(d|t)."""
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
    return [d for d, _ in score.most_common()]


def sweep(test, cluster_size, route_fn, breadths):
    """For each breadth b, route per route_fn; report ceiling, mean & median pool size."""
    n = len(test)
    rows = []
    for b in breadths:
        covered = 0
        pool_sizes = []
        for fn, c, prem_clusters in test:
            routed = route_fn(fn, c, prem_clusters, b)
            pool_sizes.append(sum(cluster_size[cid] for cid in routed))
            if prem_clusters <= routed:
                covered += 1
        pool_sizes.sort()
        rows.append({
            "breadth": b,
            "complete_coverage_ceiling": round(covered / n, 4),
            "mean_pool_size": round(sum(pool_sizes) / n, 1),
            "median_pool_size": pool_sizes[len(pool_sizes) // 2],
        })
    return rows


def evaluate(cluster, premises, split):
    W = build_bridge_weights(cluster, premises, split)
    cluster_size = collections.Counter(cluster.values())
    top_static = {c: [d for d, _ in W[c].most_common(max(K_SWEEP))] for c in W}

    idf, stop = build_idf(cluster)
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    tok_cluster, tok_total, global_demand = train_foreign_predictor(train, idf, stop)
    demand_rank = [d for d, _ in global_demand.most_common()]

    # ---- arm: STATIC top-K bridge router (e-0029) ----
    def route_static(fn, c, prem_clusters, K):
        return {c} | set(top_static.get(c, [])[:K])
    static_rows = sweep(test, cluster_size, route_static, K_SWEEP)

    # ---- arm: GOAL-SPECIFIC router (home + top-m predicted foreign clusters) ----
    # cache per-theorem prediction (exclude only the home cluster).
    pred_cache = {fn: predict_foreign(fn, tok_cluster, tok_total, idf, stop, {c})
                  for fn, c, _ in test}

    def route_goal(fn, c, prem_clusters, m):
        return {c} | set(pred_cache[fn][:m])
    goal_rows = sweep(test, cluster_size, route_goal, M_SWEEP)

    # ---- arm: ORACLE (home + the goal's m most-demanded ACTUAL foreign clusters) ----
    def route_oracle(fn, c, prem_clusters, m):
        foreign = sorted(prem_clusters - {c}, key=lambda d: -global_demand[d])
        return {c} | set(foreign[:m])
    oracle_rows = sweep(test, cluster_size, route_oracle, M_SWEEP)

    # ---- predictor accuracy: of cross-domain theorems, how often is a top-k predicted
    #      foreign cluster among the goal's ACTUAL foreign clusters? ----
    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    full = {1: 0, 2: 0, 3: 0, 5: 0}   # does top-k contain ALL actual foreign clusters?
    for fn, c, prem_clusters in test:
        foreign = prem_clusters - {c}
        pred = pred_cache[fn]
        for k in hits:
            if set(pred[:k]) & foreign:
                hits[k] += 1
            if foreign <= set(pred[:k]):
                full[k] += 1
    pred_acc = {f"top{k}_hits_a_foreign_cluster": round(v / n, 4) for k, v in hits.items()}
    pred_full = {f"top{k}_covers_ALL_foreign_clusters": round(v / n, 4) for k, v in full.items()}

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
    for t in (0.50, 0.60, 0.674, 0.70, 0.80, 0.90, 0.95):
        h = {"target_ceiling": t,
             "static": cheapest(static_rows, t),
             "goal": cheapest(goal_rows, t),
             "oracle": cheapest(oracle_rows, t)}
        sg = h["static"]
        gg = h["goal"]
        if sg and gg and sg["mean_pool_size"] > 0:
            h["goal_pool_ratio_vs_static"] = round(gg["mean_pool_size"] / sg["mean_pool_size"], 3)
        head2head.append(h)

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free, upper bound above any intra-cluster ranker)",
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "total_clustered_decls": sum(cluster_size.values()),
        "leakage_control": "bridge weights W, IDF, token x foreign-cluster co-occurrence counts, and global foreign-cluster demand ALL estimated on TRAIN only; coverage measured on held-out TEST theorems; the only query signal is the test theorem's own NAME (always available to a prover).",
        "goal_predictor_accuracy": pred_acc,
        "goal_predictor_full_coverage": pred_full,
        "static_sweep": static_rows,
        "goal_sweep": goal_rows,
        "oracle_sweep": oracle_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"train_cross = {res['n_train_cross_domain']}")
    print("\ngoal predictor: top-k hits a foreign cluster:", res["goal_predictor_accuracy"])
    print("goal predictor: top-k covers ALL foreign clusters:", res["goal_predictor_full_coverage"])
    for arm in ("static_sweep", "goal_sweep", "oracle_sweep"):
        print(f"\n{arm}:  {'b':>3} {'ceiling':>9} {'mean_pool':>11} {'med_pool':>9}")
        for r in res[arm]:
            print(f"   {r['breadth']:>3} {r['complete_coverage_ceiling']:>9} "
                  f"{r['mean_pool_size']:>11} {r['median_pool_size']:>9}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"b={x['breadth']} pool={x['mean_pool_size']}" if x else "unreachable"
        ratio = h.get("goal_pool_ratio_vs_static")
        rtxt = f"  goal/static={ratio}" if ratio is not None else ""
        print(f"   ceiling>={h['target_ceiling']}: static[{fmt(h['static'])}]  "
              f"goal[{fmt(h['goal'])}]  oracle[{fmt(h['oracle'])}]{rtxt}")


if __name__ == "__main__":
    main()
