"""
Per-THEOREM escape prediction: the lever a-0020 named (q-0011, follow-on to e-0022/a-0020).

a-0020 (e-0022) established that the 'hard third' a-0018 left unreachable by the home+top-5
bridge router is CONCENTRATED, not diffuse: when a cross-domain theorem escapes its top-5
router it escapes into a MEDIAN of 1 cluster, and the whole hard third's escapes span just
~25 clusters.  It exploited this with a fixed GLOBAL escape-target whitelist (the same M
clusters added to EVERY theorem), but flagged the real lever in its closing line:

    "...the right lever is per-THEOREM (not just per-home-cluster, a-0019) escape
     prediction, since the median escaped demand is exactly one cluster."

A global whitelist pays for its M clusters on all 432 theorems even though each theorem
typically needs just ONE of them.  A per-theorem predictor that routes each theorem only to
ITS likely escape cluster(s) should reach the same complete-coverage ceiling at a far smaller
mean candidate pool.  This script builds and tests exactly that predictor.

Predictor (leakage-safe, TRAIN-only, content signal a prover always has = the goal's name):
  For each TRAIN cross-domain theorem that escapes its own home+top-5 router, take its
  escaped target clusters E = prem_clusters - (home+top5) and its NAME tokens (same
  tokenizer as e-0016).  Accumulate, per name token t and escape cluster d, a co-occurrence
  count n(t,d) and a token frequency n(t).  At test time, score a candidate escape cluster d
  for a query theorem by   score(d) = sum over the query's tokens t of  IDF(t) * P(d | t),
  P(d|t)=n(t,d)/n(t), and route to home + top5 + the top-m highest-scoring NEW clusters.
  (Tokens, IDF, co-occurrence counts all come from TRAIN only.)

Arms (all under the identical e-0018/e-0022 setup: same 432 held-out cross-domain TEST
theorems, same train-only bridge weights W, same exact ranking-free complete-coverage
CEILING metric -- only the routing policy changes):
  * predictor(m)  : home + top5 + per-theorem top-m predicted escape clusters.
  * whitelist(M)  : home + top5 + global top-M escape whitelist (the a-0020 policy; reference).
  * oracle(m)     : home + top5 + the theorem's m most-demanded ACTUAL escape clusters
                    (leaky upper bound = best any per-theorem router could do at breadth m).

Head-to-head: at matched complete-coverage ceiling, which policy reaches it with the
smaller mean candidate pool?  If predictor beats whitelist, per-theorem escape prediction
is the cheaper lever a-0020 predicted; if it ties/loses, the name signal is too weak to
localize escapes and the global whitelist is as good as it gets (an equally honest null).

Output: results/bridge_retrieval_escape_predictor.json
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
OUT = RESULTS / "bridge_retrieval_escape_predictor.json"

K_BASE = 5                                  # fixed bridge router breadth (e-0018 baseline)
M_SWEEP = [0, 1, 2, 3, 5, 8, 12, 20]        # extra escape clusters added (predictor / oracle)
M_WHITELIST = [0, 1, 2, 3, 5, 8, 12, 20]    # global whitelist breadths (a-0020 reference)
DF_STOP_FRAC = 0.20                          # drop name tokens in >20% of decls as stop-tokens

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


def train_escape_predictor(train, router_base, idf, stop):
    """token t x escape cluster d -> co-occurrence counts, from TRAIN escapes only."""
    tok_cluster = collections.defaultdict(collections.Counter)   # n(t,d)
    tok_total = collections.Counter()                            # n(t)
    global_freq = collections.Counter()                          # global whitelist ranking
    for fn, c, prem_clusters in train:
        esc = prem_clusters - router_base(c)
        if not esc:
            continue
        toks = (tokenize(fn) - stop) & idf.keys()
        for t in toks:
            tok_total[t] += 1
            for d in esc:
                tok_cluster[t][d] += 1
        for d in esc:
            global_freq[d] += 1
    return tok_cluster, tok_total, global_freq


def predict_escape(fn, tok_cluster, tok_total, idf, stop, exclude):
    """Rank candidate escape clusters for a query theorem by sum_t IDF(t) * P(d|t)."""
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


def sweep(test, cluster_size, router_base, pick_fn, breadths):
    """For each breadth m, route home+top5 + m picked clusters; report ceiling & pool."""
    n = len(test)
    rows = []
    for m in breadths:
        covered = 0
        pool_sizes = []
        for fn, c, prem_clusters in test:
            base = router_base(c)
            extra = pick_fn(fn, c, prem_clusters, base, m)
            routed = base | set(extra)
            pool_sizes.append(sum(cluster_size[cid] for cid in routed))
            if prem_clusters <= routed:
                covered += 1
        pool_sizes.sort()
        rows.append({
            "m": m,
            "complete_coverage_ceiling": round(covered / n, 4),
            "mean_pool_size": round(sum(pool_sizes) / n, 1),
            "median_pool_size": pool_sizes[len(pool_sizes) // 2],
        })
    return rows


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

    tok_cluster, tok_total, global_freq = train_escape_predictor(train, router_base, idf, stop)
    whitelist = [d for d, _ in global_freq.most_common()]      # fixed global ranking (a-0020)

    # ---- arm: per-theorem PREDICTOR (top-m predicted escape clusters) ----
    def pick_predictor(fn, c, prem_clusters, base, m):
        if m == 0:
            return []
        return predict_escape(fn, tok_cluster, tok_total, idf, stop, base)[:m]
    predictor_rows = sweep(test, cluster_size, router_base, pick_predictor, M_SWEEP)

    # ---- arm: global WHITELIST (a-0020 reference) ----
    def pick_whitelist(fn, c, prem_clusters, base, m):
        return [d for d in whitelist if d not in base][:m]
    whitelist_rows = sweep(test, cluster_size, router_base, pick_whitelist, M_WHITELIST)

    # ---- arm: ORACLE (theorem's m most-demanded ACTUAL escape clusters; leaky upper bound) ----
    esc_demand = collections.Counter()      # global demand, only to order each theorem's escapes
    for fn, c, prem_clusters in train:
        for d in prem_clusters - router_base(c):
            esc_demand[d] += 1

    def pick_oracle(fn, c, prem_clusters, base, m):
        esc = [d for d in prem_clusters - base]
        esc.sort(key=lambda d: -esc_demand[d])
        return esc[:m]
    oracle_rows = sweep(test, cluster_size, router_base, pick_oracle, M_SWEEP)

    # ---- predictor accuracy: of the theorems that escape, how often is the top-1 / top-m
    #      predicted cluster among the theorem's actual escape targets? ----
    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    n_hard = 0
    for fn, c, prem_clusters in test:
        esc = prem_clusters - router_base(c)
        if not esc:
            continue
        n_hard += 1
        pred = predict_escape(fn, tok_cluster, tok_total, idf, stop, router_base(c))
        for k in hits:
            if set(pred[:k]) & esc:
                hits[k] += 1
    pred_acc = {f"top{k}_hits_an_escape_target": round(v / max(1, n_hard), 4)
                for k, v in hits.items()}

    # ---- head-to-head: cheapest mean pool to reach each target ceiling ----
    def cheapest(rows, target):
        best = None
        for r in rows:
            if r["complete_coverage_ceiling"] >= target:
                if best is None or r["mean_pool_size"] < best["mean_pool_size"]:
                    best = r
        return None if best is None else {"m": best["m"],
                                          "ceiling": best["complete_coverage_ceiling"],
                                          "mean_pool_size": best["mean_pool_size"]}
    head2head = []
    for t in (0.70, 0.72, 0.75, 0.80, 0.85, 0.90):
        head2head.append({
            "target_ceiling": t,
            "predictor": cheapest(predictor_rows, t),
            "whitelist": cheapest(whitelist_rows, t),
            "oracle": cheapest(oracle_rows, t),
        })

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed pool",
        "n_cross_domain_test_theorems": n,
        "n_hard_third_test_theorems": n_hard,
        "total_clustered_decls": sum(cluster_size.values()),
        "k_base": K_BASE,
        "leakage_control": "bridge weights W, IDF, escape-token co-occurrence counts and the global whitelist ALL estimated on TRAIN only; coverage measured on held-out TEST theorems; query signal is the test theorem's own NAME (always available to a prover).",
        "escape_predictor_accuracy": pred_acc,
        "predictor_sweep": predictor_rows,
        "whitelist_sweep": whitelist_rows,
        "oracle_sweep": oracle_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"hard third = {res['n_hard_third_test_theorems']}")
    print("escape-predictor accuracy (top-k predicted cluster hits an actual escape target):")
    for k, v in res["escape_predictor_accuracy"].items():
        print(f"   {k}: {v}")
    for arm in ("predictor_sweep", "whitelist_sweep", "oracle_sweep"):
        print(f"\n{arm}:  {'m':>3} {'ceiling':>9} {'mean_pool':>10}")
        for r in res[arm]:
            print(f"   {r['m']:>3} {r['complete_coverage_ceiling']:>9} {r['mean_pool_size']:>10}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"m={x['m']} pool={x['mean_pool_size']}" if x else "unreachable"
        print(f"   ceiling>={h['target_ceiling']}: "
              f"predictor[{fmt(h['predictor'])}]  whitelist[{fmt(h['whitelist'])}]  "
              f"oracle[{fmt(h['oracle'])}]")


if __name__ == "__main__":
    main()
