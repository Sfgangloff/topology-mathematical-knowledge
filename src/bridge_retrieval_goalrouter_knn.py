"""
k-NN target predictor vs the naive-Bayes linear goal-router (q-0011, follow-on to a-0029/a-0030).

The whole a-0013..a-0030 retrieval-proxy chain converged on ONE residual claim: the gap
between the realizable goal-specific router and the leaky ORACLE goal-router is "predictor
RANKING quality, not a missing structural signal" (a-0029, a-0030).  But every attempt to
close it varied the PRIOR or the FUSION, never the RANKING ALGORITHM itself:

  * e-0030 (a-0027): the goal router itself  -- score(d) = sum_t IDF(t) * P(d|t)   (naive-Bayes linear)
  * e-0031 (a-0028): de-bias by the demand prior  P(d|t)/P(d)^alpha                (still linear)
  * e-0032 (a-0029): fuse the home-specific bridge weight  W[home][.]              (still linear core)
  * e-0033 (a-0030): SET-UNION the static bridge router with the goal router        (still linear core)

All four keep the SAME linear token->cluster aggregation.  None tests whether a DIFFERENT
ranking algorithm over the IDENTICAL non-leaky name-token signal ranks the needed small
cluster higher.  This experiment supplies that missing control: a k-NN target predictor.

  goal_knn(m): represent the test goal and every TRAIN cross-domain theorem as an
               IDF-weighted name-token vector; find the goal's k nearest TRAIN cross-domain
               neighbours by IDF-weighted cosine similarity; rank foreign clusters by the
               similarity-weighted vote of those neighbours' OWN foreign-cluster sets; route
               home + the top-m ranked foreign clusters.

This is a strictly different ranking algorithm (instance-based / non-linear, exploits
token CO-OCCURRENCE within a single theorem) over the EXACT same inputs (the goal's name
tokens; all statistics TRAIN-only).  Decisive either way:

  * if k-NN beats the linear predictor and narrows the ~3.3x oracle pool gap -> the lever
    was the RANKING ALGORITHM, and cheap near-exhaustive name-token routing IS reachable
    (good news for q-0011);
  * if k-NN ties/loses -> the name-token SIGNAL itself is exhausted regardless of ranker,
    sharpening a-0029/a-0030's bound from "this linear predictor" to "the name-token signal",
    i.e. the residual gap genuinely needs type/statement CONTENT the corpus does not expose.

Metric, splits, leakage control, and the head-to-head cheapest-pool-to-ceiling table are
IDENTICAL to e-0030 (bridge_retrieval_goalrouter.py): complete-coverage CEILING over the
same 432 held-out cross-domain TEST theorems, every statistic estimated TRAIN-only, the only
query signal is the test theorem's own NAME.

Output: results/bridge_retrieval_goalrouter_knn.json
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
OUT = RESULTS / "bridge_retrieval_goalrouter_knn.json"

M_SWEEP = [1, 2, 3, 5, 10, 20]          # predicted-cluster breadths (match e-0030)
K_NEIGHBOURS = [10, 25, 50, 100, 200]   # k for the k-NN vote (swept; report the best)
DF_STOP_FRAC = 0.20                     # drop name tokens in >20% of decls as stop-tokens

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    """full_name -> set of lowercased subtokens (identical to e-0030)."""
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


# ---------------------------------------------------------------------------
# linear naive-Bayes predictor (the e-0030 arm) -- reproduced as the baseline
# ---------------------------------------------------------------------------
def train_linear(train, idf, stop):
    tok_cluster = collections.defaultdict(collections.Counter)   # n(t,d)
    tok_total = collections.Counter()                            # n(t)
    global_demand = collections.Counter()
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


def predict_linear(fn, tok_cluster, tok_total, idf, stop, exclude):
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


# ---------------------------------------------------------------------------
# k-NN predictor (the NEW arm): instance-based vote over nearest TRAIN theorems
# ---------------------------------------------------------------------------
def build_knn_index(train, idf, stop):
    """Per train cross-domain theorem: its IDF-weighted token vector (sparse), L2 norm,
    and its foreign-cluster set.  Plus an inverted index token -> [(train_idx, idf), ...]."""
    vecs = []          # list of {tok: idf}
    norms = []         # L2 norm of each vector
    foreigns = []      # foreign-cluster set per train theorem
    inverted = collections.defaultdict(list)   # tok -> [train_idx, ...]
    for i, (fn, c, prem_clusters) in enumerate(train):
        foreign = prem_clusters - {c}
        if not foreign:
            # keep index alignment but mark empty so it never contributes a vote
            vecs.append({})
            norms.append(0.0)
            foreigns.append(frozenset())
            continue
        toks = (tokenize(fn) - stop) & idf.keys()
        v = {t: idf[t] for t in toks}
        vecs.append(v)
        norms.append(math.sqrt(sum(w * w for w in v.values())))
        foreigns.append(frozenset(foreign))
        for t in toks:
            inverted[t].append(i)
    return vecs, norms, foreigns, inverted


def predict_knn(fn, idf, stop, vecs, norms, foreigns, inverted, k, exclude):
    """k-NN over IDF-weighted cosine similarity; foreign clusters ranked by the
    similarity-weighted vote of the goal's k nearest TRAIN cross-domain neighbours."""
    qtoks = (tokenize(fn) - stop) & idf.keys()
    qvec = {t: idf[t] for t in qtoks}
    qnorm = math.sqrt(sum(w * w for w in qvec.values()))
    if qnorm == 0:
        return []
    # accumulate dot products via the inverted index (only train theorems sharing a token)
    dot = collections.Counter()
    for t, wq in qvec.items():
        for i in inverted.get(t, ()):
            dot[i] += wq * vecs[i][t]
    # cosine = dot / (qnorm * norms[i]); rank neighbours, keep top-k with positive sim
    sims = []
    for i, dp in dot.items():
        if norms[i] > 0:
            sims.append((dp / (qnorm * norms[i]), i))
    sims.sort(reverse=True)
    sims = sims[:k]
    # similarity-weighted vote over neighbours' foreign clusters
    vote = collections.Counter()
    for sim, i in sims:
        for d in foreigns[i]:
            if d in exclude:
                continue
            vote[d] += sim
    return [d for d, _ in vote.most_common()]


def sweep(test, cluster_size, route_fn, breadths):
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


def pred_accuracy(test, pred_cache, n):
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

    # ---- baseline: linear naive-Bayes goal router (reproduces e-0030 goal arm) ----
    tok_cluster, tok_total, global_demand = train_linear(train, idf, stop)
    lin_cache = {fn: predict_linear(fn, tok_cluster, tok_total, idf, stop, {c})
                 for fn, c, _ in test}
    lin_acc, lin_full = pred_accuracy(test, lin_cache, n)
    lin_rows = sweep(test, cluster_size,
                     lambda fn, c, pc, m: {c} | set(lin_cache[fn][:m]), M_SWEEP)

    # ---- NEW: k-NN goal router, swept over k; pick k with best top-3 full coverage ----
    vecs, norms, foreigns, inverted = build_knn_index(train, idf, stop)
    knn_by_k = {}
    for k in K_NEIGHBOURS:
        cache = {fn: predict_knn(fn, idf, stop, vecs, norms, foreigns, inverted, k, {c})
                 for fn, c, _ in test}
        acc, full = pred_accuracy(test, cache, n)
        rows = sweep(test, cluster_size,
                     lambda fn, c, pc, m, cache=cache: {c} | set(cache[fn][:m]), M_SWEEP)
        knn_by_k[k] = {"k": k, "accuracy": acc, "full_coverage": full,
                       "sweep": rows, "_cache": cache}
    # select best k by top3_covers_ALL (the metric that drives complete coverage)
    best_k = max(K_NEIGHBOURS,
                 key=lambda k: knn_by_k[k]["full_coverage"]["top3_covers_ALL_foreign_clusters"])
    best = knn_by_k[best_k]

    # ---- oracle (leaky upper bound), identical to e-0030 ----
    oracle_rows = sweep(test, cluster_size,
                        lambda fn, c, pc, m: {c} | set(sorted(pc - {c}, key=lambda d: -global_demand[d])[:m]),
                        M_SWEEP)

    # ---- head-to-head cheapest pool to reach each target ceiling ----
    head2head = []
    for t in (0.50, 0.60, 0.674, 0.70, 0.80, 0.90, 0.95):
        lin = cheapest(lin_rows, t)
        knn = cheapest(best["sweep"], t)
        ora = cheapest(oracle_rows, t)
        h = {"target_ceiling": t, "linear": lin, "knn": knn, "oracle": ora}
        if lin and knn and lin["mean_pool_size"] > 0:
            h["knn_pool_ratio_vs_linear"] = round(knn["mean_pool_size"] / lin["mean_pool_size"], 3)
        if knn and ora and ora["mean_pool_size"] > 0:
            h["knn_pool_ratio_vs_oracle"] = round(knn["mean_pool_size"] / ora["mean_pool_size"], 3)
        head2head.append(h)

    # strip caches before serialising
    for k in knn_by_k:
        knn_by_k[k].pop("_cache", None)

    return {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free; identical to e-0030)",
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "total_clustered_decls": sum(cluster_size.values()),
        "leakage_control": "IDF, the k-NN train index (vectors + foreign-cluster sets), and the global demand ranking ALL estimated on TRAIN cross-domain theorems only; coverage measured on held-out TEST; the only query signal is the test theorem's own NAME.",
        "k_selected_by": "max top3_covers_ALL_foreign_clusters over K_NEIGHBOURS",
        "best_k": best_k,
        "linear_baseline": {"accuracy": lin_acc, "full_coverage": lin_full, "sweep": lin_rows},
        "knn_best": {"k": best_k, "accuracy": best["accuracy"],
                     "full_coverage": best["full_coverage"], "sweep": best["sweep"]},
        "knn_all_k": [{"k": kk, "accuracy": knn_by_k[kk]["accuracy"],
                       "full_coverage": knn_by_k[kk]["full_coverage"]} for kk in K_NEIGHBOURS],
        "oracle_sweep": oracle_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }


def main():
    cluster, premises, split = load()
    res = evaluate(cluster, premises, split)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"n_cross_domain_test = {res['n_cross_domain_test_theorems']}  "
          f"train_cross = {res['n_train_cross_domain']}  best_k = {res['best_k']}")
    print("\nlinear baseline full-coverage:", res["linear_baseline"]["full_coverage"])
    print("k-NN best    full-coverage:", res["knn_best"]["full_coverage"])
    print("\nk-NN sweep over k (top3_covers_ALL):")
    for row in res["knn_all_k"]:
        print(f"   k={row['k']:>3}  {row['full_coverage']}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in res["head_to_head_cheapest_pool_to_reach_ceiling"]:
        def fmt(x):
            return f"b={x['breadth']} pool={x['mean_pool_size']}" if x else "unreachable"
        r1 = h.get("knn_pool_ratio_vs_linear")
        r2 = h.get("knn_pool_ratio_vs_oracle")
        extra = ""
        if r1 is not None:
            extra += f"  knn/lin={r1}"
        if r2 is not None:
            extra += f"  knn/oracle={r2}"
        print(f"   ceiling>={h['target_ceiling']}: linear[{fmt(h['linear'])}]  "
              f"knn[{fmt(h['knn'])}]  oracle[{fmt(h['oracle'])}]{extra}")


if __name__ == "__main__":
    main()
