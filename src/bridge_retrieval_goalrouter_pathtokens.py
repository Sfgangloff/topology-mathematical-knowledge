"""
Module-PATH goal router (q-0011, follow-on to e-0030..e-0034 / a-0029..a-0031).

The whole goal-router chain (e-0030 linear, e-0031 demand de-bias, e-0032 home
bridge prior, e-0033 set-union hybrid, e-0034 k-NN) predicted the foreign
clusters a cross-domain goal needs from the goal's NAME tokens only, and every
arm hit the SAME wall (a-0029..a-0031): name/structural co-occurrence signal
exhausts, a ~3x oracle pool gap to the high-coverage tail survives, and only
type/statement CONTENT -- which corpus.jsonl does NOT expose -- could close it.

But that chain never used the ONE other signal the corpus DOES expose and that a
prover DOES have before it proves: the goal's MODULE PATH (file_path). A theorem
being written in Mathlib/Analysis/Fourier/... is a strong, leakage-safe locality
cue for which foreign clusters it will bridge to. This experiment adds module-path
tokens to the IDENTICAL goal-router harness (same 432 held-out cross-domain TEST
theorems, same train-only co-occurrence predictor, same exact ranking-free
complete-coverage CEILING metric, same oracle) and asks: does the goal's module
location -- the last available signal -- close any of the oracle gap that name
tokens could not?

Arms (only the query token set of the goal-specific predictor changes):
  * name(m) : tokens = name subtokens                  (reproduces the e-0030 goal arm)
  * path(m) : tokens = module-path subtokens only
  * both(m) : tokens = name UNION path subtokens
  * oracle(m): home + the goal's m most-demanded ACTUAL foreign clusters (leaky UB)

Leakage control: IDF, token x foreign-cluster co-occurrence counts and demand are
estimated on TRAIN only; coverage measured on held-out TEST; the only query signals
are the test theorem's own NAME and FILE_PATH, both available before proving.

Output: results/bridge_retrieval_goalrouter_pathtokens.json
"""

import json
import re
import math
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_retrieval_goalrouter import (  # noqa: E402
    DATA, RESULTS, M_SWEEP, DF_STOP_FRAC, _CAMEL,
    tokenize, load, cross_domain_set, sweep,
)

CORPUS = DATA / "corpus.jsonl"
STEPS = DATA / "tactic_steps.jsonl"
OUT = RESULTS / "bridge_retrieval_goalrouter_pathtokens.json"


def tokenize_path(file_path):
    """file_path -> set of lowercased subtokens.

    Split on '/', drop the '.lean' extension and ubiquitous container prefixes
    (.lake/packages/<pkg>), then subtokenize each path component like a name."""
    if not file_path:
        return set()
    parts = file_path.replace("\\", "/").split("/")
    parts = [p for p in parts if p not in (".lake", "packages", "src")]
    toks = set()
    for part in parts:
        part = re.sub(r"\.lean$", "", part)
        for piece in part.replace("_", ".").split("."):
            for m in _CAMEL.findall(piece):
                if len(m) >= 2:
                    toks.add(m.lower())
    return toks


def load_paths(cluster):
    """decl full_name -> file_path, for every clustered decl (corpus has all decls)."""
    path = {}
    for line in open(CORPUS):
        d = json.loads(line)
        fn = d.get("full_name")
        if fn in cluster:
            path[fn] = d.get("file_path", "")
    # tactic_steps also carries file_path for theorems with proofs; fill any gaps.
    for line in open(STEPS):
        d = json.loads(line)
        fn = d["full_name"]
        if fn in cluster and fn not in path:
            path[fn] = d.get("file_path", "")
    return path


def build_idf(cluster, decl_path, mode):
    """IDF over clustered decls; document tokens chosen per mode (name/path/both)."""
    df = collections.Counter()
    for name in cluster:
        df.update(query_tokens(name, decl_path, mode, idf=None, stop=None, raw=True))
    N = len(cluster)
    stop = {t for t, c in df.items() if c > DF_STOP_FRAC * N}
    idf = {t: math.log(N / c) for t, c in df.items() if t not in stop}
    return idf, stop


def query_tokens(fn, decl_path, mode, idf, stop, raw=False):
    """Token set for a decl under the given mode. raw=True returns the unfiltered
    token set (for df counting); otherwise restrict to non-stop, in-IDF tokens."""
    if mode == "name":
        toks = tokenize(fn)
    elif mode == "path":
        toks = tokenize_path(decl_path.get(fn, ""))
    else:  # both
        toks = tokenize(fn) | tokenize_path(decl_path.get(fn, ""))
    if raw:
        return toks
    return (toks - stop) & idf.keys()


def train_predictor(train, decl_path, mode, idf, stop):
    tok_cluster = collections.defaultdict(collections.Counter)
    tok_total = collections.Counter()
    global_demand = collections.Counter()
    for fn, c, prem_clusters in train:
        foreign = prem_clusters - {c}
        if not foreign:
            continue
        for t in query_tokens(fn, decl_path, mode, idf, stop):
            tok_total[t] += 1
            for d in foreign:
                tok_cluster[t][d] += 1
        for d in foreign:
            global_demand[d] += 1
    return tok_cluster, tok_total, global_demand


def predict(fn, decl_path, mode, tok_cluster, tok_total, idf, stop, exclude):
    score = collections.Counter()
    for t in query_tokens(fn, decl_path, mode, idf, stop):
        nt = tok_total.get(t, 0)
        if nt == 0:
            continue
        w = idf[t]
        for d, ntd in tok_cluster[t].items():
            if d in exclude:
                continue
            score[d] += w * (ntd / nt)
    return [d for d, _ in score.most_common()]


def predictor_accuracy(test, pred_cache, n):
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


def main():
    cluster, premises, split = load()
    decl_path = load_paths(cluster)
    cluster_size = collections.Counter(cluster.values())
    train = cross_domain_set(premises, split, cluster, "train")
    test = cross_domain_set(premises, split, cluster, "test")
    n = len(test)

    n_test_with_path = sum(1 for fn, _, _ in test if decl_path.get(fn))

    arms = {}
    acc = {}
    for mode in ("name", "path", "both"):
        idf, stop = build_idf(cluster, decl_path, mode)
        tok_cluster, tok_total, global_demand = train_predictor(train, decl_path, mode, idf, stop)
        pred_cache = {fn: predict(fn, decl_path, mode, tok_cluster, tok_total, idf, stop, {c})
                      for fn, c, _ in test}

        def route(fn, c, prem_clusters, m, _pc=pred_cache):
            return {c} | set(_pc[fn][:m])

        arms[mode] = sweep(test, cluster_size, route, M_SWEEP)
        acc[mode] = dict(zip(("hits", "full"), predictor_accuracy(test, pred_cache, n)))
        if mode == "name":
            demand = global_demand

    # oracle arm (identical to the base harness)
    def route_oracle(fn, c, prem_clusters, m):
        foreign = sorted(prem_clusters - {c}, key=lambda d: -demand[d])
        return {c} | set(foreign[:m])
    oracle_rows = sweep(test, cluster_size, route_oracle, M_SWEEP)

    head2head = []
    for t in (0.50, 0.60, 0.674, 0.70, 0.80, 0.90, 0.95):
        h = {"target_ceiling": t,
             "name": cheapest(arms["name"], t),
             "path": cheapest(arms["path"], t),
             "both": cheapest(arms["both"], t),
             "oracle": cheapest(oracle_rows, t)}
        nm, bo = h["name"], h["both"]
        if nm and bo and nm["mean_pool_size"] > 0:
            h["both_pool_ratio_vs_name"] = round(bo["mean_pool_size"] / nm["mean_pool_size"], 3)
        head2head.append(h)

    res = {
        "metric": "complete-coverage ceiling = fraction of cross-domain TEST theorems whose ENTIRE clustered premise set is inside the routed cluster pool (exact, ranking-free, upper bound above any intra-cluster ranker)",
        "n_cross_domain_test_theorems": n,
        "n_train_cross_domain": len(train),
        "n_test_theorems_with_known_path": n_test_with_path,
        "leakage_control": "IDF, token x foreign-cluster co-occurrence, demand estimated on TRAIN only; coverage on held-out TEST; query signals = test theorem NAME and FILE_PATH, both available before proving.",
        "predictor_accuracy": {m: acc[m]["hits"] for m in acc},
        "predictor_full_coverage": {m: acc[m]["full"] for m in acc},
        "name_sweep": arms["name"],
        "path_sweep": arms["path"],
        "both_sweep": arms["both"],
        "oracle_sweep": oracle_rows,
        "head_to_head_cheapest_pool_to_reach_ceiling": head2head,
    }
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"n_cross_domain_test = {n}  (with known path: {n_test_with_path})")
    for m in ("name", "path", "both"):
        print(f"\npredictor[{m}] top-k hits a foreign cluster:", acc[m]["hits"])
        print(f"predictor[{m}] top-k covers ALL foreign clusters:", acc[m]["full"])
    for arm in ("name_sweep", "path_sweep", "both_sweep", "oracle_sweep"):
        print(f"\n{arm}:  {'b':>3} {'ceiling':>9} {'mean_pool':>11} {'med_pool':>9}")
        for r in res[arm]:
            print(f"   {r['breadth']:>3} {r['complete_coverage_ceiling']:>9} "
                  f"{r['mean_pool_size']:>11} {r['median_pool_size']:>9}")
    print("\nhead-to-head cheapest mean pool to reach target ceiling:")
    for h in head2head:
        def fmt(x):
            return f"b={x['breadth']} pool={x['mean_pool_size']}" if x else "unreachable"
        ratio = h.get("both_pool_ratio_vs_name")
        rtxt = f"  both/name={ratio}" if ratio is not None else ""
        print(f"   ceiling>={h['target_ceiling']}: name[{fmt(h['name'])}]  "
              f"path[{fmt(h['path'])}]  both[{fmt(h['both'])}]  oracle[{fmt(h['oracle'])}]{rtxt}")


if __name__ == "__main__":
    main()
