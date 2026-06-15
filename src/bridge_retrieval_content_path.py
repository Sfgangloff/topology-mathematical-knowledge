"""
Path-augmented content intra-cluster premise ranking (q-0011; dual to e-0036/pathtokens
goal router, follow-on to e-0016 / a-0015).

e-0016 (a-0015) found the strongest intra-cluster ranker: rank candidate declarations in
the bridge-routed pool by IDF-weighted overlap between the QUERY theorem's NAME tokens and
each candidate's NAME tokens.  It cut 0.90 cross-domain premise recall to ~19.5k candidates
(2.46x cheaper than popularity).  But it used ONLY name tokens.  The module-PATH goal-router
experiment (bridge_retrieval_goalrouter_pathtokens.py) showed module path is leakage-safe
and available to a prover before it proves, yet did NOT close the oracle gap at the cluster
ROUTING level.  The symmetric question was never asked: does the goal's (and each
candidate's) MODULE PATH help at the intra-cluster RANKING level -- i.e. when ordering
individual declarations inside the already-routed pool?

Intuition cuts both ways and is worth measuring:
  + Two declarations in nearby sub-modules (Mathlib/Topology/MetricSpace/...) are often
    related even when their names diverge, so path tokens could surface rarely-cited
    specialized premises name overlap misses -- the cross-domain tail a-0014/a-0015 care about.
  - But cross-domain premises live in FOREIGN modules by construction, so their path tokens
    overlap the query's path LESS than home decls do; path signal could instead pull the
    ranking toward same-domain (home) declarations and HURT cross-domain recall.

This script holds e-0016 fixed (same bridge router home+top-5, same 432 held-out cross-domain
TEST theorems, same candidate accounting, same leakage control) and changes ONLY the
intra-cluster token bag:
  * content_name : query & candidate tokens = NAME subtokens          (reproduces e-0016)
  * content_both : query & candidate tokens = NAME UNION PATH subtokens
  * popularity   : train-citation prior (e-0015 reference)
  * random       : exact hypergeometric mean recall (control)

IDF/stop vocabulary is rebuilt per arm over all clustered decls; popularity/W from TRAIN
only; recall on TEST.  Query signals (test theorem NAME and FILE_PATH) are both available
to a prover before it proves the goal.

If content_both reaches 0.90 cross-domain recall at a budget below content_name's ~19.5k,
module path is a free intra-cluster lever beyond names.  If it ties or loses, the corpus's
locality signal is already saturated by name tokens / foreign premises sit too far in path
space -- an equally honest bound that further tightens a-0015.

Output: results/bridge_retrieval_content_path.json
"""

import json
import re
import math
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CORPUS = DATA / "corpus.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_content_path.json"

K_ROUTE = 5
BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
TARGETS = [0.70, 0.80, 0.90]
DF_STOP_FRAC = 0.20

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    toks = set()
    for part in name.replace("_", ".").split("."):
        for m in _CAMEL.findall(part):
            if len(m) >= 2:
                toks.add(m.lower())
    return toks


def tokenize_path(file_path):
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


def load_paths(cluster):
    path = {}
    for line in open(CORPUS):
        d = json.loads(line)
        fn = d.get("full_name")
        if fn in cluster:
            path[fn] = d.get("file_path", "")
    for line in open(STEPS):
        d = json.loads(line)
        fn = d["full_name"]
        if fn in cluster and fn not in path:
            path[fn] = d.get("file_path", "")
    return path


def build_priors(cluster, premises, split):
    W = collections.defaultdict(lambda: collections.Counter())
    cite = collections.Counter()
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        c = cluster.get(fn)
        for p in ps:
            cite[p] += 1
        if c is None:
            continue
        seen = set()
        for p in ps:
            d = cluster.get(p)
            if d is None or d == c or d in seen:
                continue
            seen.add(d)
            W[c][d] += 1
    return W, cite


def build_content_index(cluster, decl_path, use_path):
    """Per-decl token sets + IDF over all clustered decls. use_path=True unions path tokens."""
    decl_tokens = {}
    df = collections.Counter()
    for name in cluster:
        t = tokenize(name)
        if use_path:
            t = t | tokenize_path(decl_path.get(name, ""))
        decl_tokens[name] = t
        for tok in t:
            df[tok] += 1
    N = len(cluster)
    stop = {tok for tok, c in df.items() if c > DF_STOP_FRAC * N}
    idf = {tok: math.log(N / c) for tok, c in df.items() if tok not in stop}
    decl_tokens = {n: (t - stop) for n, t in decl_tokens.items()}
    return decl_tokens, idf, stop


def interp_budget(curve, target):
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


def rank_score(pool, q_tokens, decl_tokens, idf, cite):
    score = {}
    if q_tokens:
        for name in pool:
            s = 0.0
            tt = decl_tokens.get(name)
            if tt:
                for tok in q_tokens:
                    if tok in tt:
                        s += idf.get(tok, 0.0)
            if s:
                score[name] = s
    return sorted(pool, key=lambda n: (-score.get(n, 0.0), -cite.get(n, 0), n))


def evaluate(cluster, premises, split, W, cite, decl_path,
             dt_name, idf_name, stop_name, dt_both, idf_both, stop_both):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0))
                  for cid, ms in members.items()}
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    nb = len(BUDGETS)
    rec_name = [0.0] * nb
    rec_both = [0.0] * nb
    rec_pop = [0.0] * nb
    rec_rand = [0.0] * nb
    cost = [0.0] * nb
    pool_size_sum = 0.0
    ceiling_sum = 0.0
    n_cross = 0
    n_with_path = 0
    both_tail_gain_cites = []   # premises content_both gets at B=2000 that content_name misses

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        if {cluster[p] for p in prem} <= {c}:
            continue
        n_cross += 1
        if decl_path.get(fn):
            n_with_path += 1
        total = len(prem)

        selected = {c} | set(topk_bridge.get(c, []))
        pool = []
        for cid in selected:
            pool.extend(pop_sorted.get(cid, []))
        pool_n = len(pool)
        pool_size_sum += pool_n
        recoverable = prem.intersection(pool)
        ceiling_sum += len(recoverable) / total

        q_name = tokenize(fn) - stop_name
        q_both = (tokenize(fn) | tokenize_path(decl_path.get(fn, ""))) - stop_both
        name_ranked = rank_score(pool, q_name, dt_name, idf_name, cite)
        both_ranked = rank_score(pool, q_both, dt_both, idf_both, cite)
        pop_ranked = sorted(pool, key=lambda n: (-cite.get(n, 0), n))

        for bi, B in enumerate(BUDGETS):
            cost[bi] += min(B, pool_n)
            rec_name[bi] += len(prem.intersection(name_ranked[:B])) / total
            rec_both[bi] += len(prem.intersection(both_ranked[:B])) / total
            rec_pop[bi] += len(prem.intersection(pop_ranked[:B])) / total
            frac = (min(B, pool_n) / pool_n) if pool_n else 0.0
            rec_rand[bi] += (len(recoverable) * frac) / total

        B = 2000
        b_hit = prem.intersection(both_ranked[:B])
        n_hit = prem.intersection(name_ranked[:B])
        for p in (b_hit - n_hit):
            both_tail_gain_cites.append(cite.get(p, 0))

    def curve(rec):
        return [(round(cost[bi] / n_cross, 1), round(rec[bi] / n_cross, 4))
                for bi in range(nb)]

    name_curve = curve(rec_name)
    both_curve = curve(rec_both)
    pop_curve = curve(rec_pop)
    rand_curve = curve(rec_rand)
    tail = sorted(both_tail_gain_cites)
    med = tail[len(tail) // 2] if tail else None

    return {
        "n_cross_domain_test_theorems": n_cross,
        "n_test_theorems_with_file_path": n_with_path,
        "router": f"bridge: home + top-{K_ROUTE} bridge-neighbour clusters",
        "mean_whole_pool_candidate_cost": round(pool_size_sum / n_cross, 1),
        "whole_pool_premise_recall_ceiling": round(ceiling_sum / n_cross, 4),
        "note": "curve points are (mean_candidate_cost, mean_premise_recall) over the budget grid",
        "budget_grid": BUDGETS,
        "content_name_curve": name_curve,
        "content_both_curve": both_curve,
        "popularity_curve": pop_curve,
        "random_curve": rand_curve,
        "content_name_budget_to_reach": {f"recall>={t}": interp_budget(name_curve, t) for t in TARGETS},
        "content_both_budget_to_reach": {f"recall>={t}": interp_budget(both_curve, t) for t in TARGETS},
        "popularity_budget_to_reach": {f"recall>={t}": interp_budget(pop_curve, t) for t in TARGETS},
        "diagnostic_B2000_both_minus_name": {
            "n_premises_path_aug_surfaces_that_name_misses": len(tail),
            "median_train_citation_count_of_those_premises": med,
            "note": "premises in content_both's top-2000 but NOT content_name's; gain attributable to path tokens",
        },
    }


def main():
    cluster, premises, split = load()
    decl_path = load_paths(cluster)
    W, cite = build_priors(cluster, premises, split)
    dt_name, idf_name, stop_name = build_content_index(cluster, decl_path, use_path=False)
    dt_both, idf_both, stop_both = build_content_index(cluster, decl_path, use_path=True)
    res = evaluate(cluster, premises, split, W, cite, decl_path,
                   dt_name, idf_name, stop_name, dt_both, idf_both, stop_both)
    res["params"] = {"k_route": K_ROUTE, "budget_grid": BUDGETS, "targets": TARGETS,
                     "df_stop_frac": DF_STOP_FRAC,
                     "name_vocab_after_stop": len(idf_name),
                     "both_vocab_after_stop": len(idf_both)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
