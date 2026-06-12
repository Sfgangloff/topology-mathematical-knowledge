"""
Router ablation: does BRIDGE routing still earn its keep once the intra-cluster ranker
is the strong content ranker?  (q-0011)

The entire e-0012..e-0016 chain held the ROUTER fixed (bridge: home + top-5 bridge-neighbour
clusters) and varied the intra-cluster ranker, concluding (a-0015) that a content-aware
ranker -- IDF-weighted overlap of the query theorem's NAME tokens with candidate decl
names -- cracks the cross-domain premise tail (0.90 recall at ~19.5k candidates vs the
popularity prior's ~47.9k).

That leaves the dual control untested: hold the strong CONTENT ranker fixed and vary the
ROUTER.  If a content ranker over the WHOLE library reaches the same cross-domain recall at
the same candidate budget as the bridge-routed pool, then bridge structure adds nothing
once content ranking is good -- the content ranker subsumes the router, an honest bound
that would temper the bridge claim.  If bridge routing lets the content ranker reach a
target recall at a far smaller budget than whole-library content ranking, the bridge
structure genuinely prunes the search space.

Four arms -- SAME content ranker, SAME 432 held-out cross-domain TEST theorems, SAME
candidate accounting (budget B = number of individual decls ranked).  Only the candidate
POOL (the router) changes:

    home_only   : pool = home cluster only.                       (no cross-domain routing)
    bridge      : pool = home + top-5 bridge-neighbour clusters.  (the e-0016 router)
    popularity  : pool = home + top-5 globally most-cited clusters.(hub router, content-ranked)
    whole_lib   : pool = all 138 clusters.                        (no routing; ranker does all)

Cross-domain premise recall is measured vs candidate budget for each arm; we interpolate
the budget each needs to hit 0.70/0.80/0.90 recall, and report each arm's recall ceiling
(fraction of needed premises that live anywhere in the arm's pool -- 1.0 for whole_lib by
construction, <1 for the routed arms that may exclude a needed cluster).

Leakage control identical to e-0016: bridge weights W, citation popularity, IDF vocabulary
all from the TRAIN split / name vocabulary only; recall on TEST.  The query signal is the
test theorem's own NAME, which a prover always has.

Output: results/bridge_retrieval_router_ablation.json
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
OUT = RESULTS / "bridge_retrieval_router_ablation.json"

K_ROUTE = 5
BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 92594]
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


def build_priors(cluster, premises, split):
    """TRAIN-only bridge weights W[c][d], per-decl citation popularity, per-cluster mass."""
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
    # global cluster popularity = total train citations landing on the cluster's members
    cluster_mass = collections.Counter()
    for p, n in cite.items():
        cid = cluster.get(p)
        if cid is not None:
            cluster_mass[cid] += n
    return W, cite, cluster_mass


def build_content_index(cluster):
    decl_tokens = {}
    df = collections.Counter()
    for name in cluster:
        t = tokenize(name)
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


ARMS = ["home_only", "bridge", "popularity", "whole_lib"]


def evaluate(cluster, premises, split, W, cite, cluster_mass, decl_tokens, idf, stop):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    all_clusters = list(members.keys())
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}
    pop_clusters_global = [cid for cid, _ in cluster_mass.most_common()]

    nb = len(BUDGETS)
    rec = {a: [0.0] * nb for a in ARMS}
    cost = {a: [0.0] * nb for a in ARMS}
    ceiling = {a: 0.0 for a in ARMS}
    pool_n_sum = {a: 0.0 for a in ARMS}
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
        if {cluster[p] for p in prem} <= {c}:
            continue
        n_cross += 1
        total = len(prem)

        # router -> selected cluster set per arm
        bridge_sel = {c} | set(topk_bridge.get(c, []))
        pop_sel = {c}
        for cid in pop_clusters_global:
            if len(pop_sel) >= K_ROUTE + 1:
                break
            pop_sel.add(cid)
        selected = {
            "home_only": {c},
            "bridge": bridge_sel,
            "popularity": pop_sel,
            "whole_lib": set(all_clusters),
        }

        # query content scores computed once over the union of all pooled decls
        q_tokens = tokenize(fn) - stop
        union = set()
        for s in selected.values():
            for cid in s:
                union.update(members[cid])
        score = {}
        if q_tokens:
            for name in union:
                tt = decl_tokens.get(name)
                if not tt:
                    continue
                sc = 0.0
                for tok in q_tokens:
                    if tok in tt:
                        sc += idf.get(tok, 0.0)
                if sc:
                    score[name] = sc

        for arm in ARMS:
            pool = []
            for cid in selected[arm]:
                pool.extend(members[cid])
            pool_n = len(pool)
            pool_n_sum[arm] += pool_n
            recoverable = prem.intersection(pool)
            ceiling[arm] += len(recoverable) / total
            ranked = sorted(pool, key=lambda n: (-score.get(n, 0.0), -cite.get(n, 0), n))
            for bi, B in enumerate(BUDGETS):
                cost[arm][bi] += min(B, pool_n)
                rec[arm][bi] += len(prem.intersection(ranked[:B])) / total

    def curve(arm):
        return [(round(cost[arm][bi] / n_cross, 1), round(rec[arm][bi] / n_cross, 4))
                for bi in range(nb)]

    out = {
        "n_cross_domain_test_theorems": n_cross,
        "intra_cluster_ranker": "FIXED: IDF-weighted query-name token overlap (content), pop tiebreak",
        "ablated_dimension": "the ROUTER (candidate pool); ranker held fixed across all arms",
        "k_route": K_ROUTE,
        "note": "curve points are (mean_candidate_cost, mean_cross_domain_premise_recall)",
        "budget_grid": BUDGETS,
        "arms": {},
    }
    for arm in ARMS:
        cv = curve(arm)
        out["arms"][arm] = {
            "mean_pool_size": round(pool_n_sum[arm] / n_cross, 1),
            "recall_ceiling": round(ceiling[arm] / n_cross, 4),
            "curve": cv,
            "budget_to_reach": {f"recall>={t}": interp_budget(cv, t) for t in TARGETS},
        }
    return out


def main():
    cluster, premises, split = load()
    W, cite, cluster_mass = build_priors(cluster, premises, split)
    decl_tokens, idf, stop = build_content_index(cluster)
    res = evaluate(cluster, premises, split, W, cite, cluster_mass,
                   decl_tokens, idf, stop)
    res["params"] = {"k_route": K_ROUTE, "budget_grid": BUDGETS, "targets": TARGETS,
                     "df_stop_frac": DF_STOP_FRAC, "vocab_size_after_stop": len(idf)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
