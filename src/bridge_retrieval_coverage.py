"""
Complete-coverage rate: a prover-relevant tightening of the q-0011 retrieval proxy.

The entire e-0012..e-0017 chain measured MEAN premise recall vs candidate budget --
the average FRACTION of a theorem's needed premises that land in the retrieved set.
Every answer (a-0013..a-0016) closes with the same honest caveat: "still a premise-recall
proxy, not a measured prover success rate."  This script narrows that gap without any
proving run, by changing only the metric.

A retrieval-augmented prover can only make a cross-domain step if EVERY premise that step
needs is in its context window; one missing premise breaks the step regardless of how many
others were retrieved.  So mean recall is too forgiving: recall 0.90 can mean "90% of each
theorem's premises retrieved" while almost NO theorem gets its COMPLETE set.  The
prover-relevant quantity is the COMPLETE-COVERAGE RATE: the fraction of cross-domain test
theorems for which ALL clustered premises are inside the top-B retrieved declarations.
This is a strict necessary condition for retrieval-augmented proving (necessary, not
sufficient -- having the premises doesn't guarantee the prover finds the proof), so it
is a genuine upper bound on the fraction of cross-domain theorems such a prover could solve.

Everything else is held identical to e-0016 (bridge_retrieval_content.py): same bridge
router (home + top-5 bridge-neighbour clusters), same candidate accounting (budget B =
number of individual decls ranked), same 432 held-out cross-domain TEST theorems, same
three intra-cluster rankers (content = IDF-weighted query-name token overlap, popularity =
train-citation prior, random = exact combinatorial mean), same leakage control (W, IDF,
popularity from train / name-vocab only).  ONLY the reported metric changes: mean recall ->
complete-coverage rate.

Random arm: for a theorem with k recoverable premises in a routed pool of size P, the
probability that a uniform random prefix of B holds ALL k is the hypergeometric
C(P-k, B-k) / C(P, B) when B>=k else 0 -- computed exactly (no sampling).

Output: results/bridge_retrieval_coverage.json
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
OUT = RESULTS / "bridge_retrieval_coverage.json"

K_ROUTE = 5
BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
TARGETS = [0.25, 0.50, 0.75]
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


def hyper_all(P, k, B):
    """P(random prefix of size B contains ALL k specials) in a pool of size P.
    = C(P-k, B-k) / C(P, B) for B>=k, else 0.  Numerically stable via log-gamma."""
    if k == 0:
        return 1.0
    if B < k:
        return 0.0
    if B >= P:
        return 1.0
    lg = math.lgamma
    # log[ C(P-k,B-k) / C(P,B) ] = log((P-B)! (P-k)!) - log(P! (P-B-k)!)  ... derive directly:
    # C(P-k,B-k)/C(P,B) = prod_{i=0}^{k-1} (B-i)/(P-i)
    logp = 0.0
    for i in range(k):
        logp += math.log(B - i) - math.log(P - i)
    return math.exp(logp)


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


def evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0))
                  for cid, ms in members.items()}
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    nb = len(BUDGETS)
    cov_content = [0.0] * nb
    cov_pop = [0.0] * nb
    cov_rand = [0.0] * nb
    cost = [0.0] * nb
    ceiling_cov = 0.0          # fraction of theorems whose FULL premise set is in the routed pool
    n_cross = 0
    prem_counts = []

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
        prem_counts.append(total)

        selected = {c} | set(topk_bridge.get(c, []))
        pool = []
        for cid in selected:
            pool.extend(pop_sorted.get(cid, []))
        pool_n = len(pool)
        recoverable = prem.intersection(pool)
        k = len(recoverable)
        full_in_pool = (k == total)
        if full_in_pool:
            ceiling_cov += 1

        q_tokens = tokenize(fn) - stop
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
        content_ranked = sorted(
            pool, key=lambda n: (-score.get(n, 0.0), -cite.get(n, 0), n))
        pop_ranked = sorted(pool, key=lambda n: (-cite.get(n, 0), n))

        for bi, B in enumerate(BUDGETS):
            cost[bi] += min(B, pool_n)
            # complete coverage = ALL of prem present. Can only happen if full_in_pool.
            if full_in_pool:
                if prem.issubset(content_ranked[:B]):
                    cov_content[bi] += 1
                if prem.issubset(pop_ranked[:B]):
                    cov_pop[bi] += 1
                cov_rand[bi] += hyper_all(pool_n, k, B)

    def curve(cov):
        return [(round(cost[bi] / n_cross, 1), round(cov[bi] / n_cross, 4))
                for bi in range(nb)]

    content_curve = curve(cov_content)
    pop_curve = curve(cov_pop)
    rand_curve = curve(cov_rand)
    prem_counts.sort()
    med_prem = prem_counts[len(prem_counts) // 2]

    return {
        "metric": "complete-coverage rate (fraction of cross-domain test theorems whose ENTIRE clustered premise set is in the top-B retrieved decls)",
        "n_cross_domain_test_theorems": n_cross,
        "router": f"bridge: home + top-{K_ROUTE} bridge-neighbour clusters",
        "median_premises_per_cross_domain_theorem": med_prem,
        "max_premises_per_cross_domain_theorem": prem_counts[-1],
        "whole_pool_complete_coverage_ceiling": round(ceiling_cov / n_cross, 4),
        "ceiling_note": "fraction whose FULL premise set is even inside the bridge-routed pool; nobody can exceed this at any budget",
        "note": "curve points are (mean_candidate_cost, complete_coverage_rate) over the budget grid",
        "budget_grid": BUDGETS,
        "content_curve": content_curve,
        "popularity_curve": pop_curve,
        "random_curve": rand_curve,
        "content_budget_to_reach": {f"coverage>={t}": interp_budget(content_curve, t) for t in TARGETS},
        "popularity_budget_to_reach": {f"coverage>={t}": interp_budget(pop_curve, t) for t in TARGETS},
        "random_budget_to_reach": {f"coverage>={t}": interp_budget(rand_curve, t) for t in TARGETS},
    }


def main():
    cluster, premises, split = load()
    W, cite = build_priors(cluster, premises, split)
    decl_tokens, idf, stop = build_content_index(cluster)
    res = evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop)
    res["params"] = {"k_route": K_ROUTE, "budget_grid": BUDGETS, "targets": TARGETS,
                     "df_stop_frac": DF_STOP_FRAC}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
