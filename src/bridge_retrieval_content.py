"""
Content-aware intra-cluster premise ranking: the remaining lever named by a-0014 (q-0011).

e-0015 fixed the bridge cluster routing (home + top-5 bridge-neighbour clusters) and
ranked individual declarations INSIDE the routed clusters by a leakage-safe train-citation
POPULARITY prior.  Result (a-0014): popularity is a real lever at moderate recall (0.70 at
~12.8k candidates vs random's ~40.1k) but collapses near the ceiling -- 0.90 premise recall
still needs ~48.5k of ~53.8k candidates (~90% of the pool), barely better than random,
because "the last ~20% of cross-domain premises are rarely-cited specialized lemmas ...
a popularity prior cannot surface them cheaply."  Its closing line:

    "...a content-aware intra-cluster ranker is the remaining lever."

This script tests exactly that.  Same bridge routing, same candidate accounting, same
held-out cross-domain TEST theorems as e-0015 -- the ONLY thing changed is the
intra-cluster ordering.  Instead of popularity we rank candidate declarations by
CONTENT similarity to the query theorem: lexical overlap between the query theorem's
NAME tokens and each candidate declaration's name tokens, IDF-weighted.

Why name tokens?  The corpus exposes no type/statement text (tactic_steps.jsonl and
corpus.jsonl carry only full_name + file_path), but Lean/Mathlib declaration names are
densely semantic (`GromovHausdorff.ghDist_le_hausdorffDist`, `IsCompact.inter`).  A
bridge theorem's name typically carries tokens from BOTH domains it links, so a
specialized cross-domain premise -- even one almost never cited -- often shares rare,
high-IDF name tokens with the theorem that uses it.  Popularity is blind to this; a
content ranker is not.  This is the only leakage-free content signal available, and it
needs nothing the query prover wouldn't already have (the goal theorem's own name).

Tokenization: split full_name on '.', camelCase boundaries, '_' and digits; lowercase.
IDF over all clustered declarations; tokens appearing in >20% of decls are dropped as
stop-tokens (eq, of, is, ...).  Content score(decl) = sum of IDF over tokens shared with
the query name; ties broken by the popularity prior (so the content arm is "content
first, popularity as fallback" -- a strict superset of information over e-0015).

Three arms over the SAME bridge-routed candidate set:
    * content    : declarations ranked by IDF-weighted name overlap with the query (+pop tiebreak).
    * popularity : the e-0015 prior (recomputed here as an in-script reference; matches e-0015).
    * random     : exact hypergeometric mean recall of a random prefix (control).

Leakage control: IDF/popularity/W all come from name vocabulary + the TRAIN split only;
recall is measured on TEST theorems.  The query signal is the test theorem's NAME, which
a prover trying to prove it always has.

If content ranking reaches the 0.90 ceiling at a budget far below popularity's ~48.5k,
content IS the cheap tail lever a-0014 was missing.  If it does not, the cross-domain tail
is intrinsically expensive even with content -- an equally honest, publishable result.

Output: results/bridge_retrieval_content.json
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
OUT = RESULTS / "bridge_retrieval_content.json"

K_ROUTE = 5
BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
TARGETS = [0.70, 0.80, 0.90]
DF_STOP_FRAC = 0.20          # drop name tokens appearing in >20% of declarations

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    """full_name -> set of lowercased subtokens (split on '.', '_', camelCase, digits)."""
    toks = set()
    for part in name.replace("_", ".").split("."):
        for m in _CAMEL.findall(part):
            if len(m) >= 2:                       # drop 1-char noise
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


def build_priors(cluster, premises, split):
    """TRAIN-only bridge weights W[c][d] and citation popularity prior cite[p]."""
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
    """Per-decl token sets + IDF over all clustered declarations (vocabulary stats only)."""
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
    # strip stop-tokens from each decl's set so scoring only touches informative tokens
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


def evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0))
                  for cid, ms in members.items()}
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    nb = len(BUDGETS)
    rec_content = [0.0] * nb
    rec_pop = [0.0] * nb
    rec_rand = [0.0] * nb
    cost = [0.0] * nb
    pool_size_sum = 0.0
    ceiling_sum = 0.0
    n_cross = 0
    # diagnostic: of the premises content surfaces in its top-2000 but popularity does
    # not, how rarely-cited are they? (does content reach the unpopular tail?)
    content_tail_gain_cites = []

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        if {cluster[p] for p in prem} <= {c}:        # within-domain only; skip
            continue
        n_cross += 1
        total = len(prem)

        selected = {c} | set(topk_bridge.get(c, []))
        pool = []
        for cid in selected:
            pool.extend(pop_sorted.get(cid, []))     # already de-dup: clusters disjoint
        pool_n = len(pool)
        pool_size_sum += pool_n
        recoverable = prem.intersection(pool)
        ceiling_sum += len(recoverable) / total

        # --- content scores over the pool ---
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
        # content ranking: score desc, tiebreak popularity desc, then name (stable)
        content_ranked = sorted(
            pool, key=lambda n: (-score.get(n, 0.0), -cite.get(n, 0), n))
        # popularity ranking (e-0015 reference)
        pop_ranked = sorted(pool, key=lambda n: (-cite.get(n, 0), n))

        for bi, B in enumerate(BUDGETS):
            cost[bi] += min(B, pool_n)
            rec_content[bi] += len(prem.intersection(content_ranked[:B])) / total
            rec_pop[bi] += len(prem.intersection(pop_ranked[:B])) / total
            frac = (min(B, pool_n) / pool_n) if pool_n else 0.0
            rec_rand[bi] += (len(recoverable) * frac) / total

        # diagnostic at B=2000: premises content gets but popularity misses, and their cites
        B = 2000
        c_hit = prem.intersection(content_ranked[:B])
        p_hit = prem.intersection(pop_ranked[:B])
        for p in (c_hit - p_hit):
            content_tail_gain_cites.append(cite.get(p, 0))

    def curve(rec):
        return [(round(cost[bi] / n_cross, 1), round(rec[bi] / n_cross, 4))
                for bi in range(nb)]

    content_curve = curve(rec_content)
    pop_curve = curve(rec_pop)
    rand_curve = curve(rec_rand)
    tail = sorted(content_tail_gain_cites)
    med = tail[len(tail) // 2] if tail else None

    return {
        "n_cross_domain_test_theorems": n_cross,
        "router": f"bridge: home + top-{K_ROUTE} bridge-neighbour clusters",
        "intra_cluster_ranker": "IDF-weighted query-name token overlap (content), pop tiebreak",
        "mean_whole_pool_candidate_cost": round(pool_size_sum / n_cross, 1),
        "whole_pool_premise_recall_ceiling": round(ceiling_sum / n_cross, 4),
        "note": "curve points are (mean_candidate_cost, mean_premise_recall) for the budget grid",
        "budget_grid": BUDGETS,
        "content_curve": content_curve,
        "popularity_curve": pop_curve,
        "random_curve": rand_curve,
        "content_budget_to_reach": {f"recall>={t}": interp_budget(content_curve, t) for t in TARGETS},
        "popularity_budget_to_reach": {f"recall>={t}": interp_budget(pop_curve, t) for t in TARGETS},
        "random_budget_to_reach": {f"recall>={t}": interp_budget(rand_curve, t) for t in TARGETS},
        "diagnostic_B2000_content_minus_popularity": {
            "n_premises_content_surfaces_that_popularity_misses": len(tail),
            "median_train_citation_count_of_those_premises": med,
            "note": "premises in content's top-2000 but NOT popularity's top-2000; low cites = the rarely-cited tail a-0014 said popularity cannot reach",
        },
    }


def main():
    cluster, premises, split = load()
    W, cite = build_priors(cluster, premises, split)
    decl_tokens, idf, stop = build_content_index(cluster)
    res = evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop)
    res["params"] = {"k_route": K_ROUTE, "budget_grid": BUDGETS, "targets": TARGETS,
                     "df_stop_frac": DF_STOP_FRAC, "vocab_size_after_stop": len(idf),
                     "n_stop_tokens": len(stop)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
