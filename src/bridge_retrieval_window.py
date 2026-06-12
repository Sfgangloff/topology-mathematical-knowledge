"""
Prover-realistic retrieval WINDOW comparison: within-domain vs bridge-aware.

The whole q-0011 retrieval chain (e-0011..e-0026) reports budget-to-reach numbers in
the TENS OF THOUSANDS of candidate declarations (e.g. 0.90 cross-domain recall at ~19-90k
decls).  No retrieval-augmented prover holds tens of thousands of premises in its context
window; a realistic premise window is ~10-200 declarations.  So every prior answer's
"budget-to-reach" lives in a regime a prover never operates in.

This script asks the operational question at the scale a prover ACTUALLY works: at a small
retrieval window B (the number of individual premise declarations a prover can hold), how
much does a BRIDGE-AWARE retriever (home + top-5 bridge-neighbour clusters) improve over a
WITHIN-DOMAIN retriever (home cluster only) -- holding the strong content ranker fixed?

Two head-to-head arms, identical content ranker (IDF-weighted query-name token overlap,
popularity tiebreak), identical leakage control (W, IDF, citation popularity from TRAIN /
name-vocab only), identical 432 held-out cross-domain TEST theorems (plus an ALL-test view):
  - home_only : candidate pool = the theorem's home cluster only      (non-bridge prover)
  - bridge    : candidate pool = home + top-5 bridge-neighbour clusters (bridge-aware prover)

Two metrics at each window B in {10,25,50,100,200,500,1000}:
  - mean recall          : avg fraction of a theorem's clustered premises in the top-B
  - complete-coverage    : fraction of theorems whose ENTIRE clustered premise set is in top-B
                           (the strict prover-solvable necessary condition, e-0018's bar)

Note on the structural zero: for a CROSS-DOMAIN theorem, by definition >=1 premise lives
outside the home cluster, so home_only complete-coverage is identically 0 at ANY window --
a within-domain prover can NEVER fully equip a cross-domain goal.  That makes mean recall the
informative head-to-head metric for cross-domain theorems; complete-coverage is reported to
make the structural ceiling explicit and to show bridge's finite-window complete-coverage.

We also report the ALL-test-theorem view (cross + within), because within-domain theorems
are the majority and dilute the aggregate payoff -- the honest population-level number.

Output: results/bridge_retrieval_window.json
"""

import json
import re
import math
import heapq
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_window.json"

K_ROUTE = 5
WINDOWS = [10, 25, 50, 100, 200, 500, 1000]
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


def content_rank_top(pool, q_tokens, decl_tokens, idf, cite, topn):
    """Return the top-`topn` declarations of `pool` by (content score, popularity, name).
    Uses heapq.nlargest so we never full-sort the (up to ~53k) pool."""
    qi = {tok: idf.get(tok, 0.0) for tok in q_tokens}

    def key(name):
        s = 0.0
        tt = decl_tokens.get(name)
        if tt:
            for tok in qi:
                if tok in tt:
                    s += qi[tok]
        return (s, cite.get(name, 0), name)

    return heapq.nlargest(topn, pool, key=key)


def evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    # pool ordering inside a cluster doesn't matter (we re-rank), but keep deterministic
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0)) for cid, ms in members.items()}
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    nb = len(WINDOWS)
    # accumulators: [arm][cross/all] -> per-window sums
    def zeros():
        return {"recall": [0.0] * nb, "complete": [0.0] * nb}
    acc = {
        ("home_only", "cross"): zeros(), ("bridge", "cross"): zeros(),
        ("home_only", "all"): zeros(), ("bridge", "all"): zeros(),
    }
    n_cross = 0
    n_all = 0
    topn = max(WINDOWS)

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        total = len(prem)
        is_cross = ({cluster[p] for p in prem} - {c}) != set()
        n_all += 1
        if is_cross:
            n_cross += 1

        q_tokens = tokenize(fn) - stop

        for arm in ("home_only", "bridge"):
            if arm == "home_only":
                selected = {c}
            else:
                selected = {c} | set(topk_bridge.get(c, []))
            pool = []
            for cid in selected:
                pool.extend(pop_sorted.get(cid, []))
            ranked = content_rank_top(pool, q_tokens, decl_tokens, idf, cite, topn)

            groups = [("all", acc[(arm, "all")])]
            if is_cross:
                groups.append(("cross", acc[(arm, "cross")]))
            for bi, B in enumerate(WINDOWS):
                topB = set(ranked[:B])
                hit = len(prem & topB)
                rec = hit / total
                full = 1.0 if hit == total else 0.0
                for _, a in groups:
                    a["recall"][bi] += rec
                    a["complete"][bi] += full

    def curves(arm, grp, n):
        a = acc[(arm, grp)]
        return {
            "mean_recall": [round(a["recall"][bi] / n, 4) for bi in range(nb)],
            "complete_coverage": [round(a["complete"][bi] / n, 4) for bi in range(nb)],
        }

    out = {
        "metric_note": "head-to-head within-domain (home_only) vs bridge-aware (home+top-5) at a prover-realistic premise WINDOW B (individual decls), content ranker fixed",
        "windows": WINDOWS,
        "k_route": K_ROUTE,
        "n_cross_domain_test_theorems": n_cross,
        "n_all_test_theorems": n_all,
        "frac_cross_domain": round(n_cross / n_all, 4),
        "cross_domain": {
            "home_only": curves("home_only", "cross", n_cross),
            "bridge": curves("bridge", "cross", n_cross),
        },
        "all_test": {
            "home_only": curves("home_only", "all", n_all),
            "bridge": curves("bridge", "all", n_all),
        },
    }
    # uplift table (bridge - home_only) at each window, cross-domain
    h = out["cross_domain"]["home_only"]
    b = out["cross_domain"]["bridge"]
    out["cross_domain_uplift_bridge_minus_home"] = {
        "mean_recall": [round(b["mean_recall"][bi] - h["mean_recall"][bi], 4) for bi in range(nb)],
        "complete_coverage": [round(b["complete_coverage"][bi] - h["complete_coverage"][bi], 4) for bi in range(nb)],
    }
    return out


def main():
    cluster, premises, split = load()
    W, cite = build_priors(cluster, premises, split)
    decl_tokens, idf, stop = build_content_index(cluster)
    res = evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop)
    res["params"] = {"k_route": K_ROUTE, "windows": WINDOWS, "df_stop_frac": DF_STOP_FRAC,
                     "content_ranker": "IDF-weighted query-name token overlap, popularity tiebreak",
                     "leakage_control": "W, IDF, citation popularity from train/name-vocab only"}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
