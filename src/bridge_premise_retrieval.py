"""
Bridge-aware premise retrieval simulation (cheap proxy for q-0011).

Retrieval-augmented theorem provers live or die by PREMISE SELECTION: given a
goal, pick the lemmas its proof will use.  This script asks a cheap, CPU-only
question that bears directly on the untested payoff of the project ("does making
provers bridge-aware help?"):

    How much does a BRIDGE-AWARE retriever -- one allowed to reach into the
    cluster's most strongly bridge-connected neighbour clusters -- recover of a
    theorem's true premises, versus a WITHIN-DOMAIN retriever confined to the
    theorem's own cluster?

Honest, leakage-controlled design:
  * Home cluster of a theorem  = its Leiden cluster (data/clusters.json).
  * Premise set of a theorem    = every <a>Name</a> premise across its tactic
    steps that is itself a clustered node.
  * The cluster->cluster bridge weight W[c][d] (how often a proof in cluster c
    cites a premise in cluster d) is estimated on the TRAIN split ONLY.
  * Recall is then measured on held-out TEST theorems.

Retrievers compared (candidate = set of clusters the retriever may pull from):
  * home-only          : {home}
  * bridge-aware(k)    : {home} U  top-k neighbour clusters by W[home][.]
  * all-clusters       : every cluster (recall == 1, sanity upper bound)

Recall for a theorem = |premises whose cluster is in candidate| / |clustered
premises|.  Averaged uniformly over test theorems that have >=1 clustered
premise.  We also report the same restricted to CROSS-DOMAIN theorems (those
with at least one out-of-home premise), where the bridge structure actually
matters.

Output: results/bridge_premise_retrieval.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_premise_retrieval.json"

KS = [1, 2, 3, 5, 10]


def load():
    cluster = json.load(open(CLUSTERS))  # node name -> cluster id (int)
    # theorem -> (split, set(premises)) aggregated over all its tactic steps
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


def build_bridge_graph(cluster, premises, split):
    """W[c][d] = # train theorems in cluster c citing a premise in cluster d (d!=c)."""
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


def neighbours(W, c, k):
    return {d for d, _ in W[c].most_common(k)}


def evaluate(cluster, premises, split, W):
    # Precompute top-10 neighbours per cluster once.
    topk_cache = {c: [d for d, _ in W[c].most_common(max(KS))] for c in W}

    agg = {kind: {"recall_sum": 0.0, "n": 0} for kind in
           ["home"] + [f"bridge{k}" for k in KS]}
    agg_cross = {kind: {"recall_sum": 0.0, "n": 0} for kind in
                 ["home"] + [f"bridge{k}" for k in KS]}

    n_test = n_with_prem = n_cross = 0
    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        n_test += 1
        c = cluster.get(fn)
        if c is None:
            continue
        # clustered premises only
        prem_clusters = [cluster[p] for p in ps if p in cluster]
        if not prem_clusters:
            continue
        n_with_prem += 1
        total = len(prem_clusters)
        in_home = sum(1 for d in prem_clusters if d == c)
        is_cross = in_home < total
        if is_cross:
            n_cross += 1

        nbrs = topk_cache.get(c, [])
        for kind, cand in (
            [("home", {c})] +
            [(f"bridge{k}", {c} | set(nbrs[:k])) for k in KS]
        ):
            rec = sum(1 for d in prem_clusters if d in cand) / total
            agg[kind]["recall_sum"] += rec
            agg[kind]["n"] += 1
            if is_cross:
                agg_cross[kind]["recall_sum"] += rec
                agg_cross[kind]["n"] += 1

    def finalize(a):
        return {kind: round(v["recall_sum"] / v["n"], 4) if v["n"] else None
                for kind, v in a.items()}

    return {
        "n_test_theorems": n_test,
        "n_test_with_clustered_premises": n_with_prem,
        "n_test_cross_domain": n_cross,
        "frac_cross_domain": round(n_cross / n_with_prem, 4) if n_with_prem else None,
        "mean_recall_all": finalize(agg),
        "mean_recall_cross_domain_only": finalize(agg_cross),
    }


def main():
    cluster, premises, split = load()
    W = build_bridge_graph(cluster, premises, split)
    res = evaluate(cluster, premises, split, W)
    res["params"] = {"ks": KS, "n_clusters": len(set(cluster.values()))}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
