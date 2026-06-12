"""
Baseline controls for the bridge-aware premise-retrieval simulation (q-0011).

bridge_premise_retrieval.py showed a bridge-aware retriever (home cluster + the
top-k neighbour clusters ranked by the train-estimated bridge weight W[home][.])
lifts cross-domain premise recall from 61.9% (home only) to 88.2% at k=3.  But
that number alone does NOT establish that the *bridge structure* is informative:
maybe adding ANY k extra clusters helps just as much.  This script adds the
controls the original lacked, asking whether bridge-weighted neighbour selection
beats naive alternatives at the SAME budget k:

    * home            : {home}                          (lower bound)
    * bridge(k)       : {home} U top-k by W[home][.]    (the proposed method)
    * popularity(k)   : {home} U top-k globally most-cited clusters
                        (the same k clusters for every theorem; "just add the
                        biggest hubs")
    * random(k)       : {home} U k random clusters, averaged over RANDOM_TRIALS
                        deterministic seeds (chance baseline)
    * all-clusters    : recall == 1 (sanity upper bound)

If bridge(k) clearly beats popularity(k) and random(k) at matched k, the per-home
bridge ranking carries real cross-domain signal.  If it merely ties popularity,
the gain is just "more clusters" and the bridge graph adds nothing.

Leakage control is identical to the original: W and the popularity ranking are
estimated on TRAIN theorems only; recall is measured on held-out TEST theorems.

Output: results/bridge_retrieval_baselines.json
"""

import json
import random
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_baselines.json"

KS = [1, 2, 3, 5, 10]
RANDOM_TRIALS = 20
RANDOM_SEED = 42


def load():
    cluster = json.load(open(CLUSTERS))  # node name -> cluster id (int)
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
    """W[c][d] = # train theorems in cluster c citing a premise in cluster d (d!=c).
    pop[d] = total train cross-cluster citations landing in d (global popularity)."""
    W = collections.defaultdict(lambda: collections.Counter())
    pop = collections.Counter()
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
            pop[d] += 1
    return W, pop


def evaluate(cluster, premises, split, W, pop):
    all_clusters = list(set(cluster.values()))
    topk_bridge = {c: [d for d, _ in W[c].most_common(max(KS))] for c in W}
    pop_ranked = [d for d, _ in pop.most_common()]  # global popularity order

    rng = random.Random(RANDOM_SEED)
    # Pre-draw RANDOM_TRIALS independent random cluster orderings (shared across
    # theorems within a trial; a fresh sample per theorem would only lower variance).
    random_orders = [rng.sample(all_clusters, len(all_clusters))
                     for _ in range(RANDOM_TRIALS)]

    methods = ["home"] + [f"bridge{k}" for k in KS] + \
              [f"pop{k}" for k in KS] + [f"rand{k}" for k in KS]
    agg = {m: {"s": 0.0, "n": 0} for m in methods}
    agg_cross = {m: {"s": 0.0, "n": 0} for m in methods}

    n_test = n_with_prem = n_cross = 0
    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        n_test += 1
        c = cluster.get(fn)
        if c is None:
            continue
        prem_clusters = [cluster[p] for p in ps if p in cluster]
        if not prem_clusters:
            continue
        n_with_prem += 1
        total = len(prem_clusters)
        in_home = sum(1 for d in prem_clusters if d == c)
        is_cross = in_home < total
        if is_cross:
            n_cross += 1

        br = topk_bridge.get(c, [])
        # popularity / random candidates must exclude the home cluster so all
        # methods spend their k budget on *additional* clusters.
        pop_extra = [d for d in pop_ranked if d != c]

        cands = {"home": {c}}
        for k in KS:
            cands[f"bridge{k}"] = {c} | set(br[:k])
            cands[f"pop{k}"] = {c} | set(pop_extra[:k])

        def record(method, cand):
            rec = sum(1 for d in prem_clusters if d in cand) / total
            agg[method]["s"] += rec
            agg[method]["n"] += 1
            if is_cross:
                agg_cross[method]["s"] += rec
                agg_cross[method]["n"] += 1

        for method, cand in cands.items():
            record(method, cand)

        # random: average recall over the trials, then record once per theorem.
        for k in KS:
            recs = recs_cross = 0.0
            for order in random_orders:
                extra = [d for d in order if d != c][:k]
                cand = {c} | set(extra)
                recs += sum(1 for d in prem_clusters if d in cand) / total
            mean_rec = recs / RANDOM_TRIALS
            agg[f"rand{k}"]["s"] += mean_rec
            agg[f"rand{k}"]["n"] += 1
            if is_cross:
                agg_cross[f"rand{k}"]["s"] += mean_rec
                agg_cross[f"rand{k}"]["n"] += 1

    def finalize(a):
        return {m: round(v["s"] / v["n"], 4) if v["n"] else None
                for m, v in a.items()}

    return {
        "n_test_theorems": n_test,
        "n_test_with_clustered_premises": n_with_prem,
        "n_test_cross_domain": n_cross,
        "frac_cross_domain": round(n_cross / n_with_prem, 4) if n_with_prem else None,
        "mean_recall_all": finalize(agg),
        "mean_recall_cross_domain_only": finalize(agg_cross),
        "top_global_popularity_clusters": pop.most_common(5),
    }


def main():
    cluster, premises, split = load()
    W, pop = build_bridge_graph(cluster, premises, split)
    res = evaluate(cluster, premises, split, W, pop)
    res["params"] = {"ks": KS, "random_trials": RANDOM_TRIALS,
                     "random_seed": RANDOM_SEED,
                     "n_clusters": len(set(cluster.values()))}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
