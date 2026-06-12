"""
Specialist-stratified control for the bridge-aware premise-retrieval simulation (q-0011).

bridge_retrieval_baselines.py (e-0012) showed that, in aggregate, a bridge-aware
retriever ({home} U top-k neighbour clusters ranked by the train bridge weight
W[home][.]) barely beats a popularity baseline ({home} U the k globally most-cited
clusters): per-home bridge ranking wins by only <=1.2pt.  Conclusion so far:
bridges help premise selection mostly by routing to a few integrative hubs, not
via theorem-specific structure.

But that aggregate hides a confound.  A popularity retriever sends EVERY theorem to
the SAME global hubs.  For the many cross-domain theorems whose out-of-home premises
happen to live in those hubs, popularity and bridge necessarily agree.  The bridge
graph can only add theorem-specific value for theorems that need a SPECIALIZED
neighbour cluster - one NOT among the global top-k hubs.  If those exist and bridge(k)
beats pop(k) decisively on them, then the bridge structure does carry per-theorem
routing signal; it is just diluted in the mean by the hub-served majority.

This script stratifies the held-out cross-domain test theorems, at each budget k, into:

    * hub-served(k)        : every out-of-home premise cluster is within the global
                             top-k most-cited clusters (popularity already covers them)
    * specialist-needing(k): at least one out-of-home premise cluster lies OUTSIDE the
                             global top-k hubs (popularity cannot reach it; only a
                             per-home bridge ranking might)

and reports home / bridge(k) / pop(k) recall within each stratum.  Leakage control is
identical to e-0012: W and the popularity ranking come from TRAIN theorems only; recall
is measured on TEST theorems.

Output: results/bridge_retrieval_specialist.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_specialist.json"

KS = [1, 2, 3, 5, 10]


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
    topk_bridge = {c: [d for d, _ in W[c].most_common(max(KS))] for c in W}
    pop_ranked = [d for d, _ in pop.most_common()]

    # accumulators keyed by (k, stratum, method) -> {s, n}
    strata = ["hub_served", "specialist"]
    methods = ["home", "bridge", "pop"]
    agg = {k: {s: {m: {"s": 0.0, "n": 0} for m in methods} for s in strata}
           for k in KS}
    counts = {k: {s: 0 for s in strata} for k in KS}

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem_clusters = [cluster[p] for p in ps if p in cluster]
        if not prem_clusters:
            continue
        total = len(prem_clusters)
        out_home = set(d for d in prem_clusters if d != c)
        if not out_home:  # within-domain theorem; skip (cross-domain analysis)
            continue

        br = topk_bridge.get(c, [])
        pop_extra = [d for d in pop_ranked if d != c]

        for k in KS:
            hub_set = set(pop_extra[:k])  # global top-k hubs for this budget
            # stratify on whether popularity could in principle reach every need
            stratum = "hub_served" if out_home <= hub_set else "specialist"
            counts[k][stratum] += 1

            cand_home = {c}
            cand_bridge = {c} | set(br[:k])
            cand_pop = {c} | hub_set
            for method, cand in (("home", cand_home),
                                 ("bridge", cand_bridge),
                                 ("pop", cand_pop)):
                rec = sum(1 for d in prem_clusters if d in cand) / total
                a = agg[k][stratum][method]
                a["s"] += rec
                a["n"] += 1

    def finalize():
        out = {}
        for k in KS:
            out[str(k)] = {}
            for s in strata:
                out[str(k)][s] = {
                    "n": counts[k][s],
                    "recall": {m: (round(agg[k][s][m]["s"] / agg[k][s][m]["n"], 4)
                                   if agg[k][s][m]["n"] else None)
                               for m in methods},
                }
                rb = out[str(k)][s]["recall"]["bridge"]
                rp = out[str(k)][s]["recall"]["pop"]
                out[str(k)][s]["bridge_minus_pop"] = (
                    round(rb - rp, 4) if rb is not None and rp is not None else None)
        return out

    return {"by_k": finalize(),
            "top_global_popularity_clusters": pop.most_common(10)}


def main():
    cluster, premises, split = load()
    W, pop = build_bridge_graph(cluster, premises, split)
    res = evaluate(cluster, premises, split, W, pop)
    res["params"] = {"ks": KS, "n_clusters": len(set(cluster.values())),
                     "stratum_def": "specialist = >=1 out-of-home premise cluster "
                     "outside the global top-k most-cited clusters"}
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
