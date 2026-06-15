"""
Does bridge-aware retrieval pay off SPECIFICALLY on the project's own detected
operational-bridge theorems? (q-0011 x q-0006/q-0007 link, CPU-only)

The whole q-0011 retrieval chain (e-0011..e-0036) measures premise recall on
generic CROSS-DOMAIN test theorems (>=1 out-of-home premise) and NEVER on the
theorems the detection half (bridge_theorems.py, q-0006/q-0007) actually calls
"operational bridges": those whose proof draws premises from >=2 distinct PURE
clusters.  The two halves of the paper are disconnected.  This asks the coherence
question:

    Do the DETECTED operational-bridge theorems benefit MORE from bridge-aware
    retrieval (home -> home+top-k neighbour clusters) than ordinary cross-domain
    theorems do?

If yes, detection identifies exactly the theorems where bridge-aware retrieval
pays off (the pipeline is coherent).  If the lift is the same or smaller, bridge
DETECTION and bridge-aware RETRIEVAL are independent phenomena.

Leakage control identical to bridge_premise_retrieval.py: W estimated on TRAIN
only; home cluster + pure-cluster labels are graph-level (the same labels the
detection half uses); recall measured on held-out TEST theorems.

Definitions, all reusing the existing data:
  * home cluster      = data/clusters.json
  * premises          = <a>Name</a> premises across a theorem's tactic steps
                        (data/tactic_steps.jsonl), clustered nodes only
  * pure cluster      = cluster_labels: not is_mixed AND dominant_fraction>=0.50
                        (the exact bridge_theorems.py PURITY_THRESHOLD rule)
  * detected bridge   = test theorem whose premises span >=2 distinct PURE
                        clusters (bridge_theorems.py min_pure_clusters=2)
  * cross-domain      = test theorem with >=1 out-of-home premise

Groups (partition of test theorems with >=1 clustered premise):
  * detected_bridge       : >=2 distinct pure premise clusters
  * cross_nonbridge       : cross-domain but NOT a detected bridge
  * within_domain         : all premises in home cluster

Output: results/bridge_retrieval_on_detected.json
"""

import json
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
LABELS = DATA / "cluster_labels.json"
OUT = RESULTS / "bridge_retrieval_on_detected.json"

KS = [1, 2, 3, 5, 10]
PURITY_THRESHOLD = 0.50


def load():
    cluster = json.load(open(CLUSTERS))
    labels = {int(k): v for k, v in json.load(open(LABELS)).items()}
    premises = collections.defaultdict(set)
    split = {}
    for line in open(STEPS):
        d = json.loads(line)
        fn = d["full_name"]
        split[fn] = d["split"]
        for p in d.get("premises", []):
            if p != fn:
                premises[fn].add(p)
    return cluster, premises, split, labels


def build_bridge_graph(cluster, premises, split):
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


def finalize(a):
    out = {}
    for kind, v in a.items():
        out[kind] = round(v["recall_sum"] / v["n"], 4) if v["n"] else None
    return out


def evaluate(cluster, premises, split, W, pure_clusters):
    topk_cache = {c: [d for d, _ in W[c].most_common(max(KS))] for c in W}
    kinds = ["home"] + [f"bridge{k}" for k in KS]

    groups = ["detected_bridge", "cross_nonbridge", "within_domain"]
    agg = {g: {k: {"recall_sum": 0.0, "n": 0} for k in kinds} for g in groups}
    counts = collections.Counter()
    # number of distinct pure premise clusters distribution per group
    npure_sum = collections.Counter()

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
        in_home = sum(1 for d in prem_clusters if d == c)
        is_cross = in_home < total
        # distinct PURE clusters spanned by the premises (detection criterion)
        pure_spanned = {d for d in prem_clusters if d in pure_clusters}
        is_detected = len(pure_spanned) >= 2

        if is_detected:
            g = "detected_bridge"
        elif is_cross:
            g = "cross_nonbridge"
        else:
            g = "within_domain"
        counts[g] += 1
        npure_sum[g] += len(pure_spanned)

        nbrs = topk_cache.get(c, [])
        for kind, cand in (
            [("home", {c})] +
            [(f"bridge{k}", {c} | set(nbrs[:k])) for k in KS]
        ):
            rec = sum(1 for d in prem_clusters if d in cand) / total
            agg[g][kind]["recall_sum"] += rec
            agg[g][kind]["n"] += 1

    result = {"groups": {}}
    for g in groups:
        rec = finalize(agg[g])
        n = counts[g]
        lift = {}
        if rec["home"] is not None:
            for k in KS:
                bk = rec[f"bridge{k}"]
                lift[f"lift_top{k}"] = round(bk - rec["home"], 4) if bk is not None else None
        result["groups"][g] = {
            "n": n,
            "mean_distinct_pure_premise_clusters": round(npure_sum[g] / n, 3) if n else None,
            "mean_recall": rec,
            "lift_over_home": lift,
        }
    result["n_test_with_clustered_premises"] = sum(counts.values())
    return result


def main():
    cluster, premises, split, labels = load()
    pure_clusters = {
        cid for cid, info in labels.items()
        if not info["is_mixed"] and info["dominant_fraction"] >= PURITY_THRESHOLD
    }
    W = build_bridge_graph(cluster, premises, split)
    res = evaluate(cluster, premises, split, W, pure_clusters)
    res["params"] = {
        "ks": KS,
        "purity_threshold": PURITY_THRESHOLD,
        "n_pure_clusters": len(pure_clusters),
        "n_clusters": len(set(cluster.values())),
    }
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
