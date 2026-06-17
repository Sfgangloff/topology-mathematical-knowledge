"""
q-0025 + q-0029: The loose-vs-strict bridge operating point.

The headline detector fixes theta=2 (count elbow), recovering 2/8 historical
bridges strict vs 6/8 loose. Reviewers ask: what precision/recall operating point
does theta trade, does any wider theta recover a 3rd celebrated bridge, and should
the headline detector be loose or strict?

We sweep theta in {1,2,3,4,5,7,10,15,20, inf(=loose)} and report at each:
  * n_bridges (strict set size; theta=inf = the full loose set)
  * historical RECALL: how many of the 8 known bridges are detected (strict)
  * a NULL-BASED PRECISION PROXY. There is no full ground-truth bridge set, so
    precision is proxied by structure: shuffle the Leiden cluster labels across
    nodes (size-preserving, n=10), re-run the SAME depth filter, and measure what
    fraction of the detected set a random relabelling also flags
    (null_recall). signal_precision_proxy = 1 - null_recall = fraction of the
    detected set that is NOT explained by a structureless null.
    (Mirrors the permutation-null logic of a-0055/a-0057.)

Reusing the existing pipeline (cluster cache, depth_filter, validate_bridges).
"""
import json, pickle, random, statistics
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
sys.path.insert(0, str(Path(__file__).parent))
from cluster import load as load_clusters
from label_clusters import load as load_labels
from depth_filter import indegree_sweep
from validate_bridges import KNOWN_BRIDGES, search_theorems

THETAS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
N_SHUFFLES = 10
SEED = 42


def surviving_set(theorem_bridges, theta):
    return {b["theorem"] for b in theorem_bridges if b["min_cross_indegree"] <= theta}


def historical_recall(all_theorems, detected_set, loose_set):
    out = []
    for bridge in KNOWN_BRIDGES:
        matches = []
        for _side, patterns in bridge["sides"]:
            matches += search_theorems(all_theorems, patterns)
        out.append({"name": bridge["name"],
                    "strict": any(t in detected_set for t in matches),
                    "loose": any(t in loose_set for t in matches)})
    return out


def main():
    random.seed(SEED)
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    cluster_map = load_clusters()
    labels = load_labels()

    _t, _c, theorem_bridges = indegree_sweep(G, cluster_map, labels)
    all_bt_set = {b["theorem"] for b in json.load(open(RES / "bridge_theorems.json"))}
    all_theorems = list(cluster_map.keys())

    # ---- null: shuffled-label strict sets for the precision proxy ----
    # We need, per theta, the null detected sets. Recomputing indegree_sweep under
    # shuffled labels is the principled move (the filter depends on cluster labels).
    # indegree_sweep takes (G, cluster_map, labels); we permute cluster_map values.
    pure_nodes = [n for n in cluster_map]
    vals = [cluster_map[n] for n in pure_nodes]

    null_sets_by_theta = {t: [] for t in THETAS}
    for s in range(N_SHUFFLES):
        shuffled = vals[:]
        random.shuffle(shuffled)
        cm_s = dict(zip(pure_nodes, shuffled))
        _t2, _c2, tb_s = indegree_sweep(G, cm_s, labels)
        for t in THETAS:
            null_sets_by_theta[t].append(surviving_set(tb_s, t))
        print(f"  null shuffle {s+1}/{N_SHUFFLES} done")

    rows = []
    for theta in THETAS:
        surv = surviving_set(theorem_bridges, theta)
        hist = historical_recall(all_theorems, surv, all_bt_set)
        n_strict = sum(1 for h in hist if h["strict"])
        # precision proxy from null
        null_recalls = []
        for ns in null_sets_by_theta[theta]:
            null_recalls.append(len(ns & surv) / len(surv) if surv else 0)
        nr_mean = statistics.mean(null_recalls)
        rows.append({
            "theta": theta,
            "n_bridges": len(surv),
            "historical_recall_strict": n_strict,
            "historical_recall_frac": round(n_strict / 8, 3),
            "null_recall_mean": round(nr_mean, 4),
            "signal_precision_proxy": round(1 - nr_mean, 4),
            "null_set_size_mean": round(statistics.mean(len(ns) for ns in null_sets_by_theta[theta]), 1),
            "recovered_bridges": [h["name"] for h in hist if h["strict"]],
        })

    # loose row (theta=inf)
    loose_hist = historical_recall(all_theorems, all_bt_set, all_bt_set)
    rows.append({
        "theta": "inf_loose",
        "n_bridges": len(all_bt_set),
        "historical_recall_strict": sum(1 for h in loose_hist if h["loose"]),
        "historical_recall_frac": round(sum(1 for h in loose_hist if h["loose"]) / 8, 3),
        "null_recall_mean": None,
        "signal_precision_proxy": None,
        "null_set_size_mean": None,
        "recovered_bridges": [h["name"] for h in loose_hist if h["loose"]],
    })

    out = {
        "thetas": THETAS,
        "n_shuffles": N_SHUFFLES,
        "n_raw_candidates": len(theorem_bridges),
        "loose_set_size": len(all_bt_set),
        "operating_points": rows,
        "note": "signal_precision_proxy = 1 - (fraction of strict set a label-permutation null also flags); recall is fraction of 8 historical bridges strict-detected (loose for theta=inf).",
    }
    json.dump(out, open(RES / "operating_point_curve.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
