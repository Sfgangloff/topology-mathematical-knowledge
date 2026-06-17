"""
q-0030: Is there a CLUSTERING-INDEPENDENT operational definition of a cross-domain
bridge whose detected SET is (a) stable across independent partitions and
(b) beats a label-permutation null?

q-0020 (a-0053) showed the Leiden bridge SET is a partition artifact: the
specific 34 are not recovered by an independent partition (Jaccard ~0.005) and a
permutation null recovers ~98% of them. The natural successor: drop clustering
entirely and define bridges via the HUMAN library partition (top-level Mathlib
module = directory label, no community detection at all). Two falsification tests:

  TEST A  STABILITY ACROSS INDEPENDENT PARTITIONS.
    Define bridges with the depth-filtered detector under three INDEPENDENT
    human partitions taken from file paths (no dependency-graph optimisation):
      P1 = top-level module      (Mathlib.<X>....)         e.g. Topology
      P2 = depth-2 submodule     (Mathlib.<X>.<Y>...)      e.g. Topology.Algebra
      P3 = depth-3 submodule     (Mathlib.<X>.<Y>.<Z>...)
    A clustering-independent definition is "stable" only if the detected SET is
    reproducible across these partitions (Jaccard of bridge sets). Compare the
    pairwise Jaccards to the Leiden-vs-module Jaccard (~0.005, a-0053).

  TEST B  LABEL-PERMUTATION NULL.
    For the top-level-module partition, shuffle module labels across nodes
    (preserving cell sizes; n=20 shuffles) and re-detect bridges. Report observed
    count, null mean/std, z-score, and -- crucially -- the SET-level recall of the
    observed bridges by the null (what fraction of observed bridges a random
    relabelling also flags). If the null reproduces the observed SET (high recall),
    the clustering-independent definition is ALSO a partition artifact.

Verdict feeds q-0030: does any clustering-independent bridge set survive, or must
detection move to proof-structure (conceptual) bridges (q-0009)?
"""
import json, pickle, random
from collections import Counter
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
THETA = 2
N_SHUFFLES = 20
SEED = 42


def submodule(module: str, depth: int) -> str | None:
    """Mathlib.<X>.<Y>... -> join of the first `depth` segments after 'Mathlib'.
    Returns None for non-Mathlib (library) nodes so they are excluded as cells."""
    parts = module.split(".")
    if "Mathlib" not in parts:
        return None
    i = parts.index("Mathlib")
    seg = parts[i + 1: i + 1 + depth]
    if not seg:
        return None
    return ".".join(seg)


def detect_bridges(G, part_map, in_deg, valid_parts, theta):
    bridges = []
    for node in G.nodes():
        self_p = part_map.get(node)
        if self_p is None:
            continue
        cross = [(part_map[p], in_deg.get(p, 0))
                 for p in G.successors(node)
                 if p in part_map and part_map[p] in valid_parts
                 and part_map[p] != self_p]
        if len(cross) < 1:
            continue
        reached = {c for c, _ in cross}
        if len(reached) < 2:
            continue
        if min(d for _, d in cross) <= theta:
            bridges.append(node)
    return set(bridges)


def pure_cells(part_map, min_size=50):
    sizes = Counter(v for v in part_map.values() if v is not None)
    return {c for c, n in sizes.items() if n >= min_size}, sizes


def jacc(a, b):
    u = a | b
    return round(len(a & b) / len(u), 4) if u else 0.0


def main():
    random.seed(SEED)
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    in_deg = dict(G.in_degree())
    node_mod = {n: G.nodes[n].get("module", "") for n in G.nodes()}

    # ---- three independent human partitions from file paths ----
    parts = {}
    for depth, name in [(1, "P1_toplevel"), (2, "P2_depth2"), (3, "P3_depth3")]:
        pm = {n: submodule(node_mod[n], depth) for n in G.nodes()}
        valid, sizes = pure_cells(pm)
        bset = detect_bridges(G, pm, in_deg, valid, THETA)
        parts[name] = {"map": pm, "valid": valid, "bridges": bset,
                       "n_cells": len(valid), "n_bridges": len(bset)}

    # ---- TEST A: stability across the three independent partitions ----
    names = ["P1_toplevel", "P2_depth2", "P3_depth3"]
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a}__vs__{b}"] = {
                "jaccard": jacc(parts[a]["bridges"], parts[b]["bridges"]),
                "overlap": len(parts[a]["bridges"] & parts[b]["bridges"]),
                "n_a": parts[a]["n_bridges"], "n_b": parts[b]["n_bridges"],
                # recall of the COARSER partition's bridges by the finer one
                "recall_coarser_by_finer": round(
                    len(parts[a]["bridges"] & parts[b]["bridges"]) /
                    len(parts[a]["bridges"]), 4) if parts[a]["bridges"] else 0,
            }

    # ---- TEST B: label-permutation null on P1 (top-level module) ----
    pm = parts["P1_toplevel"]["map"]
    valid = parts["P1_toplevel"]["valid"]
    obs = parts["P1_toplevel"]["bridges"]
    # nodes that have a (non-None) label, in fixed order; shuffle labels among them
    labelled = [n for n in G.nodes() if pm[n] is not None]
    labels_list = [pm[n] for n in labelled]
    null_counts = []
    null_recalls = []  # fraction of observed bridges also flagged under shuffle
    for s in range(N_SHUFFLES):
        shuffled = labels_list[:]
        random.shuffle(shuffled)
        pm_s = dict(pm)
        for n, lab in zip(labelled, shuffled):
            pm_s[n] = lab
        # valid cells unchanged in size distribution (same multiset of labels)
        bset_s = detect_bridges(G, pm_s, in_deg, valid, THETA)
        null_counts.append(len(bset_s))
        null_recalls.append(len(bset_s & obs) / len(obs) if obs else 0)

    import statistics
    null_mean = statistics.mean(null_counts)
    null_std = statistics.pstdev(null_counts)
    z = (len(obs) - null_mean) / null_std if null_std > 0 else float("nan")

    out = {
        "theta": THETA,
        "min_cell_size": 50,
        "partitions": {n: {"n_cells": parts[n]["n_cells"],
                           "n_bridges": parts[n]["n_bridges"]} for n in names},
        "TEST_A_stability_across_independent_partitions": pairwise,
        "leiden_vs_module_jaccard_reference_a0053": 0.005,
        "TEST_B_label_permutation_null": {
            "observed_bridges": len(obs),
            "n_shuffles": N_SHUFFLES,
            "null_mean_count": round(null_mean, 1),
            "null_std_count": round(null_std, 1),
            "count_z_score": round(z, 2),
            "null_recall_of_observed_mean": round(statistics.mean(null_recalls), 4),
            "null_recall_of_observed_min": round(min(null_recalls), 4),
            "null_recall_of_observed_max": round(max(null_recalls), 4),
        },
    }
    json.dump(out, open(RES / "clustering_independent_bridge.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
