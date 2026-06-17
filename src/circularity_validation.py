"""
q-0020 (Reviewer JAGf): Is the bridge definition CIRCULAR? Bridges are theorems whose
premises cross CLUSTER boundaries, but the clusters come from Leiden on the SAME
dependency graph, so the detector may partly rediscover boundaries it invented.

We validate bridges INDEPENDENTLY of the Leiden clustering with three tests, all
re-using the canonical depth-filtered cross-cell-premise detector (theta=2 elbow):

  (A) EXTERNAL ground-truth partition. Replace Leiden cells with the top-level
      Mathlib MODULE of each node (Topology, Algebra, ... = library organisation,
      NOT optimised over the edge structure). Re-detect bridges. If the external
      partition recovers the Leiden bridges, the bridges are NOT a clustering
      artifact. Granularity differs (138 Leiden clusters vs 33 module cells) so we
      report recall/precision of the Leiden strict-bridge set, not raw counts.

  (C) PERMUTATION NULL. Random partition with the SAME cell-size profile as the
      module partition (labels shuffled across nodes). A theorem is a "null bridge"
      if its depth<=theta premises span >=2 random cells. This isolates "real
      cross-DOMAIN premise" (A) from "premise crosses ANY boundary" (the boundary
      the partition invents). 20 shuffles; report mean null recall of the Leiden set
      and mean null bridge count.

  (B) HELD-OUT split. Randomly split LCC nodes 50/50. Cluster (Leiden, seed=42) the
      TRAIN half ONLY, propagate labels, and detect bridges among nodes whose
      premises are scored against the TRAIN-derived cells. Validate against the
      canonical full-graph Leiden bridges restricted to the held-out (TEST) nodes.
      If bridges survive being defined on a clustering that never saw them, the
      detector is not merely re-reading its own input partition.

Agreement metric throughout: Jaccard and recall of the canonical Leiden STRICT
bridge set (results/bridge_theorems_filtered.json). CPU only, no model inference.

Run:  python3.12 -m src.circularity_validation
"""
import json
import pickle
import random
import statistics
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters
from bridge_theorems import find_bridge_theorems
from depth_filter import indegree_sweep, find_elbow, filter_bridges

DATA = Path(__file__).parent.parent / "data"
RES = Path(__file__).parent.parent / "results"
OUT = RES / "circularity_validation.json"

THETA = 2
SEED = 42
N_NULL = 20
NONMATH = {"Init", "Batteries", "Std", "Lean", "lake", "External", "Unknown", "Mathlib"}


def top_module(G, node):
    fp = G.nodes[node].get("file_path", "") if G.has_node(node) else ""
    if not fp:
        return "Unknown"
    parts = Path(fp).with_suffix("").parts
    if not parts:
        return "Unknown"
    for i, s in enumerate(parts):
        if s == "Mathlib":
            return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    for p in parts:
        if p in ("Init", "Batteries", "Std", "lake"):
            return p
    return "External"


def detect_strict_bridges(G, part_map, in_deg, valid_cells, theta):
    """Theorems whose proof cites depth<=theta premises from >=2 distinct VALID
    cells, each different from the theorem's own cell. Returns the theorem set."""
    out = set()
    for node in G.nodes():
        self_c = part_map.get(node)
        if self_c is None:
            continue
        cross = [(p, part_map[p], in_deg.get(p, 0))
                 for p in G.successors(node)
                 if p in part_map and part_map[p] in valid_cells
                 and part_map[p] != self_c]
        if not cross:
            continue
        if len({c for _, c, _ in cross}) < 2:
            continue
        if min(d for _, _, d in cross) <= theta:
            out.add(node)
    return out


def jaccard(a, b):
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else 1.0 if not a and not b else 0.0


def recall(reference, found):
    return round(len(reference & found) / len(reference), 4) if reference else 0.0


def precision(reference, found):
    return round(len(reference & found) / len(found), 4) if found else 0.0


def _nx_to_igraph_sorted(G_und):
    nodes = sorted(G_und.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G_und.edges()]
    return ig.Graph(n=len(nodes), edges=edges), nodes


def main():
    print("Loading...")
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    leiden_map = {k: int(v) for k, v in json.load(open(DATA / "clusters.json")).items()}
    leiden_bridges = {b["theorem"] for b in
                      json.load(open(RES / "bridge_theorems_filtered.json"))}
    in_deg = dict(G.in_degree())
    lcc_nodes = set(leiden_map)
    print(f"LCC nodes={len(lcc_nodes):,}  canonical Leiden strict bridges={len(leiden_bridges)}")

    # Leiden pure cells (int cids) -- canonical pure-cluster universe
    labels_int = label_clusters(leiden_map, G)
    leiden_pure = {cid for cid, info in labels_int.items()
                   if not info["is_mixed"] and info["dominant_fraction"] >= 0.50}
    leiden_recomp = detect_strict_bridges(G, leiden_map, in_deg, leiden_pure, THETA)
    print(f"Leiden recompute strict bridges={len(leiden_recomp)} "
          f"(matches file: recall={recall(leiden_bridges, leiden_recomp)})")

    # ---- (A) EXTERNAL module partition (ground-truth library labels) ----
    mod_map = {n: top_module(G, n) for n in G.nodes()}
    mod_sizes = Counter(mod_map[n] for n in lcc_nodes)
    mod_pure = {m for m, c in mod_sizes.items() if m not in NONMATH and c >= 50}
    mod_bridges = detect_strict_bridges(G, mod_map, in_deg, mod_pure, THETA)
    A = {
        "n_module_cells_over_lcc": len(mod_sizes),
        "n_module_pure_cells": len(mod_pure),
        "n_leiden_clusters": len(set(leiden_map.values())),
        "n_module_bridges": len(mod_bridges),
        "n_leiden_bridges": len(leiden_bridges),
        "overlap": len(leiden_bridges & mod_bridges),
        "jaccard": jaccard(leiden_bridges, mod_bridges),
        "recall_of_leiden_by_module": recall(leiden_bridges, mod_bridges),
        "precision_module_in_leiden": precision(leiden_bridges, mod_bridges),
        "overlap_examples": sorted(leiden_bridges & mod_bridges)[:20],
        "leiden_only_examples": sorted(leiden_bridges - mod_bridges)[:15],
    }
    print(f"(A) module bridges={len(mod_bridges)}  recall_of_Leiden={A['recall_of_leiden_by_module']}  "
          f"precision={A['precision_module_in_leiden']}")

    # ---- (C) PERMUTATION NULL matched to module cell-size profile ----
    # Shuffle the module labels across LCC nodes (destroys real domain<->node link,
    # keeps cell-count and size profile). A real cross-domain premise should be
    # recovered FAR less by random cells than by the true module cells.
    rng = random.Random(SEED)
    lcc_list = list(lcc_nodes)
    mod_labels_seq = [mod_map[n] for n in lcc_list]
    null_recalls, null_counts, null_jacc = [], [], []
    for t in range(N_NULL):
        shuffled = mod_labels_seq[:]
        rng.shuffle(shuffled)
        null_map = dict(zip(lcc_list, shuffled))
        null_sizes = Counter(null_map.values())
        null_pure = {m for m, c in null_sizes.items() if m not in NONMATH and c >= 50}
        nb = detect_strict_bridges(G, null_map, in_deg, null_pure, THETA)
        null_recalls.append(recall(leiden_bridges, nb))
        null_counts.append(len(nb))
        null_jacc.append(jaccard(leiden_bridges, nb))
    # Leiden-granularity null: shuffle the Leiden cell labels (138 cells, same size
    # profile as the real Leiden partition) -- the granularity-matched control for
    # the circularity question (does a RANDOM partition of the SAME granularity as
    # Leiden recover the bridges as well as the real Leiden / module partition?).
    leiden_labels_seq = [leiden_map[n] for n in lcc_list]
    lnull_recalls, lnull_counts = [], []
    for t in range(N_NULL):
        sh = leiden_labels_seq[:]
        rng.shuffle(sh)
        lnull_map = dict(zip(lcc_list, sh))
        lnull_sizes = Counter(lnull_map.values())
        # 'pure' here = any cell of size>=50 (random cells have no dominant module)
        lnull_pure = {c for c, k in lnull_sizes.items() if k >= 50}
        nb = detect_strict_bridges(G, lnull_map, in_deg, lnull_pure, THETA)
        lnull_recalls.append(recall(leiden_bridges, nb))
        lnull_counts.append(len(nb))

    C = {
        "n_shuffles": N_NULL,
        "leiden_granularity_null_recall_mean": round(statistics.mean(lnull_recalls), 4),
        "leiden_granularity_null_count_mean": round(statistics.mean(lnull_counts), 1),
        "null_recall_of_leiden_mean": round(statistics.mean(null_recalls), 4),
        "null_recall_of_leiden_max": round(max(null_recalls), 4),
        "null_bridge_count_mean": round(statistics.mean(null_counts), 1),
        "null_jaccard_mean": round(statistics.mean(null_jacc), 4),
        "module_recall_over_null_recall": round(
            A["recall_of_leiden_by_module"] / statistics.mean(null_recalls), 2)
            if statistics.mean(null_recalls) > 0 else None,
    }
    print(f"(C) null recall mean={C['null_recall_of_leiden_mean']} "
          f"(module {A['recall_of_leiden_by_module']} -> {C['module_recall_over_null_recall']}x null); "
          f"null count mean={C['null_bridge_count_mean']}")

    # ---- (B) HELD-OUT split: cluster TRAIN half, detect bridges on TEST nodes ----
    rng2 = random.Random(SEED)
    shuffled_nodes = lcc_list[:]
    rng2.shuffle(shuffled_nodes)
    half = len(shuffled_nodes) // 2
    train = set(shuffled_nodes[:half])
    test = set(shuffled_nodes[half:])
    G_und = G.to_undirected()
    G_train = G_und.subgraph(train).copy()
    tlcc = max(nx.connected_components(G_train), key=len)
    G_train_lcc = G_train.subgraph(tlcc).copy()
    G_ig, nodes = _nx_to_igraph_sorted(G_train_lcc)
    part = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, n_iterations=10, seed=SEED)
    train_map = {nodes[i]: part.membership[i] for i in range(len(nodes))}
    train_labels = label_clusters(train_map, G)
    train_pure = {cid for cid, info in train_labels.items()
                  if not info["is_mixed"] and info["dominant_fraction"] >= 0.50}
    # Propagate train cell labels to TEST nodes by majority vote of their TRAIN
    # neighbours, so a TEST theorem has a 'self cell' it can be a bridge OUT of,
    # WITHOUT the test node ever influencing the partition. Nodes with no train
    # neighbour stay unassigned (skipped, as in the canonical detector).
    full_map = dict(train_map)
    for n in test:
        nbr_cells = Counter(train_map[m] for m in G_und.neighbors(n)
                            if m in train_map) if G_und.has_node(n) else Counter()
        if nbr_cells:
            full_map[n] = nbr_cells.most_common(1)[0][0]
    # Detect bridges scoring premises against TRAIN-derived pure cells; restrict to
    # TEST nodes (never used to build the partition).
    heldout_bridges_all = detect_strict_bridges(G, full_map, in_deg, train_pure, THETA)
    heldout_bridges_test = heldout_bridges_all & test
    # Reference = canonical Leiden bridges that land in the TEST half.
    leiden_bridges_test = leiden_bridges & test
    B = {
        "train_lcc_nodes": G_train_lcc.number_of_nodes(),
        "train_clusters": len(set(train_map.values())),
        "train_pure_cells": len(train_pure),
        "n_leiden_bridges_in_test": len(leiden_bridges_test),
        "n_heldout_bridges_in_test": len(heldout_bridges_test),
        "overlap_test": len(leiden_bridges_test & heldout_bridges_test),
        "recall_of_leiden_test_by_heldout": recall(leiden_bridges_test, heldout_bridges_test),
        "jaccard_test": jaccard(leiden_bridges_test, heldout_bridges_test),
        "examples": sorted(leiden_bridges_test & heldout_bridges_test)[:15],
    }
    print(f"(B) held-out: leiden-bridges-in-test={len(leiden_bridges_test)} "
          f"heldout-detected-in-test={len(heldout_bridges_test)} "
          f"recall={B['recall_of_leiden_test_by_heldout']}")

    out = {"theta": THETA, "leiden_recompute_recall": recall(leiden_bridges, leiden_recomp),
           "A_external_module_partition": A,
           "C_permutation_null": C,
           "B_heldout_split": B}
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({"A_recall": A["recall_of_leiden_by_module"],
                      "A_precision": A["precision_module_in_leiden"],
                      "A_jaccard": A["jaccard"],
                      "C_null_recall_mean": C["null_recall_of_leiden_mean"],
                      "C_signal_over_null": C["module_recall_over_null_recall"],
                      "B_heldout_recall": B["recall_of_leiden_test_by_heldout"]}, indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
