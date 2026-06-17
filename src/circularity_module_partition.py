"""
q-0020: Is the bridge definition circular? Bridges are theorems crossing CLUSTER
boundaries, but clusters come from the SAME dependency graph, so the method may
partly recover its own input.

TEST: Re-run the SAME depth-filtered bridge detector, but replace Leiden clusters
with an EXTERNAL partition that does NOT come from the dependency graph at all:
the top-level Mathlib MODULE label of each node (Topology, Algebra, Analysis, ...).
Module labels come from file paths / library organisation, independent of the
edge structure Leiden optimised over.

If the module-partition bridges substantially overlap the Leiden-partition bridges,
the detected bridges are NOT an artifact of the clustering choice (the cross-domain
signal survives swapping the partition for an independent one). If they barely
overlap, the bridge set is partition-specific (circularity concern stands).

Mirror depth_filter.indegree_sweep exactly but with module-based "pure clusters".
"""
import json, pickle
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
THETA = 2  # reference depth threshold


def top_module(module: str) -> str:
    parts = module.split(".")
    if "Mathlib" in parts:
        i = parts.index("Mathlib")
        return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    for key in ("Batteries", "Init", "Std", "Lean"):
        if key in module:
            return key
    return "Unknown"


def detect_bridges(G, part_map, in_deg, valid_parts, theta):
    """Theorems whose proof cites depth<=theta premises from >=2 distinct
    'pure' partition cells (different from the theorem's own cell)."""
    bridges = []
    for node in G.nodes():
        self_p = part_map.get(node)
        if self_p is None:
            continue
        cross = [(p, part_map[p], in_deg.get(p, 0))
                 for p in G.successors(node)
                 if p in part_map and part_map[p] in valid_parts
                 and part_map[p] != self_p]
        if not cross:
            continue
        reached = {c for _, c, _ in cross}
        if len(reached) < 2:
            continue
        if min(d for _, _, d in cross) <= theta:
            bridges.append(node)
    return set(bridges)


def main():
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    leiden_map = json.load(open(DATA / "clusters.json"))
    labels = json.load(open(DATA / "cluster_labels.json"))
    leiden_bridges = {b["theorem"] for b in
                      json.load(open(RES / "bridge_theorems_filtered.json"))}

    in_deg = dict(G.in_degree())

    # --- Leiden pure clusters (reference) ---
    leiden_pure = {cid for cid, info in labels.items()
                   if not info["is_mixed"] and info["dominant_fraction"] >= 0.50}

    # --- Module partition (external, library organisation) ---
    mod_map = {n: top_module(G.nodes[n].get("module", "")) for n in G.nodes()}
    mod_sizes = Counter(mod_map.values())
    NONMATH = {"Unknown", "Init", "Batteries", "Std", "Lean", ""}
    # "pure" module cells = real math modules of meaningful size
    mod_pure = {m for m, c in mod_sizes.items()
                if m not in NONMATH and c >= 50}

    # Recompute Leiden bridges with the SAME function (sanity check it matches file)
    leiden_recomputed = detect_bridges(G, leiden_map, in_deg, leiden_pure, THETA)

    # Module-partition bridges
    mod_bridges = detect_bridges(G, mod_map, in_deg, mod_pure, THETA)

    inter = leiden_bridges & mod_bridges
    union = leiden_bridges | mod_bridges
    out = {
        "theta": THETA,
        "n_module_cells_total": len(mod_sizes),
        "n_module_pure_cells": len(mod_pure),
        "module_pure_cells": sorted(mod_pure),
        "n_leiden_bridges": len(leiden_bridges),
        "n_leiden_recomputed": len(leiden_recomputed),
        "leiden_recompute_matches_file": leiden_bridges == leiden_recomputed,
        "n_module_bridges": len(mod_bridges),
        "n_overlap": len(inter),
        "jaccard": round(len(inter) / len(union), 3) if union else 0,
        "recall_of_leiden_by_module": round(len(inter) / len(leiden_bridges), 3),
        "overlap_theorems": sorted(inter)[:40],
        "module_only_examples": sorted(mod_bridges - leiden_bridges)[:20],
        "leiden_only_examples": sorted(leiden_bridges - mod_bridges)[:20],
    }
    json.dump(out, open(RES / "circularity_module_partition.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("overlap_theorems", "module_only_examples",
                                   "leiden_only_examples", "module_pure_cells")},
                     indent=2))
    print("overlap sample:", out["overlap_theorems"][:10])


if __name__ == "__main__":
    main()
