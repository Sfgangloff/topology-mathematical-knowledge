"""
q-0021 (Reviewer EvB8): Do non-mathematical modules (Init, Batteries, Std, Lean
core, lake packages = aesop/Qq/proofwidgets, External, Unknown, Deprecated)
CONTAMINATE the clusters / hubs / bridge diagram, and which headline results are
artifacts of dataset hygiene vs robust mathematical structure?

METHOD: re-run the CANONICAL pipeline UNCHANGED, twice, on the SAME
deterministically-sorted LCC, but in the HYGIENE arm delete every non-math node
BEFORE Leiden:

  build LCC  ->  (optionally drop non-math nodes)  ->  Leiden(seed=42, mod, 10it)
            ->  label_clusters  ->  find_bridge_theorems (raw)
            ->  depth_filter (elbow theta -> strict bridges)
            ->  neural_complexity (CN hub ranking)

Then compare BASELINE vs HYGIENE on every headline:
  * a-0001 cluster count + purity (#clusters, #pure, #mixed, mean dominant frac)
  * a-0010 SetTheory hub: CN rank of SetTheory (strict bridges), top-5 CN clusters
  * a-0007/a-0051 the 34-strict / 280-raw bridge sets: counts, Jaccard, recovery
  * does Batteries / a non-math module still appear as a main cluster/hub?

Determinism: nodes sorted before igraph conversion (matches the deterministic
node-order fix from bridge_detection_stability / e-0051). CPU only.

Run:  python3.12 -m src.hygiene_filter
"""
import json
import pickle
import statistics
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters
from bridge_theorems import find_bridge_theorems
from depth_filter import indegree_sweep, find_elbow, filter_bridges
from neural_complexity import compute_neural_complexity

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "hygiene_filter.json"

SEED = 42
N_ITER = 10
BIG = 10 ** 9

# Non-mathematical top-modules (per label_clusters._top_module taxonomy).
NONMATH = {"Init", "Batteries", "Std", "Lean", "lake", "External", "Unknown", "Mathlib"}


def top_module_of_node(G, node):
    fp = G.nodes[node].get("file_path", "")
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


def is_nonmath_node(G, node):
    # Deprecated lives inside Mathlib paths (e.g. .../Deprecated/...), treat as non-math.
    fp = G.nodes[node].get("file_path", "")
    if "Deprecated" in fp or "/Tactic/" in fp or fp.endswith("Tactic.lean"):
        return True
    return top_module_of_node(G, node) in NONMATH


def _nx_to_igraph_sorted(G_und):
    nodes = sorted(G_und.nodes())            # deterministic order (e-0051 fix)
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_und.edges()]
    return ig.Graph(n=len(nodes), edges=edges), nodes


def run_pipeline(G_full, G_und):
    """LCC of G_und -> Leiden -> labels -> raw bridges -> strict bridges -> CN.
    Returns a dict of all artifacts."""
    lcc = max(nx.connected_components(G_und), key=len)
    G_lcc = G_und.subgraph(lcc).copy()
    G_ig, nodes = _nx_to_igraph_sorted(G_lcc)
    part = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, n_iterations=N_ITER, seed=SEED)
    m = list(part.membership)
    cluster_map = {nodes[i]: m[i] for i in range(len(nodes))}

    labels = label_clusters(cluster_map, G_full)
    raw = find_bridge_theorems(G_full, cluster_map, labels, top_k=BIG)
    raw_set = {b["theorem"] for b in raw}

    thr, cnt, tbridges = indegree_sweep(G_full, cluster_map, labels)
    theta = find_elbow(thr, cnt) if tbridges else None
    strict = filter_bridges(tbridges, theta, labels) if theta is not None else []
    strict_set = {b["theorem"] for b in strict}

    # neural complexity needs cross_premises -> use strict-filtered bridges
    cn_metrics, global_cn, _W, _ids = compute_neural_complexity(strict, labels, cluster_map)

    return {
        "n_lcc": G_lcc.number_of_nodes(),
        "n_clusters": len(set(m)),
        "labels": labels,
        "cluster_map": cluster_map,
        "raw_set": raw_set,
        "strict_set": strict_set,
        "theta": theta,
        "cn_metrics": cn_metrics,
        "global_cn": global_cn,
    }


def purity_stats(labels):
    pure = [v for v in labels.values()
            if not v["is_mixed"] and v["dominant_fraction"] >= 0.50]
    mixed = [v for v in labels.values() if v["is_mixed"]]
    fracs = [v["dominant_fraction"] for v in labels.values()]
    return {
        "n_clusters": len(labels),
        "n_pure": len(pure),
        "n_mixed": len(mixed),
        "mean_dominant_fraction": round(statistics.mean(fracs), 4) if fracs else 0,
        "median_dominant_fraction": round(statistics.median(fracs), 4) if fracs else 0,
    }


def nonmath_cluster_presence(labels):
    """Clusters whose dominant module is non-mathematical (contamination as a hub)."""
    out = []
    for cid, info in sorted(labels.items(), key=lambda x: -x[1]["size"]):
        dm = info["dominant_module"]
        if dm in NONMATH:
            out.append({"cid": cid, "label": info["label"], "size": info["size"],
                        "dominant_module": dm, "dominant_fraction": info["dominant_fraction"]})
    return out


def cn_rank(cn_metrics, name_substr):
    ranked = sorted(cn_metrics.items(), key=lambda x: -x[1]["neural_complexity"])
    for r, (cid, m) in enumerate(ranked, 1):
        if name_substr.lower() in m["label"].lower():
            return {"rank": r, "cid": cid, "label": m["label"],
                    "cn": m["neural_complexity"], "n_total": len(ranked)}
    return {"rank": None, "n_total": len(ranked)}


def top_cn(cn_metrics, k=8):
    ranked = sorted(cn_metrics.items(), key=lambda x: -x[1]["neural_complexity"])
    return [{"rank": r, "label": m["label"], "cn": m["neural_complexity"],
             "is_mixed": m["is_mixed"]}
            for r, (cid, m) in enumerate(ranked[:k], 1)]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else 0.0


def main():
    print("Loading graph...")
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    G_und = G.to_undirected()
    print(f"Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Canonical stored bridge sets (the published objects)
    canon_raw = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems.json"))}
    canon_strict = {b["theorem"] for b in json.load(open(RESULTS_DIR / "bridge_theorems_filtered.json"))}

    # ---- BASELINE arm: full graph (reproduces canonical with deterministic order) ----
    print("\n=== BASELINE (all nodes) ===")
    base = run_pipeline(G, G_und)
    print(f"  LCC={base['n_lcc']:,}  clusters={base['n_clusters']}  theta={base['theta']}  "
          f"raw={len(base['raw_set'])}  strict={len(base['strict_set'])}")

    # ---- HYGIENE arm: drop non-math nodes BEFORE clustering ----
    print("\n=== HYGIENE (non-math nodes removed) ===")
    drop = [n for n in G_und.nodes() if is_nonmath_node(G, n)]
    print(f"  dropping {len(drop):,} non-math nodes")
    G_und_h = G_und.copy()
    G_und_h.remove_nodes_from(drop)
    # build_full for labelling on the SAME filtered node set so labels see only math
    G_h = G.copy()
    G_h.remove_nodes_from(drop)
    hyg = run_pipeline(G_h, G_und_h)
    print(f"  LCC={hyg['n_lcc']:,}  clusters={hyg['n_clusters']}  theta={hyg['theta']}  "
          f"raw={len(hyg['raw_set'])}  strict={len(hyg['strict_set'])}")

    # ---- Comparisons ----
    base_pur = purity_stats(base["labels"])
    hyg_pur = purity_stats(hyg["labels"])

    result = {
        "params": {"seed": SEED, "n_iter": N_ITER, "nonmath_modules": sorted(NONMATH),
                   "n_nonmath_dropped": len(drop)},
        "node_counts": {
            "full_graph": G.number_of_nodes(),
            "baseline_lcc": base["n_lcc"],
            "hygiene_lcc": hyg["n_lcc"],
        },
        # a-0001 / a-0002 : cluster count + purity
        "clustering": {
            "baseline": base_pur,
            "hygiene": hyg_pur,
            "baseline_nonmath_dominant_clusters": nonmath_cluster_presence(base["labels"]),
            "n_baseline_nonmath_dominant_clusters": len(nonmath_cluster_presence(base["labels"])),
            "hygiene_nonmath_dominant_clusters": nonmath_cluster_presence(hyg["labels"]),
        },
        # a-0010 : SetTheory hub via CN
        "hub": {
            "baseline_theta": base["theta"],
            "hygiene_theta": hyg["theta"],
            "baseline_settheory_cn_rank": cn_rank(base["cn_metrics"], "SetTheory"),
            "hygiene_settheory_cn_rank": cn_rank(hyg["cn_metrics"], "SetTheory"),
            "baseline_top_cn": top_cn(base["cn_metrics"]),
            "hygiene_top_cn": top_cn(hyg["cn_metrics"]),
        },
        # a-0007 / a-0051 : bridge sets
        "bridges": {
            "canonical_stored_raw": len(canon_raw),
            "canonical_stored_strict": len(canon_strict),
            "baseline_raw": len(base["raw_set"]),
            "baseline_strict": len(base["strict_set"]),
            "hygiene_raw": len(hyg["raw_set"]),
            "hygiene_strict": len(hyg["strict_set"]),
            "jaccard_raw_base_vs_hyg": jaccard(base["raw_set"], hyg["raw_set"]),
            "jaccard_strict_base_vs_hyg": jaccard(base["strict_set"], hyg["strict_set"]),
            "recovery_canon_strict_by_baseline": jaccard(base["strict_set"], canon_strict),
            "recovery_canon_strict_by_hygiene": round(
                len(hyg["strict_set"] & canon_strict) / len(canon_strict), 4) if canon_strict else 0,
            "recovery_canon_raw_by_hygiene": round(
                len(hyg["raw_set"] & canon_raw) / len(canon_raw), 4) if canon_raw else 0,
            "hygiene_strict_in_canon_strict": round(
                len(hyg["strict_set"] & canon_strict) / len(hyg["strict_set"]), 4) if hyg["strict_set"] else 0,
            "strict_overlap_examples": sorted(base["strict_set"] & hyg["strict_set"])[:15],
            "hygiene_only_strict_examples": sorted(hyg["strict_set"] - base["strict_set"])[:15],
        },
    }

    OUT_FILE.write_text(json.dumps(result, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "clustering_baseline": base_pur,
        "clustering_hygiene": hyg_pur,
        "n_baseline_nonmath_dominant_clusters": result["clustering"]["n_baseline_nonmath_dominant_clusters"],
        "settheory_rank_base": result["hub"]["baseline_settheory_cn_rank"],
        "settheory_rank_hyg": result["hub"]["hygiene_settheory_cn_rank"],
        "bridges": result["bridges"],
    }, indent=2))
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
