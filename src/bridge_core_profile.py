"""
Materialize and CHARACTERIZE the seed-stable core of operational bridge theorems.

e-0050 (bridge_detection_stability, a-0050) showed the detected operational-bridge
list is NOT seed-stable in identity (raw Jaccard 0.31, strict 0.10), but found a small
reproducible CORE of 79 raw bridges flagged in ALL 7 seeds {0,1,7,42,123,2024,31337},
and called it "high-confidence ... worth treating as such." But e-0050 only COUNTED the
core (79) -- it never saved the list or asked WHAT these theorems are. This experiment
closes that gap. It is the natural follow-up: turn "there is a core of 79" into actual
content.

It re-runs the CANONICAL detection pipeline on each of the same 7 seeds (reusing the
exact detect code from bridge_detection_stability), intersects the raw bridge sets to
recover the 79-theorem core, and then characterizes it along four axes, all CPU-only
over existing data:

  1. Overlap with the canonical stored sets (280 raw / 34 strict): is the core a
     subset of the canonical 280, and how many strict-34 survive into the core?
  2. Structural profile (reuses e-0048 features: proof_len, n_premises, reuse_indeg,
     dag_depth): is the seed-stable core EVEN MORE of a deep-synthesis object than the
     canonical 280, or is it just the "easy" shallow detections that every seed agrees on?
  3. Domain spread: top-level Mathlib module (partition-INDEPENDENT, from file_path) of
     each core theorem -- which domains host the reproducible bridges.
  4. Historical-bridge overlap: do any of the 79 appear in the 8 known historical
     bridges (validation.json)?

Run:  python3.12 -m src.bridge_core_profile
"""

import json
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_clusters import label_clusters
from bridge_theorems import find_bridge_theorems

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS / "bridge_core_profile.json"

SEEDS = [0, 1, 7, 42, 123, 2024, 31337]
N_ITERATIONS = 10
BIG = 10 ** 9


def _nx_to_igraph(G_nx):
    # IMPORTANT: sort nodes for a DETERMINISTIC igraph node order. The LCC is a
    # Python set, so list(G.nodes()) order depends on PYTHONHASHSEED and varies
    # across processes. Leiden's own seed= does NOT fix this -- the partition is
    # driven by node-insertion order, so two runs with the same seed but different
    # hash orderings give different partitions (verified: 141 vs 143 clusters at
    # seed=0). e-0050 ran in one process with one incidental hash order, so its
    # per-seed counts and "79-core" are NOT reproducible. Sorting makes the Leiden
    # seed the ONLY source of variation, so this core is reproducible.
    nodes = sorted(G_nx.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in G_nx.edges()]
    return ig.Graph(n=len(nodes), edges=edges), nodes


def detect_raw(G, cluster_map):
    labels = label_clusters(cluster_map, G)
    raw = find_bridge_theorems(G, cluster_map, labels, top_k=BIG)
    return {b["theorem"] for b in raw}


# ---- structural features (mirror bridge_structural_profile.py) -------------
def load_features():
    tactics = json.load(open(DATA / "theorem_tactics.json"))
    proof_len = {name: len(steps) for name, steps in tactics.items()}
    out_prem = defaultdict(set)
    module = {}
    with open(DATA / "tactic_steps.jsonl") as f:
        for line in f:
            o = json.loads(line)
            name = o["full_name"]
            fp = o.get("file_path", "")
            if name not in module and fp.startswith("Mathlib/"):
                parts = fp.split("/")
                module[name] = parts[1] if len(parts) > 1 else "?"
            for p in o["premises"]:
                if p != name:
                    out_prem[name].add(p)
    reuse_indeg = defaultdict(int)
    for name, prems in out_prem.items():
        for p in prems:
            reuse_indeg[p] += 1
    return proof_len, out_prem, reuse_indeg, module


def dag_depth(out_prem):
    depth, state = {}, {}
    GRAY = 1
    for root in list(out_prem.keys()):
        if root in depth:
            continue
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                d = 0
                for p in out_prem.get(node, ()):
                    d = max(d, depth.get(p, 0) + 1)
                depth[node] = d
                state[node] = 0
                continue
            if node in depth:
                continue
            if state.get(node) == GRAY:
                depth[node] = 0
                continue
            state[node] = GRAY
            stack.append((node, True))
            for p in out_prem.get(node, ()):
                if p not in depth and state.get(p) != GRAY:
                    stack.append((p, False))
    return depth


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def rank_fraction(group_vals, pop_sorted):
    import bisect
    n = len(pop_sorted)
    if n == 0 or not group_vals:
        return 0.0
    tot = 0.0
    for v in group_vals:
        lo = bisect.bisect_left(pop_sorted, v)
        hi = bisect.bisect_right(pop_sorted, v)
        tot += (lo + hi) / 2 / n
    return tot / len(group_vals)


def feature_block(getter, all_names_set, pop_names, group):
    pop_sorted = sorted(getter(t) for t in pop_names)
    gvals = [getter(t) for t in group if t in all_names_set]
    return {
        "pop_median": median(pop_sorted),
        "pop_mean": round(sum(pop_sorted) / len(pop_sorted), 3),
        "core_median": median(gvals),
        "core_mean": round(sum(gvals) / len(gvals), 3) if gvals else None,
        "core_rankfrac": round(rank_fraction(gvals, pop_sorted), 3),
        "n_scored": len(gvals),
    }


def historical_names():
    """All theorem names mentioned anywhere in the 8 historical bridges."""
    val = json.load(open(RESULTS / "validation.json"))
    names = set()
    by_bridge = {}
    for entry in val:
        bnames = set()
        for side in entry.get("sides", []):
            for cl in side.get("clusters", {}).values():
                for t in cl.get("theorems", []):
                    bnames.add(t)
            for t in side.get("sample_theorems", []):
                bnames.add(t)
        for t in entry.get("detected_by_any_bridge", []) or []:
            bnames.add(t)
        for t in entry.get("detected_by_depth_filter", []) or []:
            bnames.add(t)
        by_bridge[entry["name"]] = bnames
        names |= bnames
    return names, by_bridge


def main():
    print("Loading graph…")
    with open(DATA / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    G_und = G.to_undirected()
    lcc = max(nx.connected_components(G_und), key=len)
    G_lcc = G_und.subgraph(lcc).copy()
    G_ig, nodes = _nx_to_igraph(G_lcc)
    print(f"LCC: {G_lcc.number_of_nodes():,} nodes")

    raw_sets = {}
    for s in SEEDS:
        part = leidenalg.find_partition(
            G_ig, leidenalg.ModularityVertexPartition, n_iterations=N_ITERATIONS, seed=s
        )
        m = list(part.membership)
        cluster_map = {nodes[i]: m[i] for i in range(len(nodes))}
        raw_sets[s] = detect_raw(G, cluster_map)
        print(f"  seed={s:>6}: raw={len(raw_sets[s])}")

    core = sorted(set.intersection(*raw_sets.values()))
    print(f"\nstable core (raw bridge in ALL {len(SEEDS)} seeds): {len(core)} theorems")

    canon_raw = {b["theorem"] for b in json.load(open(RESULTS / "bridge_theorems.json"))}
    canon_strict = {b["theorem"] for b in json.load(open(RESULTS / "bridge_theorems_filtered.json"))}
    core_set = set(core)
    in_canon_raw = sorted(core_set & canon_raw)
    in_canon_strict = sorted(core_set & canon_strict)

    proof_len, out_prem, reuse_indeg, module = load_features()
    all_names = [t for t in proof_len if t in out_prem]
    all_names_set = set(all_names)
    depth = dag_depth(out_prem)
    getters = {
        "proof_len": lambda t: proof_len.get(t, 0),
        "n_premises": lambda t: len(out_prem.get(t, ())),
        "reuse_indeg": lambda t: reuse_indeg.get(t, 0),
        "dag_depth": lambda t: depth.get(t, 0),
    }
    features = {fn: feature_block(g, all_names_set, all_names, core) for fn, g in getters.items()}

    mod_dist = Counter(module.get(t, "?") for t in core)

    hist_names, by_bridge = historical_names()
    core_hist = sorted(core_set & hist_names)
    hist_hits = {b: sorted(core_set & names) for b, names in by_bridge.items() if core_set & names}

    out = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "core_size": len(core),
        "core_theorems": core,
        "overlap_with_canonical": {
            "canonical_raw": len(canon_raw),
            "canonical_strict": len(canon_strict),
            "core_in_canonical_raw": len(in_canon_raw),
            "core_in_canonical_raw_frac": round(len(in_canon_raw) / len(core), 3) if core else 0.0,
            "core_in_canonical_strict": len(in_canon_strict),
            "core_in_canonical_strict_names": in_canon_strict,
        },
        "structural_profile": features,
        "module_distribution": dict(mod_dist.most_common()),
        "historical_overlap": {
            "n_core_in_historical": len(core_hist),
            "core_in_historical_names": core_hist,
            "per_bridge_hits": hist_hits,
        },
        "note": "core = raw operational bridges flagged in ALL 7 seeds; rankfrac vs full theorem population (0.5=no diff).",
        "reproducibility": (
            "Node order into igraph is SORTED for determinism. e-0050 used "
            "list(set)-order, which depends on PYTHONHASHSEED and varies per "
            "process, so its 79-core was an artifact of one hash ordering "
            "(re-run gave a different core). Leiden seed= alone does NOT fix the "
            "partition: same seed=0, two hash orderings -> 141 vs 143 clusters."
        ),
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))

    print("\n=== STABLE-CORE PROFILE ===")
    print(f"core size: {len(core)}")
    print(f"in canonical 280 raw: {len(in_canon_raw)} ({out['overlap_with_canonical']['core_in_canonical_raw_frac']})")
    print(f"in canonical 34 strict: {len(in_canon_strict)} -> {in_canon_strict}")
    for fn, b in features.items():
        print(f"[{fn}] pop med {b['pop_median']} / core med {b['core_median']} "
              f"(mean {b['core_mean']}, rankfrac {b['core_rankfrac']})")
    print("module dist:", dict(mod_dist.most_common(10)))
    print(f"core in historical bridges: {len(core_hist)} {core_hist}")
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
