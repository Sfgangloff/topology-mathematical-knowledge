"""
q-0031: If the only null-beating cross-domain signal is the DEPLETION of
cross-module premise links (a-0049/a-0057), can a per-domain-PAIR observed-vs-null
enrichment matrix serve as a STABLE structural map, and does it recover meaningful
mathematical adjacencies?

Construction (no clustering; human top-level Mathlib module labels):
  For every directed premise edge A->B (A's proof uses B), bin by
  (module(A), module(B)). The OBSERVED inter-module premise count O[i][j] is the
  number of edges from module i to module j (i != j).

  NULL: degree-preserving configuration expectation. With out-strength s_out[i]
  (edges leaving module i to ANY module) and in-strength s_in[j] (edges arriving
  at module j), under random rewiring that preserves each node's degree the
  expected cross-module count is
      E[i][j] = s_out[i] * s_in[j] / M
  where M = total edges. (Standard configuration-model expectation at the
  module-aggregate level.)

  ENRICHMENT[i][j] = log2( (O[i][j] + eps) / (E[i][j] + eps) ).
  >0 = more premise links than chance (domains co-develop), <0 = depleted.

Outputs: the symmetric enrichment matrix (we report the symmetrised pair
enrichment using O[i][j]+O[j][i] vs E[i][j]+E[j][i]); the top enriched and top
depleted domain pairs; and a face-validity check against a small hand-labelled
set of EXPECTED-adjacent pairs (Topology-Analysis, Analysis-MeasureTheory,
NumberTheory-Algebra, ...) vs EXPECTED-distant pairs (Topology-Combinatorics,
SetTheory-Geometry, ...): does the matrix rank expected-adjacent above
expected-distant? (AUC-style separation.)
"""
import json, pickle, math
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
EPS = 1.0


def top_module(module: str):
    parts = module.split(".")
    if "Mathlib" in parts:
        i = parts.index("Mathlib")
        return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    return None  # exclude non-Mathlib library nodes


def main():
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    node_mod = {}
    for n in G.nodes():
        m = top_module(G.nodes[n].get("module", ""))
        if m is not None:
            node_mod[n] = m

    mod_sizes = Counter(node_mod.values())
    MODS = sorted(m for m, c in mod_sizes.items() if c >= 50)
    modset = set(MODS)

    # observed inter-module premise counts (directed A->B)
    O = defaultdict(int)
    s_out = defaultdict(int)
    s_in = defaultdict(int)
    M = 0
    for a, b in G.edges():
        ma = node_mod.get(a); mb = node_mod.get(b)
        if ma is None or mb is None or ma not in modset or mb not in modset:
            continue
        s_out[ma] += 1
        s_in[mb] += 1
        M += 1
        if ma != mb:
            O[(ma, mb)] += 1

    # symmetrised pair enrichment
    pairs = []
    for i, j in combinations(MODS, 2):
        o = O[(i, j)] + O[(j, i)]
        e = (s_out[i] * s_in[j] + s_out[j] * s_in[i]) / M
        enr = math.log2((o + EPS) / (e + EPS))
        pairs.append({"pair": f"{i}-{j}", "i": i, "j": j,
                      "observed": o, "expected": round(e, 1),
                      "log2_enrichment": round(enr, 3)})

    pairs_sorted = sorted(pairs, key=lambda p: p["log2_enrichment"], reverse=True)
    enr_lookup = {(p["i"], p["j"]): p["log2_enrichment"] for p in pairs}
    enr_lookup.update({(p["j"], p["i"]): p["log2_enrichment"] for p in pairs})

    # ---- face-validity: expected-adjacent vs expected-distant pairs ----
    expected_adjacent = [
        ("Topology", "Analysis"), ("Analysis", "MeasureTheory"),
        ("MeasureTheory", "Probability"), ("NumberTheory", "Algebra"),
        ("RingTheory", "FieldTheory"), ("Algebra", "LinearAlgebra"),
        ("Topology", "Geometry"), ("CategoryTheory", "Algebra"),
        ("GroupTheory", "Algebra"), ("Analysis", "Topology"),
    ]
    expected_distant = [
        ("Combinatorics", "Topology"), ("SetTheory", "Geometry"),
        ("Computability", "Geometry"), ("NumberTheory", "Topology"),
        ("Combinatorics", "MeasureTheory"), ("Logic", "Geometry"),
        ("Computability", "MeasureTheory"), ("SetTheory", "Probability"),
        ("Dynamics", "NumberTheory"), ("ModelTheory", "Topology"),
    ]

    def lookup_avail(lst):
        out = []
        for a, b in lst:
            if a in modset and b in modset and (a, b) in enr_lookup:
                out.append({"pair": f"{a}-{b}", "enr": enr_lookup[(a, b)]})
        return out

    adj = lookup_avail(expected_adjacent)
    dist = lookup_avail(expected_distant)
    adj_vals = [x["enr"] for x in adj]
    dist_vals = [x["enr"] for x in dist]

    # AUC: P(adjacent ranked above distant)
    wins = ties = 0
    for av in adj_vals:
        for dv in dist_vals:
            if av > dv: wins += 1
            elif av == dv: ties += 1
    total = len(adj_vals) * len(dist_vals)
    auc = (wins + 0.5 * ties) / total if total else None

    import statistics
    out = {
        "n_modules": len(MODS),
        "modules": MODS,
        "total_intra_plus_inter_edges": M,
        "top_enriched_pairs": pairs_sorted[:15],
        "top_depleted_pairs": pairs_sorted[-15:],
        "n_pairs_enriched_gt0": sum(1 for p in pairs if p["log2_enrichment"] > 0),
        "n_pairs_depleted_lt0": sum(1 for p in pairs if p["log2_enrichment"] < 0),
        "n_pairs_total": len(pairs),
        "face_validity": {
            "expected_adjacent": adj,
            "expected_distant": dist,
            "adjacent_mean_enrichment": round(statistics.mean(adj_vals), 3) if adj_vals else None,
            "distant_mean_enrichment": round(statistics.mean(dist_vals), 3) if dist_vals else None,
            "separation_AUC": round(auc, 3) if auc is not None else None,
            "interpretation": "AUC=1 means matrix perfectly ranks expected-adjacent above expected-distant; 0.5 = no better than chance",
        },
    }
    json.dump(out, open(RES / "domain_adjacency_matrix.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "modules"}, indent=2))


if __name__ == "__main__":
    main()
