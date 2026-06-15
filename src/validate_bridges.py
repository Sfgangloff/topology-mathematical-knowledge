"""
Validation against known historical bridges in mathematics.

For each known bridge, we:
  1. Search the cluster map for relevant Mathlib theorem names
  2. Check which clusters they fall in (do they span ≥2 distinct clusters?)
  3. Check if they appear in our bridge detector outputs
  4. Report a precision/recall table

Known bridges tested:
  A. Zorn's Lemma ↔ Axiom of Choice ↔ Well-Ordering
  B. Galois theory ↔ Fundamental group / covering spaces
  C. Stone duality: Boolean algebras ↔ Stone spaces (compact totally disconnected)
  D. Pontryagin duality: locally compact abelian groups ↔ character groups
  E. Fixed-point theorems: Brouwer (topology) ↔ Lawvere (category theory)
  F. Gromov-Hausdorff distance (metric geometry ↔ topology ↔ set theory)
  G. Spectral theory ↔ Representation theory
  H. Lie algebras ↔ Lie groups (Engel's theorem, exponential map)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Known bridge definitions ──────────────────────────────────────────────────
# Each entry: name, list of (domain_hint, search_patterns)
# search_patterns are substrings matched against full theorem names

KNOWN_BRIDGES = [
    {
        "name": "Zorn / Choice / Well-Ordering",
        "description": "Equivalence of Zorn's Lemma, Axiom of Choice, and Well-Ordering Theorem",
        "sides": [
            ("Order/Set theory",  ["zorn", "Zorn", "WellOrder", "wellOrder",
                                   "WellFounded", "linearOrder_of"]),
            ("Logic/Foundations", ["axiomOfChoice", "exists_maximal", "Classical.choice",
                                   "nonempty_iff_ne_empty", "WellOrderExtension"]),
        ],
    },
    {
        "name": "Galois ↔ Covering spaces",
        "description": "Galois correspondence mirrors π₁ action on covering spaces",
        "sides": [
            ("Field theory / Galois", ["Galois", "galois", "GaloisGroup",
                                        "IsGalois", "GaloisCorrespondence",
                                        "separableClosure", "normalClosure"]),
            ("Topology / Covering",   ["CoveringSpace", "coveringSpace", "FundamentalGroup",
                                        "fundamentalGroup", "UniversalCover", "monodromy"]),
        ],
    },
    {
        "name": "Stone duality",
        "description": "Boolean algebras ↔ compact totally disconnected spaces",
        "sides": [
            ("Boolean algebra",  ["BooleanAlgebra", "BooleanRing", "Stone.ofBoolAlg",
                                   "BoolAlg", "Finset.toBoolAlg"]),
            ("Stone spaces",     ["StoneCech", "TotallySeparated", "totallySeparated",
                                   "Profinite", "profinite", "StoneSpace"]),
        ],
    },
    {
        "name": "Pontryagin duality",
        "description": "Locally compact abelian groups ↔ their Pontryagin duals",
        "sides": [
            ("Pontryagin / Characters", ["Pontryagin", "pontryagin", "PontryaginDual",
                                          "characterGroup", "CharacterGroup",
                                          "AddChar", "addChar"]),
            ("Harmonic analysis",        ["fourierTransform", "FourierTransform",
                                          "fourier", "Fourier", "fourierSeries"]),
        ],
    },
    {
        "name": "Fixed-point theorems",
        "description": "Brouwer (topology), Lawvere (category theory), Schauder (analysis)",
        "sides": [
            ("Topology (Brouwer)",     ["Brouwer", "brouwer", "fixedPoint",
                                        "FixedPoint", "NoRetraction"]),
            ("Category theory (Lawvere)", ["Lawvere", "lawvere", "CategoryTheory.fixedPoint",
                                           "diagonal", "Diagonal"]),
        ],
    },
    {
        "name": "Gromov-Hausdorff",
        "description": "Metric geometry ↔ topology ↔ set theory",
        "sides": [
            ("Metric geometry", ["GromovHausdorff", "gromovHausdorff", "hausdorffDist",
                                  "HausdorffDist", "ghDist"]),
            ("Topology",        ["MetricSpace", "Bounded", "TotallyBounded",
                                  "isCompact", "IsCompact"]),
        ],
    },
    {
        "name": "Spectral theory ↔ Representation theory",
        "description": "Spectral decomposition mirrors decomposition of representations",
        "sides": [
            ("Spectral theory",      ["spectrum", "Spectrum", "resolvent", "Resolvent",
                                       "eigenspace", "Eigenspace", "spectral"]),
            ("Representation theory", ["Representation", "representation", "IrreducibleRep",
                                        "CharacterOfRep", "Schur", "isotypic"]),
        ],
    },
    {
        "name": "Lie algebras ↔ Lie groups",
        "description": "Engel's theorem, Lie bracket, exponential map",
        "sides": [
            ("Lie algebras", ["LieAlgebra", "lieAlgebra", "LieBracket",
                               "Engel", "engel", "nilpotent_of_forall"]),
            ("Lie groups",   ["LieGroup", "lieGroup", "expMap", "expMapSphere",
                               "SmoothManifold", "smoothManifold"]),
        ],
    },
]


# ── Search helpers ────────────────────────────────────────────────────────────

def search_theorems(all_theorems: list[str], patterns: list[str]) -> list[str]:
    """Return theorem names matching any of the patterns (case-sensitive substring)."""
    results = []
    for thm in all_theorems:
        for pat in patterns:
            if pat in thm:
                results.append(thm)
                break
    return results


def clusters_of(theorems: list[str], cluster_map: dict) -> dict[int, list[str]]:
    """Group theorem names by their cluster id. Returns {cluster_id: [theorem_names]}."""
    by_cluster = defaultdict(list)
    for t in theorems:
        cid = cluster_map.get(t)
        if cid is not None:
            by_cluster[cid].append(t)
    return dict(by_cluster)


# ── Main validation ───────────────────────────────────────────────────────────

def validate(
    cluster_map: dict[str, int],
    labels: dict[int, dict],
    bridge_theorems_filtered: list[dict],
    bridge_theorems_all: list[dict],
) -> list[dict]:
    """Check each known historical bridge against our cluster decomposition and detectors.

    For each KNOWN_BRIDGES entry, searches both sides' theorem names in the cluster map
    and checks:
      1. Whether the two sides land in distinct clusters (domain structure recovery).
      2. Whether any matching theorem was flagged by our bridge detector (detection recall).

    "Strict" detection = appears in depth-filtered results (θ≤2 in-degree).
    "Loose" detection = appears in any unfiltered bridge theorem output.
    """
    all_theorems = list(cluster_map.keys())
    detected_set = {b["theorem"] for b in bridge_theorems_filtered}

    results = []
    for bridge in KNOWN_BRIDGES:
        entry = {"name": bridge["name"], "description": bridge["description"], "sides": []}
        side_clusters = []

        for side_name, patterns in bridge["sides"]:
            matches = search_theorems(all_theorems, patterns)
            by_cluster = clusters_of(matches, cluster_map)
            cluster_labels = {
                cid: labels.get(cid, {}).get("label", str(cid))
                for cid in by_cluster
            }
            entry["sides"].append({
                "side": side_name,
                "patterns": patterns,
                "n_matches": len(matches),
                "clusters": {str(c): {"label": cluster_labels[c], "theorems": ts[:5]}
                             for c, ts in sorted(by_cluster.items(), key=lambda x: -len(x[1]))},
                "sample_theorems": matches[:8],
            })
            side_clusters.append(set(by_cluster.keys()))

        # Check if the two sides land in DIFFERENT clusters
        if len(side_clusters) == 2:
            intersection = side_clusters[0] & side_clusters[1]
            union = side_clusters[0] | side_clusters[1]
            entry["clusters_overlap"] = len(intersection)
            entry["clusters_separated"] = len(union) - len(intersection)
            entry["bridge_spans_clusters"] = len(side_clusters[0]) > 0 and \
                                              len(side_clusters[1]) > 0 and \
                                              not side_clusters[0].issubset(side_clusters[1]) and \
                                              not side_clusters[1].issubset(side_clusters[0])
        else:
            entry["bridge_spans_clusters"] = False

        # Check against ALL matches (not just sample) in both detectors
        all_matches_full = []
        for side_name, patterns in bridge["sides"]:
            all_matches_full += search_theorems(all_theorems, patterns)

        detected_filtered = [t for t in all_matches_full if t in detected_set]
        detected_all      = [t for t in all_matches_full
                             if t in {b["theorem"] for b in bridge_theorems_all}]
        entry["detected_by_depth_filter"] = detected_filtered
        entry["detected_by_any_bridge"]   = detected_all

        results.append(entry)

    return results


def print_report(results: list[dict], labels: dict) -> None:
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST KNOWN MATHEMATICAL BRIDGES")
    print("=" * 80)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"  {r['name']}")
        print(f"  {r['description']}")
        print(f"  Bridge spans distinct clusters: "
              f"{'YES ✓' if r.get('bridge_spans_clusters') else 'no ✗'}")

        for side in r["sides"]:
            print(f"\n  [{side['side']}] — {side['n_matches']} theorems found")
            for cid_str, info in list(side["clusters"].items())[:4]:
                print(f"    Cluster {cid_str} ({info['label']}): {info['theorems'][:3]}")
            if not side["clusters"]:
                print(f"    (no matches in dataset)")

        if r.get("detected_by_depth_filter"):
            print(f"\n  DETECTED (filtered θ≤2): {r['detected_by_depth_filter'][:4]}")
        elif r.get("detected_by_any_bridge"):
            print(f"\n  DETECTED (unfiltered):   {r['detected_by_any_bridge'][:4]}")
        else:
            print(f"\n  Not detected by premise-based bridge detector")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'Bridge':<35}  {'Found':<6}  {'Spans clusters':<16}  {'Detected'}")
    print("─" * 75)
    for r in results:
        side0_found = r["sides"][0]["n_matches"] > 0 if r["sides"] else False
        side1_found = r["sides"][1]["n_matches"] > 0 if len(r["sides"]) > 1 else False
        found = "yes" if side0_found and side1_found else ("partial" if side0_found or side1_found else "no")
        spans = "YES" if r.get("bridge_spans_clusters") else "no"
        if r.get("detected_by_depth_filter"):
            detected = "YES (strict)"
        elif r.get("detected_by_any_bridge"):
            detected = "YES (loose)"
        else:
            detected = "no"
        print(f"  {r['name']:<33}  {found:<6}  {spans:<16}  {detected}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from cluster import load as load_clusters
    from label_clusters import load as load_labels

    cluster_map = load_clusters()
    labels = load_labels()

    # Load bridge theorem outputs
    def load_bridge_theorems(path: Path) -> list[dict]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    filtered = load_bridge_theorems(RESULTS_DIR / "bridge_theorems_filtered.json")
    all_bt   = load_bridge_theorems(RESULTS_DIR / "bridge_theorems.json")

    results = validate(cluster_map, labels, filtered, all_bt)
    print_report(results, labels)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "validation.json", "w") as f:
        # Serialise without sample theorems for brevity
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → results/validation.json")
