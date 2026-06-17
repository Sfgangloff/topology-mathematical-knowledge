"""
theta sensitivity of the bridge theorems and historical recovery (q-0012).

theta = the in-degree depth filter: a bridge theorem survives at theta if at
least one of its cross-cluster premises has in-degree <= theta (a "specialised"
premise rather than a ubiquitous utility lemma). The headline pipeline fixes
theta = 2 (elbow of the survival curve), yielding 34 bridge theorems and a
historical recovery of 2/8 strict / 6/8 loose.

Reviewers JAGf and HJ3i flagged theta = 2 as the single most load-bearing
threshold that is never stress-tested. This script sweeps theta in {1,2,3,5}
and reports, for each theta:

  * the number of surviving bridge theorems,
  * the Jaccard overlap and nesting of the surviving SET vs the theta=2 set,
  * the historical recovery (how many of the 8 known bridges are strict-detected
    at that theta, and which ones flip).

Loose recovery (any of the 280 raw bridges) is theta-independent by construction,
so the question is entirely about how the STRICT set and its 34-count headline
move with theta. Pure CPU: clusters are read from the cache, no re-clustering.

Output: results/theta_sensitivity.json
"""

import json
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cluster import load as load_clusters
from label_clusters import load as load_labels
from depth_filter import indegree_sweep
from validate_bridges import KNOWN_BRIDGES, search_theorems

THETAS = [1, 2, 3, 5]
REFERENCE_THETA = 2


def surviving_set(theorem_bridges, theta):
    return {b["theorem"] for b in theorem_bridges if b["min_cross_indegree"] <= theta}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def historical_strict(cluster_map, detected_set, all_bt_set):
    """For each known bridge return (strict_detected, loose_detected)."""
    all_theorems = list(cluster_map.keys())
    out = []
    for bridge in KNOWN_BRIDGES:
        matches = []
        for _side, patterns in bridge["sides"]:
            matches += search_theorems(all_theorems, patterns)
        strict = [t for t in matches if t in detected_set]
        loose = [t for t in matches if t in all_bt_set]
        out.append({
            "name": bridge["name"],
            "strict": bool(strict),
            "loose": bool(loose),
            "n_strict_matches": len(set(strict)),
        })
    return out


def main():
    with open(DATA_DIR / "theorem_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cluster_map = load_clusters()
    labels = load_labels()

    print("Computing raw bridge candidates (indegree sweep) ...")
    _thresholds, _counts, theorem_bridges = indegree_sweep(G, cluster_map, labels)
    print(f"Raw indegree-sweep candidates (>=2 pure clusters): {len(theorem_bridges)}")

    # loose set = the canonical unfiltered bridge_theorems.json (280), exactly the
    # set validate_bridges.py uses for the recorded 6/8-loose baseline. theta only
    # gates the STRICT (depth-filtered) set, so loose recovery is theta-independent.
    with open(RESULTS_DIR / "bridge_theorems.json") as f:
        all_bt_set = {b["theorem"] for b in json.load(f)}
    print(f"Canonical loose set (bridge_theorems.json): {len(all_bt_set)}")

    ref_set = surviving_set(theorem_bridges, REFERENCE_THETA)
    print(f"Reference theta={REFERENCE_THETA}: {len(ref_set)} bridge theorems")

    per_theta = {}
    for theta in THETAS:
        surv = surviving_set(theorem_bridges, theta)
        hist = historical_strict(cluster_map, surv, all_bt_set)
        n_strict = sum(1 for h in hist if h["strict"])
        n_loose = sum(1 for h in hist if h["loose"])
        per_theta[theta] = {
            "n_bridges": len(surv),
            "jaccard_vs_ref": round(jaccard(surv, ref_set), 4),
            "subset_of_ref": surv.issubset(ref_set),
            "superset_of_ref": surv.issuperset(ref_set),
            "n_added_vs_ref": len(surv - ref_set),
            "n_dropped_vs_ref": len(ref_set - surv),
            "historical_strict": n_strict,
            "historical_loose": n_loose,
            "historical_detail": {h["name"]: h["strict"] for h in hist},
        }
        print(f"\ntheta={theta}: n_bridges={len(surv)}  "
              f"jaccard_vs_ref={per_theta[theta]['jaccard_vs_ref']}  "
              f"subset={per_theta[theta]['subset_of_ref']}  "
              f"hist_strict={n_strict}/8  hist_loose={n_loose}/8")
        for h in hist:
            print(f"    {h['name'][:38]:<38}  strict={h['strict']}")

    # nesting check: are the sets monotone (theta1 subset theta2 subset theta3...)?
    sets = {t: surviving_set(theorem_bridges, t) for t in THETAS}
    monotone = all(sets[THETAS[i]].issubset(sets[THETAS[i + 1]])
                   for i in range(len(THETAS) - 1))

    # which historical bridges are STABLE strict across the whole sweep
    stable_strict = []
    flips = []
    names = [b["name"] for b in KNOWN_BRIDGES]
    for name in names:
        vals = [per_theta[t]["historical_detail"][name] for t in THETAS]
        if all(vals):
            stable_strict.append(name)
        elif any(vals):
            flips.append({"name": name, "by_theta": dict(zip(THETAS, vals))})

    summary = {
        "thetas": THETAS,
        "reference_theta": REFERENCE_THETA,
        "n_raw_candidates": len(theorem_bridges),
        "per_theta": per_theta,
        "sets_monotone_in_theta": monotone,
        "historical_stable_strict_across_sweep": stable_strict,
        "historical_strict_flips": flips,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "theta_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"sets monotone (nested) in theta: {monotone}")
    print(f"historical strict STABLE across theta in {THETAS}: "
          f"{stable_strict} ({len(stable_strict)}/8)")
    print(f"historical strict FLIPS: {[f['name'] for f in flips]}")
    print(f"\nSaved -> results/theta_sensitivity.json")


if __name__ == "__main__":
    main()
