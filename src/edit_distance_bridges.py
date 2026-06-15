"""
Step 2: Edit-distance bridge detection.

For each pair of clusters (A, B), find the pair of theorems (T_A, T_B) whose
normalised tactic sequences are most similar (lowest edit distance).

A low edit distance across a cluster boundary is surprising — it means the same
proof structure is used in two different domains, even if the specific theorems
referenced are completely different.

We use token-level edit distance on normalised tactic sequences.

Strategy (tractable approximation):
  - Only consider theorems with ≥ MIN_TACTICS tactic steps (longer proofs → more informative)
  - For each cluster, take the TOP_PER_CLUSTER longest proofs
  - Compute all pairwise cross-cluster edit distances (within-cluster pairs used as baseline)
  - Report cluster-pairs where min cross-cluster edit distance < baseline threshold

Output:
  results/edit_distance_bridges.json
  results/edit_distance_report.txt
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MIN_TACTICS     = 5      # minimum proof length (in tactics)
TOP_PER_CLUSTER = 30     # take this many longest proofs per cluster
MAX_CLUSTERS    = 30     # only consider top-N largest clusters (tractability)


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Standard token-level Levenshtein distance."""
    m, n = len(a), len(b)
    if m > 80 or n > 80:   # cap very long proofs
        a, b = a[:80], b[:80]
        m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(dp[j], dp[j-1], prev)
            prev = temp
    return dp[n]


def _normalised_ed(a: list[str], b: list[str]) -> float:
    """Edit distance normalised by the length of the longer sequence."""
    raw = _edit_distance(a, b)
    return raw / max(len(a), len(b), 1)


def find_edit_bridges(
    theorem_tactics: dict[str, list[str]],
    cluster_map: dict[str, int],
    labels: dict[int, dict],
    top_k: int = 200,
) -> list[dict]:
    """Find cross-cluster theorem pairs with anomalously similar normalized proof sequences.

    For each pair of clusters, identifies the two theorems (one per cluster) whose
    normalized tactic sequences have the smallest Levenshtein edit distance. A low
    cross-cluster edit distance is surprising: it means the same abstract proof
    structure appears in two different mathematical domains even when the specific
    theorems referenced are entirely different — a "conceptual bridge."

    The within-cluster baseline (median normalized ED) establishes what "typical
    similarity" looks like. Only pairs below the baseline are returned.

    Tractability: only the top MAX_CLUSTERS clusters (by number of qualifying proofs)
    are compared, taking TOP_PER_CLUSTER longest proofs per cluster.
    """
    from normalize_tactics import normalize_sequence

    # Pre-normalise all proofs
    normed: dict[str, list[str]] = {}
    for thm, tacs in theorem_tactics.items():
        seq = normalize_sequence([t.strip() for t in tacs])
        if len(seq) >= MIN_TACTICS:
            normed[thm] = seq

    # Group by cluster, keep longest proofs
    by_cluster: dict[int, list[tuple]] = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))

    # Sort each cluster by proof length descending, keep top N
    target_clusters = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    for cid in target_clusters:
        by_cluster[cid] = sorted(by_cluster[cid], key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]

    print(f"Comparing proofs across {len(target_clusters)} clusters "
          f"(≤{TOP_PER_CLUSTER} per cluster, ≥{MIN_TACTICS} tactics each)")

    # Compute within-cluster baseline edit distances
    within_eds: dict[int, list[float]] = {}
    for cid in target_clusters:
        proofs = by_cluster[cid]
        if len(proofs) < 2:
            within_eds[cid] = [1.0]
            continue
        eds = [
            _normalised_ed(proofs[i][1], proofs[j][1])
            for i in range(min(10, len(proofs)))
            for j in range(i + 1, min(10, len(proofs)))
        ]
        within_eds[cid] = eds

    baseline = float(np.median([ed for eds in within_eds.values() for ed in eds]))
    print(f"Median within-cluster normalised edit distance: {baseline:.3f}")

    # Cross-cluster edit distances
    results = []
    pairs = list(combinations(target_clusters, 2))
    for cid_a, cid_b in tqdm(pairs, desc="Cross-cluster ED"):
        proofs_a = by_cluster[cid_a]
        proofs_b = by_cluster[cid_b]

        best_ed = 1.0
        best_ta = best_tb = None

        for ta, seq_a in proofs_a:
            for tb, seq_b in proofs_b:
                ed = _normalised_ed(seq_a, seq_b)
                if ed < best_ed:
                    best_ed = ed
                    best_ta, best_tb = ta, tb

        if best_ta is None:
            continue

        surprise = baseline - best_ed   # positive = more similar than baseline
        if best_ed >= baseline:
            continue

        results.append({
            "cluster_a": cid_a,
            "cluster_b": cid_b,
            "label_a": labels.get(cid_a, {}).get("label", str(cid_a)),
            "label_b": labels.get(cid_b, {}).get("label", str(cid_b)),
            "theorem_a": best_ta,
            "theorem_b": best_tb,
            "norm_edit_dist": round(best_ed, 4),
            "surprise": round(surprise, 4),
            "seq_a": normed[best_ta],
            "seq_b": normed[best_tb],
        })

    results.sort(key=lambda x: x["norm_edit_dist"])
    return results[:top_k]


def save(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # Don't save sequences to JSON (too large), save a summary
    summary = [{k: v for k, v in r.items() if k not in ("seq_a", "seq_b")} for r in results]
    with open(RESULTS_DIR / "edit_distance_bridges.json", "w") as f:
        json.dump(summary, f, indent=2)


def report(results: list[dict], top: int = 30) -> None:
    print(f"\nTop {top} cross-cluster proof similarity (lowest normalised edit distance):\n")
    print(f"{'Rk':>3}  {'Domain A':<28}  {'Domain B':<28}  {'ED':>6}  {'Surprise':>8}")
    print("-" * 80)
    for i, r in enumerate(results[:top]):
        la = r["label_a"][:27]
        lb = r["label_b"][:27]
        print(f"{i+1:>3}  {la:<28}  {lb:<28}  {r['norm_edit_dist']:>6.3f}  {r['surprise']:>8.3f}")

    print(f"\nTop similarity pairs with proof sequences:\n")
    for r in results[:5]:
        print(f"--- {r['label_a']}  ↔  {r['label_b']}  (ED={r['norm_edit_dist']:.3f}) ---")
        print(f"  A: {r['theorem_a']}")
        print(f"     {' → '.join(r['seq_a'][:8])}{'...' if len(r['seq_a']) > 8 else ''}")
        print(f"  B: {r['theorem_b']}")
        print(f"     {' → '.join(r['seq_b'][:8])}{'...' if len(r['seq_b']) > 8 else ''}")
        print()


if __name__ == "__main__":
    from build_graph import load as load_graph
    from cluster import load as load_clusters
    from label_clusters import load as load_labels

    _, theorem_tactics = load_graph()
    cluster_map = load_clusters()
    labels = load_labels()

    results = find_edit_bridges(theorem_tactics, cluster_map, labels)
    print(f"\nFound {len(results)} cross-cluster bridge pairs below baseline ED")
    save(results)
    report(results)

    with open(RESULTS_DIR / "edit_distance_report.txt", "w") as f:
        for r in results[:100]:
            f.write(f"ED={r['norm_edit_dist']:.3f}  surprise={r['surprise']:.3f}\n")
            f.write(f"  A [{r['label_a']}]: {r['theorem_a']}\n")
            f.write(f"    {' → '.join(r['seq_a'][:10])}\n")
            f.write(f"  B [{r['label_b']}]: {r['theorem_b']}\n")
            f.write(f"    {' → '.join(r['seq_b'][:10])}\n\n")
    print(f"Full report → results/edit_distance_report.txt")
