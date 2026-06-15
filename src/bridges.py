"""
Bridge detection with entropy-based scoring.

Bridge score for an n-gram g:

  score(g) = H_norm(g) * rarity(g)

where:
  H_norm(g) = entropy of g's occurrence distribution across clusters,
              normalised by log2(K) so it lies in [0,1].
              High H_norm → g is genuinely spread across many clusters,
              not dominated by one.

  rarity(g) = 1 - mean_relative_freq(g)
              High rarity → g is locally uncommon in each cluster it appears in.

This penalises two failure modes:
  - Generic tactics (simp, intro): appear in many clusters BUT with high TF → low rarity.
  - Domain-specific sequences: appear in few clusters → low H_norm.
  - Sequences dominated by one cluster: H_norm < 1 even if df is high.

We also filter out n-grams whose individual tokens are all "top-k generic" tactics,
as configurable by `generic_tactic_threshold`.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from normalize_tactics import normalize_sequence

RESULTS_DIR = Path(__file__).parent.parent / "results"
BRIDGES_FILE = RESULTS_DIR / "bridges.json"

# Tactics so generic they shouldn't dominate a bridge sequence.
# Any n-gram made *entirely* of these is excluded.
GENERIC_TACTICS = {
    "simp", "intro", "exact", "apply", "rfl", "ring", "norm_num",
    "linarith", "omega", "decide", "trivial", "tauto", "aesop",
    "constructor", "use", "left", "right", "ext", "congr",
    "push_neg", "contrapose", "by_contra", "by_cases",
}


def _normalise(t: str) -> str:
    return " ".join(t.split())


def _ngrams(seq: list[str], n: int):
    """Yield all length-n contiguous subsequences of seq."""
    it = iter(seq)
    window = list(islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for item in it:
        window = window[1:] + [item]
        yield tuple(window)


def _is_all_generic(gram: tuple[str, ...]) -> bool:
    """Return True if every token in the n-gram is a domain-agnostic tactic keyword.

    Such n-grams (e.g. 'simp → exact → ring') are shared boilerplate, not bridges.
    """
    return all(t.strip().split()[0] in GENERIC_TACTICS for t in gram)


def _entropy(counts: list[float]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def extract_ngram_profiles(
    theorem_tactics: dict[str, list[str]],
    cluster_map: dict[str, int],
    n: int = 3,
    min_tactic_steps: int = 3,
) -> dict[int, Counter]:
    """Build per-cluster frequency profiles of normalized n-gram tactic sequences.

    Theorems with fewer than min_tactic_steps are skipped (too short to carry structure).
    Returns {cluster_id: Counter({n-gram tuple: count})}.
    """
    cluster_ngrams: dict[int, Counter] = defaultdict(Counter)
    for theorem, tactics in theorem_tactics.items():
        cid = cluster_map.get(theorem)
        if cid is None:
            continue
        normed = normalize_sequence([_normalise(t) for t in tactics])
        if len(normed) < min_tactic_steps:
            continue
        for gram in _ngrams(normed, n):
            cluster_ngrams[cid][gram] += 1
    return cluster_ngrams


def find_bridges(
    cluster_ngrams: dict[int, Counter],
    min_clusters: int = 2,
    min_count: int = 1,
    top_k: int = 300,
    filter_generic: bool = True,
) -> list[dict]:
    """
    min_count: minimum number of times the n-gram must appear in a cluster
               for that cluster to count. Filters out accidental one-off
               appearances that inflate n_clusters spuriously.
    """
    K = len(cluster_ngrams)
    cluster_totals = {cid: sum(c.values()) for cid, c in cluster_ngrams.items()}

    gram_clusters: dict[tuple, dict[int, int]] = defaultdict(dict)
    for cid, counter in cluster_ngrams.items():
        for gram, cnt in counter.items():
            if cnt >= min_count:
                gram_clusters[gram][cid] = cnt

    results = []
    for gram, cid_counts in gram_clusters.items():
        if len(cid_counts) < min_clusters:
            continue
        if filter_generic and _is_all_generic(gram):
            continue

        # Relative frequencies within each cluster
        rel_freqs = {
            cid: cnt / cluster_totals[cid]
            for cid, cnt in cid_counts.items()
        }

        # H_norm: entropy of occurrence distribution across clusters
        raw_counts = list(cid_counts.values())
        ent = _entropy(raw_counts)
        max_ent = math.log2(K)
        h_norm = ent / max_ent if max_ent > 0 else 0.0

        # Rarity: 1 - mean within-cluster relative frequency
        mean_rel = float(np.mean(list(rel_freqs.values())))
        rarity = 1.0 - mean_rel

        score = h_norm * rarity

        results.append({
            "ngram": list(gram),
            "n_clusters": len(cid_counts),
            "h_norm": round(h_norm, 4),
            "rarity": round(rarity, 6),
            "bridge_score": round(score, 6),
            "mean_relative_freq": round(mean_rel, 8),
            "cluster_counts": {str(k): v for k, v in cid_counts.items()},
        })

    results.sort(key=lambda x: -x["bridge_score"])
    return results[:top_k]


def save(bridges: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(BRIDGES_FILE, "w") as f:
        json.dump(bridges, f, indent=2)
    print(f"Saved {len(bridges)} bridge candidates → {BRIDGES_FILE}")


def report(bridges: list[dict], top: int = 20) -> None:
    print(f"\nTop {top} bridge tactic sequences (entropy × rarity score):\n")
    print(f"{'Rk':>3}  {'N-gram':<58}  {'NCl':>4}  {'H':>5}  {'Score':>8}")
    print("-" * 85)
    for i, b in enumerate(bridges[:top]):
        gram_str = " → ".join(b["ngram"])
        if len(gram_str) > 57:
            gram_str = gram_str[:54] + "..."
        print(
            f"{i+1:>3}  {gram_str:<58}  "
            f"{b['n_clusters']:>4}  "
            f"{b['h_norm']:>5.3f}  "
            f"{b['bridge_score']:>8.5f}"
        )


if __name__ == "__main__":
    import json
    from pathlib import Path

    DATA_DIR = Path(__file__).parent.parent / "data"
    with open(DATA_DIR / "theorem_tactics.json") as f:
        theorem_tactics = json.load(f)
    with open(DATA_DIR / "clusters.json") as f:
        cluster_map = {k: int(v) for k, v in json.load(f).items()}

    all_bridges = []
    for n in [3, 4, 5]:
        print(f"Extracting {n}-gram profiles ...")
        profiles = extract_ngram_profiles(theorem_tactics, cluster_map, n=n)
        all_bridges += find_bridges(profiles, min_clusters=2)

    all_bridges.sort(key=lambda x: -x["bridge_score"])
    save(all_bridges)
    report(all_bridges)
