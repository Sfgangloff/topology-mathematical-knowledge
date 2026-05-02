"""
Sweep n = 1..8 and measure how cross-cluster coverage degrades.

For each n we compute:
  - total unique n-grams observed
  - fraction appearing in >= k clusters, for k = 2, 3, 5
  - median and 90th-percentile of n_clusters across all n-grams

The "right" n is the elbow where coverage drops to near-singleton —
i.e. where most n-grams become cluster-specific.

Saves results/ngram_sweep.json and results/ngram_sweep.png
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from itertools import islice

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from normalize_tactics import normalize_sequence

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ngrams(seq: list[str], n: int):
    """Yield all length-n contiguous subsequences of seq."""
    it = iter(seq)
    window = list(islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for item in it:
        window = window[1:] + [item]
        yield tuple(window)


def _normalise(t: str) -> str:
    return " ".join(t.split())


# Punctuation characters that act as separators but are not meaningful tokens.
_PUNCT_RE = re.compile(r'[,;\[\]⟨⟩(){}]')


def _tokenise(normalised_tactic: str) -> list[str]:
    """Split a normalised tactic string into individual tokens.

    We strip bracket/comma punctuation and split on whitespace so that
    n-grams are computed over the token stream rather than over whole
    tactic strings.  This ensures that common tokens like ``@``, ``rw``,
    ``simp`` appear as unigrams shared across many clusters, which is
    the correct baseline for measuring cross-cluster strategy similarity.
    """
    cleaned = _PUNCT_RE.sub(" ", normalised_tactic)
    return [tok for tok in cleaned.split() if tok]


def sweep(
    theorem_tactics: dict[str, list[str]],
    cluster_map: dict[str, int],
    max_n: int = 8,
    min_tactic_steps: int = 3,
) -> list[dict]:
    """Sweep n-gram lengths 1..max_n and measure cross-cluster coverage at each n.

    For each n we compute: total unique n-grams, fraction appearing in ≥2/3/5 clusters,
    and median/90th-pct cluster spread. N-grams are computed over the token stream
    (after stripping punctuation) so that common tokens like `@`, `rw`, `simp` appear
    as shared unigrams — giving the correct baseline for cross-domain strategy similarity.

    The "right" n for bridge detection is the elbow where frac_ge2 drops below ~2%.
    Returns a list of per-n result dicts.
    """
    results = []
    cluster_ids = set(cluster_map.values())
    K = len(cluster_ids)

    # Pre-normalise all tactic sequences once
    normed_tactics = {
        thm: normalize_sequence([_normalise(t) for t in tacs])
        for thm, tacs in theorem_tactics.items()
    }

    for n in range(1, max_n + 1):
        # cluster → Counter of n-grams (computed over the token stream,
        # not over whole tactic strings, so that common tokens like @, rw,
        # simp appear as shared unigrams across clusters)
        cluster_ngrams: dict[int, Counter] = defaultdict(Counter)
        for theorem, normed in normed_tactics.items():
            cid = cluster_map.get(theorem)
            if cid is None:
                continue
            if len(normed) < min_tactic_steps:
                continue
            # Flatten tactic strings into a single token stream
            token_stream = [tok for tac in normed for tok in _tokenise(tac)]
            for gram in _ngrams(token_stream, n):
                cluster_ngrams[cid][gram] += 1

        # invert: gram → set of cluster ids
        gram_clusters: dict[tuple, set] = defaultdict(set)
        for cid, counter in cluster_ngrams.items():
            for gram in counter:
                gram_clusters[gram].add(cid)

        n_clusters_per_gram = np.array([len(v) for v in gram_clusters.values()])
        total = len(n_clusters_per_gram)

        row = {
            "n": n,
            "total_unique": total,
            "frac_ge2": float((n_clusters_per_gram >= 2).sum() / total) if total else 0,
            "frac_ge3": float((n_clusters_per_gram >= 3).sum() / total) if total else 0,
            "frac_ge5": float((n_clusters_per_gram >= 5).sum() / total) if total else 0,
            "median_clusters": float(np.median(n_clusters_per_gram)) if total else 0,
            "p90_clusters": float(np.percentile(n_clusters_per_gram, 90)) if total else 0,
            "max_clusters": int(n_clusters_per_gram.max()) if total else 0,
        }
        print(
            f"n={n}: {total:>7} unique grams | "
            f"≥2 clusters: {row['frac_ge2']:.1%} | "
            f"≥3: {row['frac_ge3']:.1%} | "
            f"≥5: {row['frac_ge5']:.1%} | "
            f"median={row['median_clusters']:.1f}  p90={row['p90_clusters']:.1f}"
        )
        results.append(row)

    return results


def plot(results: list[dict]) -> None:
    """Save a two-panel plot of n-gram cross-cluster coverage vs. n-gram length."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ns = [r["n"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(ns, [r["frac_ge2"] for r in results], "o-", label="≥ 2 clusters")
    ax.plot(ns, [r["frac_ge3"] for r in results], "s-", label="≥ 3 clusters")
    ax.plot(ns, [r["frac_ge5"] for r in results], "^-", label="≥ 5 clusters")
    ax.set_xlabel("n-gram length")
    ax.set_ylabel("Fraction of unique n-grams")
    ax.set_title("Cross-cluster coverage vs. n-gram length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    ax = axes[1]
    ax.plot(ns, [r["median_clusters"] for r in results], "o-", label="Median")
    ax.plot(ns, [r["p90_clusters"] for r in results], "s--", label="90th pct")
    ax.set_xlabel("n-gram length")
    ax.set_ylabel("# clusters n-gram appears in")
    ax.set_title("Cluster spread distribution vs. n-gram length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)

    fig.tight_layout()
    out = RESULTS_DIR / "ngram_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    from build_graph import load as load_graph
    from cluster import load as load_clusters

    _, theorem_tactics = load_graph()
    cluster_map = load_clusters()
    results = sweep(theorem_tactics, cluster_map)

    out = RESULTS_DIR / "ngram_sweep.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sweep saved → {out}")
    plot(results)
