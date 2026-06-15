"""
e-0042: Do conceptual-bridge proof pairs share domain-specific MOTIFS
(multi-token tactic n-grams) even when their individual tokens are generic?

a-0036 (e-0039) measured genericity PER SINGLE TOKEN and found the conceptual
(edit-distance) detector's MIN-closest cross-cluster pairs share only ubiquitous
scaffolding tokens (rw[@], apply@, intro?, simp...): 1/200 pairs share a
domain-specific (<=3-cluster) token, at the random floor. a-0037 (e-0041) then
showed IDF-WEIGHTING the single tokens does not help either (3/200).

BUT a-0036 explicitly flagged the untested escape hatch:

    "genericity is measured per single token, so longer domain-specific tactic
     MOTIFS assembled from individually-generic tokens are not captured -- a
     natural next probe."

This is that probe. The hypothesis: a contiguous n-gram like
('apply ?', 'exact @', 'rw [@]') may be domain-SPECIFIC (appear in <=3 clusters)
as a WHOLE even though each of its tokens is ubiquitous. If conceptual-bridge
pairs share such emergent-specific motifs above the random-control floor, the
conceptual signal IS in tactic structure and the detector just needs motif
features. If they do not, a-0037's "no normalized-tactic-token detector recovers
conceptual bridges" extends to MOTIFS too, closing the tactic-representation path.

Method (CPU-only, pure stdlib; reuses e-0039 infrastructure):
  1. Normalize every proof (>=5 tactics).
  2. For n in {2,3}: contiguous n-gram cluster document-frequency
     df_n[ngram] = number of distinct Leiden clusters the n-gram appears in.
  3. For each conceptual-bridge pair and each random cross-cluster control pair
     (SAME top-30 x top-30-longest selection the detector used), find SHARED
     contiguous n-grams and measure:
       - n_specific_motif : shared n-grams with df_n <= 3 (domain-rare AS A WHOLE)
       - n_emergent_motif : the subset of those whose every constituent token is
                            GENERIC (df_1 >= 30) -- a-0036's exact hypothesis
  4. Compare bridges vs random. Floor-level => a-0037 extends to motifs.
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalize_tactics import normalize_sequence  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MIN_TACTICS = 5
TOP_PER_CLUSTER = 30
MAX_CLUSTERS = 30
NGRAM_NS = (2, 3)

GENERIC_CLUSTERS = 30      # token/motif in >=30 of ~138 clusters: ubiquitous
SPECIFIC_CLUSTERS = 3      # motif in <=3 clusters: domain-rare structure

_RNG_STATE = 0x2545F4914F6CDD1D


def _rand():
    """Deterministic xorshift64* -> float in [0,1)."""
    global _RNG_STATE
    x = _RNG_STATE
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    _RNG_STATE = x & 0xFFFFFFFFFFFFFFFF
    return ((_RNG_STATE * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) / 2**64


def load_inputs():
    with open(DATA_DIR / "theorem_tactics.json") as f:
        theorem_tactics = json.load(f)
    with open(DATA_DIR / "clusters.json") as f:
        cluster_map = {k: int(v) for k, v in json.load(f).items()}
    return theorem_tactics, cluster_map


def build_normalized(theorem_tactics):
    normed = {}
    for thm, tacs in theorem_tactics.items():
        seq = normalize_sequence([t.strip() for t in tacs])
        if len(seq) >= MIN_TACTICS:
            normed[thm] = seq
    return normed


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def token_cluster_df(normed, cluster_map):
    """Single-token genericity: token -> #distinct clusters (the e-0039 measure)."""
    tok_clusters = defaultdict(set)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is None:
            continue
        for tok in set(seq):
            tok_clusters[tok].add(cid)
    return {tok: len(cs) for tok, cs in tok_clusters.items()}


def ngram_cluster_df(normed, cluster_map, n):
    """Motif genericity: contiguous n-gram -> #distinct clusters it appears in."""
    ng_clusters = defaultdict(set)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is None:
            continue
        for g in set(ngrams(seq, n)):
            ng_clusters[g].add(cid)
    return {g: len(cs) for g, cs in ng_clusters.items()}


def shared_motifs(seq_a, seq_b, df_n, df_tok, n):
    """Shared contiguous n-grams between two proofs, classified by genericity.

    Returns counts of:
      n_shared      : distinct shared n-grams
      n_specific    : shared n-grams with df_n <= SPECIFIC_CLUSTERS (rare as a whole)
      n_emergent    : the subset of n_specific whose EVERY token is GENERIC
                      (df_tok >= GENERIC_CLUSTERS) -- a-0036's hypothesis
      specific_list : up to a few example specific motifs
    """
    sa, sb = set(ngrams(seq_a, n)), set(ngrams(seq_b, n))
    shared = sa & sb
    specific, emergent = [], []
    for g in shared:
        d = df_n.get(g, 0)
        if 0 < d <= SPECIFIC_CLUSTERS:
            specific.append(g)
            if all(df_tok.get(t, 0) >= GENERIC_CLUSTERS for t in g):
                emergent.append(g)
    return {
        "n_shared": len(shared),
        "n_specific": len(specific),
        "n_emergent": len(emergent),
        "has_specific": len(specific) > 0,
        "has_emergent": len(emergent) > 0,
        "specific_list": specific,
        "emergent_list": emergent,
    }


def summarize(stats, n_pairs):
    return {
        "n_pairs": n_pairs,
        "mean_n_shared": round(sum(s["n_shared"] for s in stats) / n_pairs, 4),
        "frac_with_specific_motif": round(sum(s["has_specific"] for s in stats) / n_pairs, 4),
        "frac_with_emergent_motif": round(sum(s["has_emergent"] for s in stats) / n_pairs, 4),
        "mean_n_specific_motif": round(sum(s["n_specific"] for s in stats) / n_pairs, 4),
        "mean_n_emergent_motif": round(sum(s["n_emergent"] for s in stats) / n_pairs, 4),
    }


def run_for_n(n, normed, cluster_map, df_tok, bridges, pool, cluster_pairs):
    df_n = ngram_cluster_df(normed, cluster_map, n)

    # conceptual-bridge pairs
    bridge_stats, missing = [], 0
    for b in bridges:
        a, bb = b["theorem_a"], b["theorem_b"]
        if a not in normed or bb not in normed:
            missing += 1
            continue
        s = shared_motifs(normed[a], normed[bb], df_n, df_tok, n)
        s["pair"] = (a, bb)
        s["norm_edit_dist"] = b["norm_edit_dist"]
        bridge_stats.append(s)

    # random cross-cluster control (same selection), matched draw count
    DRAWS_PER_PAIR = 20
    ctrl_stats = []
    for cid_a, cid_b in cluster_pairs:
        pa, pb = pool[cid_a], pool[cid_b]
        if not pa or not pb:
            continue
        for _ in range(DRAWS_PER_PAIR):
            _, sa = pa[int(_rand() * len(pa))]
            _, sb = pb[int(_rand() * len(pb))]
            ctrl_stats.append(shared_motifs(sa, sb, df_n, df_tok, n))

    bridge_summary = summarize(bridge_stats, len(bridge_stats))
    ctrl_summary = summarize(ctrl_stats, len(ctrl_stats))

    examples = sorted([s for s in bridge_stats if s["has_specific"]],
                      key=lambda s: (-s["n_specific"], s["norm_edit_dist"]))[:8]
    return df_n, bridge_summary, ctrl_summary, bridge_stats, examples, missing


def main():
    theorem_tactics, cluster_map = load_inputs()
    normed = build_normalized(theorem_tactics)
    df_tok = token_cluster_df(normed, cluster_map)
    print(f"{len(normed)} proofs (>={MIN_TACTICS} tactics); "
          f"{len(df_tok)} distinct tokens")

    with open(RESULTS_DIR / "edit_distance_bridges.json") as f:
        bridges = json.load(f)

    # same top-30 x top-30-longest selection the detector used
    by_cluster = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))
    target = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    pool = {c: sorted(by_cluster[c], key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]
            for c in target}
    cluster_pairs = list(combinations(target, 2))

    out = {
        "params": {
            "MIN_TACTICS": MIN_TACTICS, "TOP_PER_CLUSTER": TOP_PER_CLUSTER,
            "MAX_CLUSTERS": MAX_CLUSTERS, "NGRAM_NS": list(NGRAM_NS),
            "GENERIC_CLUSTERS": GENERIC_CLUSTERS, "SPECIFIC_CLUSTERS": SPECIFIC_CLUSTERS,
        },
        "by_n": {},
    }

    print("\n=== Shared MOTIF genericity: conceptual bridges vs random control ===")
    for n in NGRAM_NS:
        df_n, bsum, csum, bstats, examples, missing = run_for_n(
            n, normed, cluster_map, df_tok, bridges, pool, cluster_pairs)
        n_specific_motifs_total = sum(1 for d in df_n.values() if 0 < d <= SPECIFIC_CLUSTERS)
        out["by_n"][str(n)] = {
            "n_distinct_motifs": len(df_n),
            "n_domain_specific_motifs": n_specific_motifs_total,
            "conceptual_bridges": bsum,
            "random_control": csum,
            "n_bridge_pairs": len(bstats),
            "n_bridge_with_specific": sum(s["has_specific"] for s in bstats),
            "n_bridge_with_emergent": sum(s["has_emergent"] for s in bstats),
            "examples": [
                {"theorem_a": s["pair"][0], "theorem_b": s["pair"][1],
                 "norm_edit_dist": s["norm_edit_dist"],
                 "n_specific": s["n_specific"], "n_emergent": s["n_emergent"],
                 "specific_motifs": [list(g) for g in s["specific_list"][:5]]}
                for s in examples],
        }
        print(f"\n-- n={n}  ({len(df_n)} distinct motifs, "
              f"{n_specific_motifs_total} domain-specific) --")
        print(f"{'metric':<34}{'bridges':>12}{'random':>12}")
        for k in ["frac_with_specific_motif", "frac_with_emergent_motif",
                  "mean_n_specific_motif", "mean_n_emergent_motif", "mean_n_shared"]:
            print(f"{k:<34}{str(bsum[k]):>12}{str(csum[k]):>12}")
        print(f"bridge pairs sharing a domain-specific motif: "
              f"{sum(s['has_specific'] for s in bstats)}/{len(bstats)}  "
              f"(emergent-from-generic: {sum(s['has_emergent'] for s in bstats)})")
        if examples:
            print("  top specific-motif example:")
            e = examples[0]
            print(f"    ED={e['norm_edit_dist']:.3f} n_specific={e['n_specific']} "
                  f"{[list(g) for g in e['specific_list'][:3]]}")
            print(f"    {e['pair'][0]} <-> {e['pair'][1]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "conceptual_motif_bridges.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/conceptual_motif_bridges.json")
    return out


if __name__ == "__main__":
    main()
