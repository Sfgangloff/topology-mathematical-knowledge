"""
e-0043 (q-0009): Do the conceptual-bridge proof pairs share domain-specific
NAME-CONTENT tokens -- the one content-bearing field the corpus DOES expose?

The a-0036/a-0037/a-0038 chain established that the edit-distance conceptual-bridge
detector's 200 MIN-closest cross-cluster proof pairs share only ubiquitous TACTIC
scaffolding -- no domain-specific shared tactic token (a-0037) or multi-token motif
(a-0038) above the random floor -- and concluded the conceptual-bridge signal "lives
in statement/type CONTENT, which the corpus does not expose."

But the corpus DOES expose one content-bearing field: the declaration's full_name.
Mathlib names are richly descriptive concept strings (Cardinal.sum_lt_prod,
PreGaloisCategory.has_decomp_connected_components) -- a partial proxy for statement
content. The retrieval chain (a-0023..a-0033) used name tokens for cluster ROUTING /
intra-cluster RANKING, but the conceptual-bridge DETECTION thread only ever looked at
tactic tokens. This closes that gap, mirroring conceptual_genericity.py (e-0039)
EXACTLY but swapping the tactic-token bag for a NAME-token bag.

Question: for the detector's 200 MIN-closest cross-cluster pairs, do the two theorems'
NAMES share domain-specific (low-cluster-frequency) concept tokens more than random
cross-cluster pairs from the same selection?

  - If YES: the corpus exposes a recoverable conceptual-bridge signal after all, in
    names rather than tactics -- refining a-0037/a-0038 (the limit is TYPES, not all
    content) and reopening a name-content conceptual detector.
  - If NO: even names carry no shared conceptual content for these tactic-similarity-
    selected pairs -- strengthening the corpus/representation-limit conclusion.

CPU-only, pure stdlib. Reuses data/theorem_tactics.json, data/clusters.json,
results/edit_distance_bridges.json and the normalize_tactics module (only to
reproduce the detector's >=5-tactic / top-30 x top-30-longest selection).
"""

import json
import re
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

# A NAME token is GENERIC if it appears in >= GENERIC_CLUSTERS distinct clusters,
# SPECIFIC if it appears in <= SPECIFIC_CLUSTERS distinct clusters. Same thresholds
# as the tactic-token probe (e-0039) so the comparison is apples-to-apples.
GENERIC_CLUSTERS = 30
SPECIFIC_CLUSTERS = 3

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def name_tokens(full_name):
    """Tokenize a Mathlib full_name into lowercase concept tokens.

    Split on '.'/'_'/non-alphanumerics, then split camelCase, lowercase, drop
    pure-numeric tokens and 1-char tokens. e.g.
      Complex.tendsto_tsum_powerSeries_nhdsWithin_stolzSet ->
      [complex, tendsto, tsum, power, series, nhds, within, stolz, set]
    """
    toks = []
    for part in re.split(r"[^A-Za-z0-9]+", full_name):
        for m in _CAMEL.findall(part):
            t = m.lower()
            if len(t) > 1 and not t.isdigit():
                toks.append(t)
    return toks


def load_inputs():
    with open(DATA_DIR / "theorem_tactics.json") as f:
        theorem_tactics = json.load(f)
    with open(DATA_DIR / "clusters.json") as f:
        cluster_map = {k: int(v) for k, v in json.load(f).items()}
    return theorem_tactics, cluster_map


def build_selection(theorem_tactics, cluster_map):
    """Reproduce the detector's selection: clustered theorems with >=5 normalized
    tactics. Returns {thm: name_token_list} over that population."""
    sel = {}
    for thm, tacs in theorem_tactics.items():
        if thm not in cluster_map:
            continue
        seq = normalize_sequence([t.strip() for t in tacs])
        if len(seq) >= MIN_TACTICS:
            sel[thm] = name_tokens(thm)
    return sel


def token_cluster_df(sel, cluster_map):
    """For each NAME token: number of distinct clusters it appears in (over the
    same >=5-tactic clustered population the detector drew from)."""
    tok_clusters = defaultdict(set)
    for thm, toks in sel.items():
        cid = cluster_map[thm]
        for tok in set(toks):
            tok_clusters[tok].add(cid)
    return {tok: len(cs) for tok, cs in tok_clusters.items()}


def shared_genericity(toks_a, toks_b, df):
    ca, cb = Counter(toks_a), Counter(toks_b)
    shared = {t: min(ca[t], cb[t]) for t in ca.keys() & cb.keys()}
    total = sum(shared.values())
    if total == 0:
        return {"shared_mass": 0, "generic_frac": None,
                "n_specific_shared": 0, "has_specific": False,
                "specific_tokens": []}
    generic_mass = sum(m for t, m in shared.items()
                       if df.get(t, 0) >= GENERIC_CLUSTERS)
    specific = [t for t in shared if 0 < df.get(t, 0) <= SPECIFIC_CLUSTERS]
    return {
        "shared_mass": total,
        "generic_frac": generic_mass / total,
        "n_specific_shared": len(specific),
        "has_specific": len(specific) > 0,
        "specific_tokens": specific,
    }


def summarize(stats):
    n = len(stats)
    with_share = [s for s in stats if s["shared_mass"] > 0]
    gfracs = [s["generic_frac"] for s in with_share]
    return {
        "n_pairs": n,
        "n_with_shared_tokens": len(with_share),
        "frac_pairs_with_shared": round(len(with_share) / n, 4),
        "mean_shared_mass": round(sum(s["shared_mass"] for s in stats) / n, 4),
        "mean_generic_frac": round(sum(gfracs) / len(gfracs), 4) if gfracs else None,
        "median_generic_frac": round(sorted(gfracs)[len(gfracs) // 2], 4) if gfracs else None,
        "frac_pairs_with_specific_shared": round(
            sum(s["has_specific"] for s in stats) / n, 4),
        "mean_n_specific_shared": round(
            sum(s["n_specific_shared"] for s in stats) / n, 4),
    }


# Deterministic xorshift64* PRNG (no Math.random / external seed dependence)
_RNG_STATE = 0x2545F4914F6CDD1D


def _rand():
    global _RNG_STATE
    x = _RNG_STATE
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    _RNG_STATE = x & 0xFFFFFFFFFFFFFFFF
    return ((_RNG_STATE * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) / 2**64


def main():
    theorem_tactics, cluster_map = load_inputs()
    sel = build_selection(theorem_tactics, cluster_map)
    df = token_cluster_df(sel, cluster_map)
    n_clusters = len(set(cluster_map[t] for t in sel))
    print(f"{len(sel)} proofs (>={MIN_TACTICS} tactics) over {n_clusters} clusters; "
          f"{len(df)} distinct NAME tokens")

    most_generic = sorted(df.items(), key=lambda x: -x[1])[:15]
    print("\nMost generic NAME tokens (appear in N clusters):")
    for tok, c in most_generic:
        print(f"  {c:>3}  {tok!r}")

    # ---- conceptual-bridge pairs (detector output) ----
    with open(RESULTS_DIR / "edit_distance_bridges.json") as f:
        bridges = json.load(f)
    bridge_stats = []
    missing = 0
    for b in bridges:
        a, bb = b["theorem_a"], b["theorem_b"]
        if a not in sel or bb not in sel:
            missing += 1
            continue
        s = shared_genericity(sel[a], sel[bb], df)
        s["pair"] = (a, bb)
        s["norm_edit_dist"] = b["norm_edit_dist"]
        bridge_stats.append(s)
    bridge_summary = summarize(bridge_stats)
    print(f"\nConceptual-bridge pairs: {len(bridge_stats)} "
          f"({missing} dropped for missing selection)")

    # ---- control: random cross-cluster pairs from the SAME selection ----
    by_cluster = defaultdict(list)
    for thm, toks in sel.items():
        by_cluster[cluster_map[thm]].append((thm, toks))
    # rank proofs by tactic length to match the detector's "top-30 longest" pool
    tac_len = {}
    for thm in sel:
        tac_len[thm] = len(theorem_tactics[thm])
    target = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    pool = {c: sorted(by_cluster[c], key=lambda x: -tac_len[x[0]])[:TOP_PER_CLUSTER]
            for c in target}
    cluster_pairs = list(combinations(target, 2))

    DRAWS_PER_PAIR = 20
    ctrl_stats = []
    for cid_a, cid_b in cluster_pairs:
        pa, pb = pool[cid_a], pool[cid_b]
        if not pa or not pb:
            continue
        for _ in range(DRAWS_PER_PAIR):
            _, ta = pa[int(_rand() * len(pa))]
            _, tb = pb[int(_rand() * len(pb))]
            ctrl_stats.append(shared_genericity(ta, tb, df))
    ctrl_summary = summarize(ctrl_stats)

    specific_examples = sorted(
        [s for s in bridge_stats if s["has_specific"]],
        key=lambda s: (-s["n_specific_shared"], s["norm_edit_dist"]))[:12]

    out = {
        "params": {
            "MIN_TACTICS": MIN_TACTICS, "TOP_PER_CLUSTER": TOP_PER_CLUSTER,
            "MAX_CLUSTERS": MAX_CLUSTERS, "GENERIC_CLUSTERS": GENERIC_CLUSTERS,
            "SPECIFIC_CLUSTERS": SPECIFIC_CLUSTERS,
            "n_total_clusters_with_tokens": n_clusters,
            "DRAWS_PER_PAIR": DRAWS_PER_PAIR,
        },
        "most_generic_name_tokens": [{"token": t, "n_clusters": c} for t, c in most_generic],
        "conceptual_bridges": bridge_summary,
        "random_cross_cluster_control": ctrl_summary,
        "specific_shared_examples": [
            {"theorem_a": s["pair"][0], "theorem_b": s["pair"][1],
             "norm_edit_dist": s["norm_edit_dist"],
             "n_specific_shared": s["n_specific_shared"],
             "specific_tokens": s["specific_tokens"],
             "generic_frac": round(s["generic_frac"], 3) if s["generic_frac"] is not None else None}
            for s in specific_examples],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "conceptual_name_bridges.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Shared NAME-content genericity: conceptual bridges vs random control ===")
    print(f"{'metric':<38}{'bridges':>12}{'random':>12}")
    for k in ["frac_pairs_with_shared", "mean_shared_mass", "mean_generic_frac",
              "median_generic_frac", "frac_pairs_with_specific_shared",
              "mean_n_specific_shared"]:
        print(f"{k:<38}{str(bridge_summary[k]):>12}{str(ctrl_summary[k]):>12}")
    print(f"\nConceptual-bridge pairs sharing >=1 domain-specific NAME token: "
          f"{sum(s['has_specific'] for s in bridge_stats)}/{len(bridge_stats)} "
          f"(control rate {ctrl_summary['frac_pairs_with_specific_shared']})")
    print("\nTop pairs with genuine domain-specific shared NAME tokens:")
    for s in specific_examples[:10]:
        print(f"  ED={s['norm_edit_dist']:.3f}  n_specific={s['n_specific_shared']}  "
              f"{s['specific_tokens']}")
        print(f"      {s['pair'][0]}  <->  {s['pair'][1]}")
    print("\nSaved -> results/conceptual_name_bridges.json")
    return out


if __name__ == "__main__":
    main()
