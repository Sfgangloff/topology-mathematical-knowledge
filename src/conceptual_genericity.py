"""
e-0039: Is the conceptual-bridge detector finding GENERIC SCAFFOLDING or
genuine shared domain-specific structure?

a-0033 (e-0037) found the operational/conceptual bridge independence is specific
to the MIN (single-closest-pair) aggregation, and HYPOTHESIZED that the
MIN-closest cross-cluster proof pairs are "generic-scaffolding coincidences" --
two otherwise-unrelated domains that happen to share one freakishly-similar
proof built from ubiquitous tactic tokens (intro/simp/exact/rw...), rather than
the Brouwer<->Lawvere-style shared CONCEPTUAL structure a-0009 cares about.

This tests that hypothesis directly and cheaply, over existing data:

  1. Token cluster-frequency (genericity): for each normalized tactic token,
     count the number of distinct Leiden clusters in which it appears (over all
     normalized proofs of >=5 tactics). A token in many clusters is GENERIC
     scaffolding; a token in few clusters is domain-SPECIFIC.

  2. For each conceptual-bridge pair (theorem_a, theorem_b) in
     results/edit_distance_bridges.json (the detector's 200 MIN-closest
     cross-cluster pairs), compute the multiset of SHARED normalized tokens
     (tokens common to both proofs) and characterize their genericity:
       - generic_frac: fraction of shared-token MASS that is generic
       - n_specific_shared: count of shared tokens that are domain-specific
       - has_specific: whether the pair shares >=1 specific token

  3. CONTROL: random cross-cluster proof pairs from the SAME top-30 x top-30
     longest-proof selection the detector used. If the MIN-closest pairs are no
     more SPECIFIC in their shared structure than random cross-cluster pairs,
     a-0033's "generic-scaffolding coincidence" reading is confirmed: the
     detector's similarity is carried by ubiquitous scaffolding, not genuine
     shared conceptual structure.

CPU-only, pure stdlib (no numpy/networkx). Reuses data/theorem_tactics.json,
data/clusters.json, data/cluster_labels.json and the normalize_tactics module.
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

# A token is GENERIC if it appears in >= GENERIC_CLUSTERS distinct clusters,
# SPECIFIC if it appears in <= SPECIFIC_CLUSTERS distinct clusters.
GENERIC_CLUSTERS = 30      # present in >=30 of ~138 clusters: ubiquitous scaffolding
SPECIFIC_CLUSTERS = 3      # present in <=3 clusters: domain-rare structure

# Deterministic PRNG (no Math.random / external seed dependence)
_RNG_STATE = 0x2545F4914F6CDD1D


def _rand():
    """Deterministic xorshift64* -> float in [0,1). Pure, reproducible."""
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
    with open(DATA_DIR / "cluster_labels.json") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return theorem_tactics, cluster_map, labels


def build_normalized(theorem_tactics):
    normed = {}
    for thm, tacs in theorem_tactics.items():
        seq = normalize_sequence([t.strip() for t in tacs])
        if len(seq) >= MIN_TACTICS:
            normed[thm] = seq
    return normed


def token_cluster_df(normed, cluster_map):
    """For each token: set of distinct clusters it appears in (over ALL >=5-tactic proofs)."""
    tok_clusters = defaultdict(set)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is None:
            continue
        for tok in set(seq):
            tok_clusters[tok].add(cid)
    return {tok: len(cs) for tok, cs in tok_clusters.items()}


def shared_genericity(seq_a, seq_b, df):
    """Characterize the genericity of the tokens shared by two proofs.

    Shared mass = sum over tokens of min(count_a, count_b) (multiset intersection).
    """
    ca, cb = Counter(seq_a), Counter(seq_b)
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
        "mean_shared_mass": round(sum(s["shared_mass"] for s in stats) / n, 3),
        "mean_generic_frac": round(sum(gfracs) / len(gfracs), 4) if gfracs else None,
        "median_generic_frac": round(sorted(gfracs)[len(gfracs) // 2], 4) if gfracs else None,
        "frac_pairs_with_specific_shared": round(
            sum(s["has_specific"] for s in stats) / n, 4),
        "mean_n_specific_shared": round(
            sum(s["n_specific_shared"] for s in stats) / n, 4),
    }


def main():
    theorem_tactics, cluster_map, labels = load_inputs()
    normed = build_normalized(theorem_tactics)
    df = token_cluster_df(normed, cluster_map)
    print(f"{len(normed)} proofs (>={MIN_TACTICS} tactics); "
          f"{len(df)} distinct normalized tokens")

    # genericity landscape
    most_generic = sorted(df.items(), key=lambda x: -x[1])[:15]
    print("\nMost generic tokens (appear in N clusters):")
    for tok, c in most_generic:
        print(f"  {c:>3}  {tok!r}")

    # ---- conceptual-bridge pairs (detector output) ----
    with open(RESULTS_DIR / "edit_distance_bridges.json") as f:
        bridges = json.load(f)
    bridge_stats = []
    missing = 0
    for b in bridges:
        a, bb = b["theorem_a"], b["theorem_b"]
        if a not in normed or bb not in normed:
            missing += 1
            continue
        s = shared_genericity(normed[a], normed[bb], df)
        s["pair"] = (a, bb)
        s["norm_edit_dist"] = b["norm_edit_dist"]
        bridge_stats.append(s)
    bridge_summary = summarize(bridge_stats)
    print(f"\nConceptual-bridge pairs: {len(bridge_stats)} "
          f"({missing} dropped for missing normalized proof)")

    # ---- control: random cross-cluster pairs from the SAME selection ----
    by_cluster = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))
    target = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    pool = {c: sorted(by_cluster[c], key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]
            for c in target}
    cluster_pairs = list(combinations(target, 2))

    # For each of the same number of cluster pairs the detector reported on, draw
    # several random (a in cluster_a, b in cluster_b) proof pairs and average.
    DRAWS_PER_PAIR = 20
    ctrl_stats = []
    for cid_a, cid_b in cluster_pairs:
        pa, pb = pool[cid_a], pool[cid_b]
        if not pa or not pb:
            continue
        for _ in range(DRAWS_PER_PAIR):
            ta, sa = pa[int(_rand() * len(pa))]
            tb, sb = pb[int(_rand() * len(pb))]
            ctrl_stats.append(shared_genericity(sa, sb, df))
    ctrl_summary = summarize(ctrl_stats)

    # ---- how many bridges carry GENUINE specific structure ----
    specific_examples = sorted(
        [s for s in bridge_stats if s["has_specific"]],
        key=lambda s: (-s["n_specific_shared"], s["norm_edit_dist"]))[:10]

    out = {
        "params": {
            "MIN_TACTICS": MIN_TACTICS,
            "TOP_PER_CLUSTER": TOP_PER_CLUSTER,
            "MAX_CLUSTERS": MAX_CLUSTERS,
            "GENERIC_CLUSTERS": GENERIC_CLUSTERS,
            "SPECIFIC_CLUSTERS": SPECIFIC_CLUSTERS,
            "n_total_clusters_with_tokens": len(set(
                cluster_map[t] for t in normed if t in cluster_map)),
        },
        "most_generic_tokens": [{"token": t, "n_clusters": c} for t, c in most_generic],
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
    with open(RESULTS_DIR / "conceptual_genericity.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- report ----
    print("\n=== Shared-structure genericity: conceptual bridges vs random control ===")
    print(f"{'metric':<38}{'bridges':>12}{'random':>12}")
    for k in ["mean_generic_frac", "median_generic_frac",
              "frac_pairs_with_specific_shared", "mean_n_specific_shared",
              "mean_shared_mass"]:
        print(f"{k:<38}{str(bridge_summary[k]):>12}{str(ctrl_summary[k]):>12}")
    print(f"\nConceptual-bridge pairs sharing >=1 domain-specific token: "
          f"{sum(s['has_specific'] for s in bridge_stats)}/{len(bridge_stats)}")
    print("\nTop pairs with genuine specific shared structure:")
    for s in specific_examples[:8]:
        print(f"  ED={s['norm_edit_dist']:.3f}  n_specific={s['n_specific_shared']}  "
              f"{s['specific_tokens']}")
        print(f"      {s['pair'][0]}  <->  {s['pair'][1]}")
    print("\nSaved -> results/conceptual_genericity.json")
    return out


if __name__ == "__main__":
    main()
