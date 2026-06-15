"""
e-0040: Does a GENERICITY-WEIGHTED (IDF) conceptual-bridge detector key on
domain-specific shared structure where the plain edit-distance detector cannot?

a-0036 (e-0039) showed the plain conceptual detector (edit_distance_bridges:
the 200 MIN-closest cross-cluster proof pairs by normalized-tactic Levenshtein)
matches on GENERIC scaffolding: its top pairs share ubiquitous tokens
('rw [@]' in 38 clusters, 'apply @' in 36, 'constructor' in 34...), and only
1/200 pairs shares even one domain-specific (<=3-cluster) tactic token --
statistically indistinguishable from random cross-cluster pairs. a-0036 ends by
naming the lever: "a real conceptual-bridge detector must key on domain-specific
shared structure (IDF/TF-...)".

This builds that detector and tests it directly, cheaply, over the SAME data and
SAME top-30 x top-30-longest-proof selection the plain detector used:

  - Token weight w(t) = log(N_clusters / df_clusters(t)), df = number of distinct
    Leiden clusters a normalized token appears in (the e-0039 genericity measure).
    Ubiquitous scaffolding (high df) -> weight ~0; domain-rare tokens -> high weight.
  - Proof-pair similarity = IDF-WEIGHTED Jaccard on the token multisets:
        sum_shared  w(t) * min(count_a, count_b)
        ---------------------------------------------
        sum_union   w(t) * max(count_a, count_b)
    so a pair is "similar" only when it shares RARE (domain-specific) tokens, not
    when it shares generic scaffolding.
  - For each cluster pair, keep the (one-per-cluster) proof pair of MAX weighted
    similarity; take the top-200 pairs overall (matching the plain detector's 200).

Then re-run the e-0039 genericity probe on these IDF-detector pairs and compare
to (a) the plain edit-distance detector's pairs and (b) random cross-cluster
pairs. Key question: does keying on rarity RAISE frac_pairs_with_specific_shared
and mean_n_specific_shared above the plain detector's ~0.005 floor -- i.e. does a
genericity-weighted detector actually find conceptual bridges with genuine shared
domain-specific structure, or do cross-domain proofs share no domain-specific
tokens at all (in which case tactic-token conceptual bridges are fundamentally
unreachable)?

CPU-only, pure stdlib. Reuses data/theorem_tactics.json, data/clusters.json,
data/cluster_labels.json, results/edit_distance_bridges.json, normalize_tactics.
"""

import json
import math
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
TOP_K = 200

# Genericity cutoffs identical to e-0039 for a like-for-like comparison.
GENERIC_CLUSTERS = 30      # >=30 of the clusters: ubiquitous scaffolding
SPECIFIC_CLUSTERS = 3      # <=3 clusters: domain-rare structure


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
    """For each token: number of distinct clusters it appears in (over all >=5-tactic proofs)."""
    tok_clusters = defaultdict(set)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is None:
            continue
        for tok in set(seq):
            tok_clusters[tok].add(cid)
    return {tok: len(cs) for tok, cs in tok_clusters.items()}


def shared_genericity(seq_a, seq_b, df):
    """e-0039 genericity probe on the tokens shared by two proofs."""
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


def idf_weighted_jaccard(ca, cb, weight):
    """IDF-weighted Jaccard on two token multisets (Counters)."""
    keys = ca.keys() | cb.keys()
    num = 0.0
    den = 0.0
    for t in keys:
        w = weight.get(t, 0.0)
        if w == 0.0:
            continue
        a, b = ca.get(t, 0), cb.get(t, 0)
        num += w * min(a, b)
        den += w * max(a, b)
    return (num / den) if den > 0 else 0.0


def main():
    theorem_tactics, cluster_map, labels = load_inputs()
    normed = build_normalized(theorem_tactics)
    df = token_cluster_df(normed, cluster_map)
    n_clusters = len(set(cluster_map[t] for t in normed if t in cluster_map))
    # IDF weight: rare-in-clusters tokens dominate; ubiquitous tokens ~0.
    weight = {t: math.log(n_clusters / d) for t, d in df.items() if d > 0}
    print(f"{len(normed)} proofs (>={MIN_TACTICS} tactics); "
          f"{len(df)} distinct tokens; {n_clusters} clusters")

    # Same top-30 x top-30-longest selection as the plain detector / e-0039.
    by_cluster = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))
    target = sorted(by_cluster.keys(), key=lambda c: -len(by_cluster[c]))[:MAX_CLUSTERS]
    pool = {c: sorted(by_cluster[c], key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]
            for c in target}
    # Precompute counters once.
    counters = {c: [(thm, seq, Counter(seq)) for thm, seq in pool[c]] for c in target}

    # IDF-weighted MAX-similarity cross-cluster pair per cluster pair.
    results = []
    for cid_a, cid_b in combinations(target, 2):
        best_sim = -1.0
        best = None
        for ta, sa, cca in counters[cid_a]:
            for tb, sb, ccb in counters[cid_b]:
                sim = idf_weighted_jaccard(cca, ccb, weight)
                if sim > best_sim:
                    best_sim = sim
                    best = (ta, sa, tb, sb)
        if best is None or best_sim <= 0:
            continue
        ta, sa, tb, sb = best
        results.append({
            "cluster_a": cid_a, "cluster_b": cid_b,
            "label_a": labels.get(cid_a, {}).get("label", str(cid_a)),
            "label_b": labels.get(cid_b, {}).get("label", str(cid_b)),
            "theorem_a": ta, "theorem_b": tb,
            "idf_sim": round(best_sim, 4),
            "seq_a": sa, "seq_b": sb,
        })
    results.sort(key=lambda x: -x["idf_sim"])
    results = results[:TOP_K]
    print(f"IDF detector: {len(results)} top cross-cluster pairs")

    # Genericity probe on the IDF detector's pairs.
    idf_stats = []
    for r in results:
        s = shared_genericity(r["seq_a"], r["seq_b"], df)
        s["pair"] = (r["theorem_a"], r["theorem_b"])
        s["idf_sim"] = r["idf_sim"]
        s["labels"] = (r["label_a"], r["label_b"])
        idf_stats.append(s)
    idf_summary = summarize(idf_stats)

    # Reference: plain edit-distance detector's pairs (same probe).
    with open(RESULTS_DIR / "edit_distance_bridges.json") as f:
        plain = json.load(f)
    plain_stats = []
    for b in plain:
        a, bb = b["theorem_a"], b["theorem_b"]
        if a in normed and bb in normed:
            plain_stats.append(shared_genericity(normed[a], normed[bb], df))
    plain_summary = summarize(plain_stats)

    # Top IDF pairs that genuinely share domain-specific structure.
    specific_examples = sorted(
        [s for s in idf_stats if s["has_specific"]],
        key=lambda s: (-s["n_specific_shared"], -s["idf_sim"]))[:12]

    out = {
        "params": {
            "MIN_TACTICS": MIN_TACTICS, "TOP_PER_CLUSTER": TOP_PER_CLUSTER,
            "MAX_CLUSTERS": MAX_CLUSTERS, "TOP_K": TOP_K,
            "GENERIC_CLUSTERS": GENERIC_CLUSTERS, "SPECIFIC_CLUSTERS": SPECIFIC_CLUSTERS,
            "n_clusters_for_idf": n_clusters,
        },
        "idf_detector": idf_summary,
        "plain_edit_distance_detector": plain_summary,
        "n_idf_pairs_with_specific": sum(s["has_specific"] for s in idf_stats),
        "n_plain_pairs_with_specific": sum(
            shared_genericity(normed[b["theorem_a"]], normed[b["theorem_b"]], df)["has_specific"]
            for b in plain if b["theorem_a"] in normed and b["theorem_b"] in normed),
        "specific_shared_examples": [
            {"theorem_a": s["pair"][0], "theorem_b": s["pair"][1],
             "labels": s["labels"], "idf_sim": s["idf_sim"],
             "n_specific_shared": s["n_specific_shared"],
             "specific_tokens": s["specific_tokens"],
             "generic_frac": round(s["generic_frac"], 3) if s["generic_frac"] is not None else None}
            for s in specific_examples],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "conceptual_idf_bridges.json", "w") as f:
        json.dump(out, f, indent=2)

    # Report.
    print("\n=== Shared-structure genericity: IDF detector vs plain edit-distance ===")
    print(f"{'metric':<38}{'IDF':>12}{'plain ED':>12}")
    for k in ["frac_pairs_with_specific_shared", "mean_n_specific_shared",
              "mean_generic_frac", "median_generic_frac", "mean_shared_mass"]:
        print(f"{k:<38}{str(idf_summary[k]):>12}{str(plain_summary[k]):>12}")
    print(f"\nIDF pairs sharing >=1 domain-specific (<= {SPECIFIC_CLUSTERS}-cluster) token: "
          f"{out['n_idf_pairs_with_specific']}/{len(idf_stats)} "
          f"(plain: {out['n_plain_pairs_with_specific']}/{len(plain_stats)})")
    print("\nTop IDF pairs with genuine domain-specific shared structure:")
    for s in specific_examples[:10]:
        print(f"  sim={s['idf_sim']:.3f}  n_specific={s['n_specific_shared']}  "
              f"{s['labels'][0]} <-> {s['labels'][1]}")
        print(f"      tokens={s['specific_tokens']}")
        print(f"      {s['pair'][0]}  <->  {s['pair'][1]}")
    print("\nSaved -> results/conceptual_idf_bridges.json")
    return out


if __name__ == "__main__":
    main()
