"""
e-0046: Is the conceptual-bridge NEGATIVE result a SAMPLING artifact?

Every probe in the conceptual-bridge thread (a-0009, a-0032, a-0034, a-0036,
a-0038, a-0039, a-0041) runs on the SAME biased selection the edit-distance
detector uses: the top-30 LARGEST clusters and the top-30 LONGEST proofs per
cluster. Each of those answers explicitly carries the caveat "same
top-30-cluster / top-30-longest-proof high-similarity sample as the detector" --
and none has ever varied it. So the whole "conceptual bridges are undetectable
from any in-corpus representation" conclusion could, in principle, be an
artifact of restricting attention to long proofs from big domains: maybe the
genuine domain-specific shared structure lives in SHORTER proofs or SMALLER
domains and the sample simply never looks there.

Only ~25 clusters carry >=30 qualifying proofs, so the "top-30 LARGEST clusters"
half of the caveat is nearly moot -- the detector already uses essentially every
usable cluster. The live axis is PROOF SELECTION (top-30 LONGEST). This closes
the caveat directly and cheaply by re-running the core a-0036 genericity test --
do the MIN-closest cross-cluster proof pairs share DOMAIN-SPECIFIC tactic
tokens, or only ubiquitous scaffolding? -- under five complementary SAMPLE
regimes (same eligible clusters, >=30 proofs each):

  repro          all eligible clusters, top-30 LONGEST proofs   (= a-0036 baseline)
  small_clusters smaller half of eligible clusters, top-30 longest proofs
  median_proofs  all eligible clusters, 30 proofs around the MEDIAN length
  short_proofs   all eligible clusters, 30 SHORTEST (>=5-tactic) proofs
  random_proofs  all eligible clusters, 30 RANDOM proofs each

Token genericity (the number of distinct Leiden clusters a token appears in) is
measured GLOBALLY over all >=5-tactic proofs in every regime -- only the
detector's SELECTION changes, so the comparison is apples-to-apples with a-0036.

If frac_pairs_with_specific_shared stays ~0 (and generic_frac stays high) across
all four regimes, the negative result is robust to the sampling caveat and the
corpus/representation-limit conclusion (a-0041) is hardened. If any regime flips
(bridges share real domain-specific structure), the whole negative chain was a
sampling artifact.

CPU-only, pure stdlib. Reuses data/theorem_tactics.json, data/clusters.json,
data/cluster_labels.json and the normalize_tactics module.
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
ELIGIBLE_MIN = 30      # cluster needs this many qualifying proofs to enter any pool
TOP_K = 200            # MIN-closest cross-cluster pairs to keep (matches detector)
ED_CAP = 80           # cap proof length for edit distance (matches detector)

GENERIC_CLUSTERS = 30  # token in >=30 of ~138 clusters: ubiquitous scaffolding
SPECIFIC_CLUSTERS = 3  # token in <=3 clusters: domain-rare structure

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


def _shuffle(lst):
    """Deterministic Fisher-Yates using _rand()."""
    a = list(lst)
    for i in range(len(a) - 1, 0, -1):
        j = int(_rand() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def _edit_distance(a, b):
    m, n = len(a), len(b)
    if m > ED_CAP or n > ED_CAP:
        a, b = a[:ED_CAP], b[:ED_CAP]
        m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ai == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def _norm_ed(a, b):
    return _edit_distance(a, b) / max(len(a), len(b), 1)


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
    """Token -> number of distinct clusters it appears in (GLOBAL, all regimes)."""
    tok_clusters = defaultdict(set)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is None:
            continue
        for tok in set(seq):
            tok_clusters[tok].add(cid)
    return {tok: len(cs) for tok, cs in tok_clusters.items()}


def shared_genericity(seq_a, seq_b, df):
    ca, cb = Counter(seq_a), Counter(seq_b)
    shared = {t: min(ca[t], cb[t]) for t in ca.keys() & cb.keys()}
    total = sum(shared.values())
    if total == 0:
        return {"shared_mass": 0, "generic_frac": None,
                "n_specific_shared": 0, "has_specific": False}
    generic_mass = sum(m for t, m in shared.items()
                       if df.get(t, 0) >= GENERIC_CLUSTERS)
    specific = [t for t in shared if 0 < df.get(t, 0) <= SPECIFIC_CLUSTERS]
    return {"shared_mass": total, "generic_frac": generic_mass / total,
            "n_specific_shared": len(specific), "has_specific": len(specific) > 0}


def summarize(stats):
    n = len(stats)
    if n == 0:
        return {"n_pairs": 0}
    gfracs = [s["generic_frac"] for s in stats if s["shared_mass"] > 0]
    return {
        "n_pairs": n,
        "n_with_shared_tokens": sum(1 for s in stats if s["shared_mass"] > 0),
        "mean_shared_mass": round(sum(s["shared_mass"] for s in stats) / n, 3),
        "mean_generic_frac": round(sum(gfracs) / len(gfracs), 4) if gfracs else None,
        "frac_pairs_with_specific_shared":
            round(sum(s["has_specific"] for s in stats) / n, 4),
        "mean_n_specific_shared":
            round(sum(s["n_specific_shared"] for s in stats) / n, 4),
    }


def select_pool(by_cluster, regime):
    """Return {cluster_id: [(thm, seq), ...]} for the given sample regime.

    Cluster set is fixed across the proof-selection regimes (all eligible
    clusters, >=ELIGIBLE_MIN proofs) so only the per-cluster proof SELECTION
    varies; small_clusters additionally restricts the cluster set."""
    eligible = sorted([c for c in by_cluster if len(by_cluster[c]) >= ELIGIBLE_MIN],
                      key=lambda c: -len(by_cluster[c]))

    longest = lambda p: sorted(p, key=lambda x: -len(x[1]))[:TOP_PER_CLUSTER]
    shortest = lambda p: sorted(p, key=lambda x: len(x[1]))[:TOP_PER_CLUSTER]

    def median(p):
        s = sorted(p, key=lambda x: len(x[1]))
        mid, half = len(s) // 2, TOP_PER_CLUSTER // 2
        lo = max(0, mid - half)
        return s[lo:lo + TOP_PER_CLUSTER]

    if regime == "repro":
        clusters, pick = eligible, longest
    elif regime == "small_clusters":
        clusters, pick = eligible[len(eligible) // 2:], longest
    elif regime == "median_proofs":
        clusters, pick = eligible, median
    elif regime == "short_proofs":
        clusters, pick = eligible, shortest
    elif regime == "random_proofs":
        clusters = eligible
        pick = lambda p: _shuffle(p)[:TOP_PER_CLUSTER]
    else:
        raise ValueError(regime)

    return {c: pick(by_cluster[c]) for c in clusters}


def min_closest_pairs(pool):
    """For each cluster pair, the single MIN-edit-distance cross-cluster proof
    pair; return the global TOP_K closest (the detector's output)."""
    results = []
    for ca, cb in combinations(sorted(pool.keys()), 2):
        pa, pb = pool[ca], pool[cb]
        best_ed, best = 1.0, None
        for ta, sa in pa:
            for tb, sb in pb:
                ed = _norm_ed(sa, sb)
                if ed < best_ed:
                    best_ed, best = ed, (ta, tb)
        if best is not None:
            results.append((best_ed, best[0], best[1]))
    results.sort(key=lambda x: x[0])
    return results[:TOP_K]


def run_regime(regime, by_cluster, df):
    pool = select_pool(by_cluster, regime)
    bridges = min_closest_pairs(pool)
    bstats = [shared_genericity(_seq[ta], _seq[tb], df) for _, ta, tb in bridges]

    # random cross-cluster control from the SAME pool
    cps = list(combinations(sorted(pool.keys()), 2))
    ctrl = []
    for ca, cb in cps:
        pa, pb = pool[ca], pool[cb]
        if not pa or not pb:
            continue
        for _ in range(20):
            ta, sa = pa[int(_rand() * len(pa))]
            tb, sb = pb[int(_rand() * len(pb))]
            ctrl.append(shared_genericity(sa, sb, df))

    return {
        "regime": regime,
        "n_clusters": len(pool),
        "mean_proof_len": round(
            sum(len(s) for ps in pool.values() for _, s in ps) /
            max(1, sum(len(ps) for ps in pool.values())), 2),
        "min_ed_of_closest": round(bridges[0][0], 4) if bridges else None,
        "max_ed_of_closest": round(bridges[-1][0], 4) if bridges else None,
        "bridges": summarize(bstats),
        "control": summarize(ctrl),
    }


_seq = {}  # populated in main; module-level so run_regime can index by theorem id


def main():
    global _seq
    theorem_tactics, cluster_map, labels = load_inputs()
    normed = build_normalized(theorem_tactics)
    _seq = normed
    df = token_cluster_df(normed, cluster_map)
    n_total_clusters = len(set(cluster_map[t] for t in normed if t in cluster_map))
    print(f"{len(normed)} proofs (>={MIN_TACTICS} tactics); "
          f"{len(df)} tokens; {n_total_clusters} clusters with proofs")

    by_cluster = defaultdict(list)
    for thm, seq in normed.items():
        cid = cluster_map.get(thm)
        if cid is not None:
            by_cluster[cid].append((thm, seq))

    regimes = ["repro", "small_clusters", "median_proofs", "short_proofs",
               "random_proofs"]
    out = {"params": {"MIN_TACTICS": MIN_TACTICS, "TOP_PER_CLUSTER": TOP_PER_CLUSTER,
                      "ELIGIBLE_MIN": ELIGIBLE_MIN, "TOP_K": TOP_K,
                      "GENERIC_CLUSTERS": GENERIC_CLUSTERS,
                      "SPECIFIC_CLUSTERS": SPECIFIC_CLUSTERS,
                      "n_total_clusters": n_total_clusters},
           "regimes": []}

    print(f"\n{'regime':<16}{'nclu':>5}{'avglen':>8}{'edrange':>14}"
          f"{'B:spec%':>9}{'C:spec%':>9}{'B:gen_frac':>11}{'C:gen_frac':>11}")
    print("-" * 92)
    for r in regimes:
        res = run_regime(r, by_cluster, df)
        out["regimes"].append(res)
        b, c = res["bridges"], res["control"]
        edr = f"{res['min_ed_of_closest']}-{res['max_ed_of_closest']}"
        print(f"{r:<16}{res['n_clusters']:>5}{res['mean_proof_len']:>8}{edr:>14}"
              f"{b.get('frac_pairs_with_specific_shared', 0):>9}"
              f"{c.get('frac_pairs_with_specific_shared', 0):>9}"
              f"{str(b.get('mean_generic_frac')):>11}{str(c.get('mean_generic_frac')):>11}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "conceptual_sample_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/conceptual_sample_robustness.json")
    print("\nB:spec% / C:spec% = fraction of bridge / control pairs sharing >=1 "
          "domain-specific (<=3-cluster) token.")
    print("gen_frac = mean fraction of shared-token mass that is GENERIC "
          "(>=30-cluster scaffolding).")
    return out


if __name__ == "__main__":
    main()
