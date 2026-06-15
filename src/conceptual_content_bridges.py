"""
e-0045 (q-0009): Do the conceptual-bridge proof pairs share domain-specific
RAW-CONTENT tokens -- the mathematical identifiers manipulated INSIDE the tactic
expressions, the richest content representation the corpus actually exposes?

The a-0036/a-0037/a-0038/a-0039 chain ruled out, for the edit-distance detector's
200 MIN-closest cross-cluster proof pairs, every representation it tried:
  - normalized tactic tokens (e-0039) -- content stripped (names -> '@')
  - IDF-weighted normalized tokens (e-0041)
  - multi-token normalized motifs (e-0042)
  - theorem full_name tokens (e-0043)
None share domain-specific structure above the random floor, and the thread concluded
"the conceptual-bridge signal lives in statement/type CONTENT, which the corpus does
not expose."

But there is ONE richer content field never tested: the RAW (un-normalized) tactic
strings themselves contain the actual mathematical identifiers manipulated mid-proof
-- `IsIntegralClosure`, `FractionRing`, `Algebra.algebraMapSubmonoid`, `injective` --
the type constructors, structures and lemmas the proof actually touches. e-0039..e-0042
DELIBERATELY normalize these away (the whole point of normalize_tactics is to compare
STRATEGY independent of content); e-0043 used only the declaration's own name. Nobody
has asked whether the tactic-structure-selected bridge pairs share the CONCEPT
vocabulary embedded in their proof text.

This mirrors conceptual_name_bridges.py (e-0043) EXACTLY but swaps the name-token bag
for a RAW-CONTENT-token bag extracted from each theorem's tactic expressions (qualified
/ CamelCase identifiers, tokenized like names).

Question: for the detector's 200 MIN-closest cross-cluster pairs, do the two proofs
share domain-specific (low-cluster-frequency) raw mathematical-content tokens more than
random cross-cluster pairs from the same selection?

  - If YES: the corpus DOES expose a recoverable conceptual-bridge signal after all,
    in raw proof content rather than tactics/names -- reopening conceptual detection
    and refining a-0037..a-0039 (the limit is the NORMALIZATION, not the corpus).
  - If NO: even the raw mathematical vocabulary the proofs manipulate carries no shared
    domain-specific content for these tactic-similarity-selected pairs -- the strongest
    possible in-corpus confirmation that the conceptual-bridge gap is a genuine
    corpus/representation limit (needs statement/type structure, absent here).

NOTE: raw content tokens overlap with PREMISES (the operational signal). a-0032 showed
operational and conceptual (tactic-structure) bridges are near-independent, so the
PREDICTION under a-0032 is NO shared content -- this experiment tests that prediction
with the broadest available content bag (premises + local types + notation, not just
the premise names).

CPU-only, pure stdlib. Reuses data/theorem_tactics.json, data/clusters.json,
results/edit_distance_bridges.json and normalize_tactics (only to reproduce the
detector's >=5-tactic / top-30 x top-30-longest selection).
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

# Same genericity thresholds as the tactic-token (e-0039) and name-token (e-0043)
# probes so the comparison is apples-to-apples.
GENERIC_CLUSTERS = 30
SPECIFIC_CLUSTERS = 3

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# A CONTENT identifier in a raw tactic string: a qualified name (has a '.') OR a
# CamelCase / capitalised token. This skips lowercase tactic keywords (rw, exact,
# simp, apply, intro, ...) and short local variables, keeping the mathematical
# objects the proof manipulates. We then tokenize each identifier like a name.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*")

# Lowercase tactic keywords / common combinators that the _IDENT pass would catch
# only if they appeared qualified; harmless but excluded for cleanliness. The
# capitalised-or-qualified gate already removes the bulk of tactic syntax.
_TACTIC_WORDS = {
    "rw", "simp", "exact", "apply", "intro", "intros", "refine", "have", "show",
    "rfl", "cases", "rcases", "obtain", "use", "constructor", "ext", "by",
    "at", "with", "fun", "let", "this", "fun_prop", "norm_num", "ring", "omega",
    "linarith", "nlinarith", "positivity", "gcongr", "convert", "calc", "set",
    "induction", "rintro", "ringnf", "field_simp", "push_cast", "exacts",
}


def is_content_ident(ident):
    """True for a qualified or Capitalised identifier (a math object), excluding
    pure tactic keywords."""
    if "." in ident:
        return True
    if ident and ident[0].isupper():
        return True
    return False


def content_tokens(tactics):
    """Extract a bag of lowercase content tokens from the RAW tactic strings of one
    theorem: qualified/CamelCase identifiers tokenized like names.

      'apply IsFractionRing.injective A (FractionRing A)' ->
        [is, fraction, ring, injective, fraction, ring]
    """
    toks = []
    for tac in tactics:
        for ident in _IDENT.findall(tac):
            if not is_content_ident(ident):
                continue
            # tokenize the identifier (split on '.'/'_'/camelCase), like name_tokens
            for part in re.split(r"[^A-Za-z0-9]+", ident):
                for m in _CAMEL.findall(part):
                    t = m.lower()
                    if len(t) > 1 and not t.isdigit() and t not in _TACTIC_WORDS:
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
    tactics. Returns {thm: content_token_list} over that population."""
    sel = {}
    for thm, tacs in theorem_tactics.items():
        if thm not in cluster_map:
            continue
        raw = [t.strip() for t in tacs]
        seq = normalize_sequence(raw)
        if len(seq) >= MIN_TACTICS:
            sel[thm] = content_tokens(raw)
    return sel


def token_cluster_df(sel, cluster_map):
    """For each CONTENT token: number of distinct clusters it appears in (over the
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


# Deterministic xorshift64* PRNG (no Math.random / external seed dependence),
# identical to conceptual_name_bridges.py so the control draw is reproducible.
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
    mean_bag = sum(len(v) for v in sel.values()) / len(sel)
    print(f"{len(sel)} proofs (>={MIN_TACTICS} tactics) over {n_clusters} clusters; "
          f"{len(df)} distinct CONTENT tokens; mean bag size {mean_bag:.1f}")

    most_generic = sorted(df.items(), key=lambda x: -x[1])[:15]
    print("\nMost generic CONTENT tokens (appear in N clusters):")
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
    tac_len = {thm: len(theorem_tactics[thm]) for thm in sel}
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
            "mean_content_bag_size": round(mean_bag, 2),
            "DRAWS_PER_PAIR": DRAWS_PER_PAIR,
        },
        "most_generic_content_tokens": [{"token": t, "n_clusters": c} for t, c in most_generic],
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
    with open(RESULTS_DIR / "conceptual_content_bridges.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Shared RAW-CONTENT genericity: conceptual bridges vs random control ===")
    print(f"{'metric':<38}{'bridges':>12}{'random':>12}")
    for k in ["frac_pairs_with_shared", "mean_shared_mass", "mean_generic_frac",
              "median_generic_frac", "frac_pairs_with_specific_shared",
              "mean_n_specific_shared"]:
        print(f"{k:<38}{str(bridge_summary[k]):>12}{str(ctrl_summary[k]):>12}")
    print(f"\nConceptual-bridge pairs sharing >=1 domain-specific CONTENT token: "
          f"{sum(s['has_specific'] for s in bridge_stats)}/{len(bridge_stats)} "
          f"(control rate {ctrl_summary['frac_pairs_with_specific_shared']})")
    print("\nTop pairs with genuine domain-specific shared CONTENT tokens:")
    for s in specific_examples[:10]:
        print(f"  ED={s['norm_edit_dist']:.3f}  n_specific={s['n_specific_shared']}  "
              f"{s['specific_tokens']}")
        print(f"      {s['pair'][0]}  <->  {s['pair'][1]}")
    print("\nSaved -> results/conceptual_content_bridges.json")
    return out


if __name__ == "__main__":
    main()
