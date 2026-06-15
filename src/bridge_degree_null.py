"""
Is operational-bridge detection just a proxy for PREMISE COUNT?

a-0044 established that the 280 detected operational bridges (premises spanning
>=2 distinct PURE clusters, results/bridge_theorems.json) are premise-heavy deep
syntheses: median 12 premises vs 3 for a typical theorem (rankfrac 0.817). That
raises a confound the detection half never controls for: a theorem with many
premises has many independent CHANCES to land in a pure cluster, so it will span
>=2 pure clusters *mechanically*, regardless of any genuine cross-domain intent.
If the bridge signal is fully explained by premise count, the detector measures
proof size, not cross-domain synthesis.

This is a cheap, stdlib-only census + analytic null over existing data
(data/tactic_steps.jsonl premises, data/clusters.json, data/cluster_labels.json).
It reproduces all 280 canonical bridges exactly from the tactic_steps premise
reconstruction (verified), then asks whether the bridge rate survives a
DEGREE-PRESERVING null.

Null model (configuration / "throw k random premises"):
  Fix each theorem's premise COUNT k = #distinct premises. Replace its actual
  premises with k premises drawn i.i.d. from a global premise distribution, then
  recompute how many distinct PURE clusters they span. Two weightings:
    - TYPE-weighted   w_c = (#distinct premises in pure cluster c) / (#distinct premises)
    - USAGE-weighted  w_c = (#premise-edge USAGES landing in pure cluster c) / (total usages)
  USAGE-weighted is the standard configuration-model null: it preserves how often
  each premise is actually invoked, so popular foundational lemmas (which live in
  big MIXED hub clusters, not pure ones) dominate -- exactly the realistic null.

  For k i.i.d. draws with P(land in pure cluster c)=w_c and P(non-pure)=w0:
    P(span >=2 distinct pure) = 1 - w0^k - sum_c [ (w0+w_c)^k - w0^k ]
  (closed form: 1 - P(0 pure) - P(exactly 1 distinct pure)). Pool is ~5x10^4
  distinct premises, so with-replacement is an excellent approximation to
  without-replacement at the relevant k.

We report observed vs null bridge rate OVERALL and STRATIFIED BY premise-count
bin, plus the per-theorem enrichment (observed n_pure / expected n_pure under
the null). If observed >> null within bins, the bridge signal is real beyond
degree; if observed ~ null, it is a premise-count artifact.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"


def load():
    out_prem = defaultdict(set)
    with open(DATA / "tactic_steps.jsonl") as f:
        for line in f:
            o = json.loads(line)
            n = o["full_name"]
            for p in o["premises"]:
                if p != n:
                    out_prem[n].add(p)
    cmap = {k: int(v) for k, v in json.load(open(DATA / "clusters.json")).items()}
    lab = json.load(open(DATA / "cluster_labels.json"))
    pure = {
        int(c)
        for c, i in lab.items()
        if (not i["is_mixed"]) and i["dominant_fraction"] >= 0.50
    }
    return out_prem, cmap, pure


def pure_cluster_of(p, cmap, pure):
    """Return the pure-cluster id of premise p, or None (non-pure / unclustered)."""
    c = cmap.get(p)
    return c if c in pure else None


def null_prob_ge2(k, w0, wvals):
    """Closed-form P(>=2 distinct pure clusters) for k iid draws.

    w0 = P(draw is non-pure); wvals = list of per-pure-cluster probabilities.
    P(>=2) = 1 - w0^k - sum_c [(w0+w_c)^k - w0^k].
    """
    if k <= 1:
        return 0.0
    p0 = w0 ** k
    p_exactly1 = 0.0
    for wc in wvals:
        if wc > 0.0:
            p_exactly1 += (w0 + wc) ** k - p0
    return max(0.0, 1.0 - p0 - p_exactly1)


def null_expected_npure(k, w0, wvals):
    """Expected number of DISTINCT pure clusters hit in k iid draws.

    E[# distinct] = sum_c (1 - (1-w_c)^k).
    """
    if k <= 0:
        return 0.0
    return sum(1.0 - (1.0 - wc) ** k for wc in wvals if wc > 0.0)


def main():
    out_prem, cmap, pure = load()

    # ---- observed ----
    theorems = [(n, prems) for n, prems in out_prem.items() if prems]
    obs_npure = {}
    obs_bridge = set()
    for n, prems in theorems:
        pcl = {cmap[p] for p in prems if cmap.get(p) in pure}
        obs_npure[n] = len(pcl)
        if len(pcl) >= 2:
            obs_bridge.add(n)
    n_pop = len(theorems)
    print(f"population (theorems w/ premises): {n_pop}")
    print(f"observed bridges (>=2 pure clusters): {len(obs_bridge)} "
          f"({100*len(obs_bridge)/n_pop:.3f}%)")

    # ---- global premise distributions over pure clusters ----
    # TYPE-weighted: over distinct premises that actually appear as a premise
    distinct_premises = set()
    usage_count = defaultdict(int)  # premise -> #theorems using it
    for n, prems in theorems:
        for p in prems:
            distinct_premises.add(p)
            usage_count[p] += 1

    type_pure = defaultdict(int)
    type_total = 0
    for p in distinct_premises:
        type_total += 1
        c = pure_cluster_of(p, cmap, pure)
        if c is not None:
            type_pure[c] += 1
    type_w0 = 1.0 - sum(type_pure.values()) / type_total
    type_wvals = [type_pure[c] / type_total for c in type_pure]

    usage_pure = defaultdict(int)
    usage_total = 0
    for p, u in usage_count.items():
        usage_total += u
        c = pure_cluster_of(p, cmap, pure)
        if c is not None:
            usage_pure[c] += u
    usage_w0 = 1.0 - sum(usage_pure.values()) / usage_total
    usage_wvals = [usage_pure[c] / usage_total for c in usage_pure]

    print(f"\ndistinct premises: {type_total}; total premise usages: {usage_total}")
    print(f"TYPE-weighted  P(premise in some pure cluster) = {1-type_w0:.4f}")
    print(f"USAGE-weighted P(premise in some pure cluster) = {1-usage_w0:.4f}  "
          f"(<- popular premises live in big MIXED hubs)")

    # ---- expected bridges under each null (sum of per-theorem probs) ----
    def null_summary(w0, wvals, tag):
        exp_bridges = 0.0
        for n, prems in theorems:
            k = len(prems)
            exp_bridges += null_prob_ge2(k, w0, wvals)
        rate = exp_bridges / n_pop
        enrich = len(obs_bridge) / exp_bridges if exp_bridges > 0 else float("inf")
        print(f"\n[{tag} null] expected bridges {exp_bridges:.1f} "
              f"({100*rate:.3f}%)  -> observed/expected enrichment = {enrich:.2f}x")
        return {"tag": tag, "expected_bridges": round(exp_bridges, 2),
                "expected_rate": round(rate, 6), "enrichment": round(enrich, 3)}

    type_s = null_summary(type_w0, type_wvals, "TYPE")
    usage_s = null_summary(usage_w0, usage_wvals, "USAGE")

    # ---- stratified by premise-count bin ----
    bins = [(2, 3), (4, 5), (6, 9), (10, 19), (20, 10 ** 9)]
    strat = []
    print(f"\n{'k-bin':>10}  {'n':>7}  {'obs%':>7}  {'TYPEnull%':>10}  {'USAGEnull%':>11}  {'enrich(USAGE)':>13}")
    for lo, hi in bins:
        grp = [(n, prems) for n, prems in theorems if lo <= len(prems) <= hi]
        if not grp:
            continue
        nobs = sum(1 for n, _ in grp if n in obs_bridge)
        et = sum(null_prob_ge2(len(prems), type_w0, type_wvals) for _, prems in grp)
        eu = sum(null_prob_ge2(len(prems), usage_w0, usage_wvals) for _, prems in grp)
        m = len(grp)
        enr = nobs / eu if eu > 0 else float("inf")
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"{label:>10}  {m:>7}  {100*nobs/m:>6.2f}%  {100*et/m:>9.3f}%  "
              f"{100*eu/m:>10.3f}%  {enr:>12.2f}x")
        strat.append({
            "bin": label, "n": m, "obs_bridges": nobs,
            "obs_rate": round(nobs / m, 5),
            "type_null_rate": round(et / m, 5),
            "usage_null_rate": round(eu / m, 5),
            "usage_enrichment": round(enr, 3),
        })

    # ---- among the 280 detected bridges, observed vs expected #distinct pure ----
    bvals_obs = [obs_npure[n] for n in obs_bridge]
    bvals_exp_usage = [
        null_expected_npure(len(out_prem[n]), usage_w0, usage_wvals) for n in obs_bridge
    ]
    print(f"\nWithin the 280 detected bridges: observed mean #distinct-pure-clusters "
          f"= {sum(bvals_obs)/len(bvals_obs):.2f} vs USAGE-null expectation "
          f"{sum(bvals_exp_usage)/len(bvals_exp_usage):.2f} "
          f"(at their actual premise counts)")

    out = {
        "population": n_pop,
        "observed_bridges": len(obs_bridge),
        "observed_rate": round(len(obs_bridge) / n_pop, 6),
        "distinct_premises": type_total,
        "total_usages": usage_total,
        "type_p_pure": round(1 - type_w0, 5),
        "usage_p_pure": round(1 - usage_w0, 5),
        "null_overall": {"type": type_s, "usage": usage_s},
        "stratified_by_premise_count": strat,
        "detected_bridges_npure_observed_mean": round(sum(bvals_obs) / len(bvals_obs), 3),
        "detected_bridges_npure_usagenull_mean": round(sum(bvals_exp_usage) / len(bvals_exp_usage), 3),
        "note": "Degree-preserving (configuration) null: keep each theorem's premise "
                "COUNT, redraw premises iid from the global premise distribution, "
                "recompute pure-cluster span analytically. enrichment = observed / "
                "null-expected bridge count.",
    }
    json.dump(out, open(RESULTS / "bridge_degree_null.json", "w"), indent=2)
    print("\nwrote results/bridge_degree_null.json")


if __name__ == "__main__":
    main()
