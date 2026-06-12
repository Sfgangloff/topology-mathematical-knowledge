"""
Selective (gated) bridge routing: route bridge ONLY for predicted-cross-domain goals.
(q-0011, direct test of the lever a-0024 named.)

a-0024 measured the head-to-head at prover-realistic premise windows B (10..1000 decls) and
found that ALWAYS-ON bridge routing HURTS the whole-population retrieval: within-domain goals
(half the test set) gain nothing from neighbour clusters and only suffer distractor dilution,
so home_only beats always-on bridge on both mean recall and complete coverage at every window.
Its operational implication was explicit: "bridge routing must be applied SELECTIVELY (only to
predicted-cross-domain goals, cf. a-0023 home prediction)."  This script tests exactly that.

A real prover meeting a fresh goal does NOT know whether the goal is cross-domain (it has no
premises yet -- those are what retrieval seeks).  So the gate must be a LEAKAGE-FREE predictor
of cross-domain-ness from the only signal available: the goal's NAME tokens.

Predictor (leakage-safe, TRAIN-only): for each name token t, P(cross|t) = (# TRAIN theorems
with token t that are cross-domain) / (# TRAIN theorems with token t).  A test goal's
cross-score = IDF-weighted mean of P(cross|t) over its tokens (a probability-like number in
[0,1]).  The decision threshold is chosen ON TRAIN ONLY: set it to the (1 - base_rate) score
quantile so the predicted-positive rate on TRAIN matches the TRAIN cross-domain base rate
(no test labels touched).

Three arms, identical content ranker / leakage control / windows as e-0027 (a-0024):
  - home_only  : pool = home cluster only                          (never bridge)
  - bridge     : pool = home + top-5 bridge neighbours             (always bridge)
  - selective  : pool = bridge IFF predicted cross-domain, else home_only   (the gate)

Metrics at each window B in {10,25,50,100,200,500,1000}, reported on the WHOLE 873-theorem
test population (the honest population-level number a-0024 used), plus cross/within splits:
  - mean recall        : avg fraction of a theorem's clustered premises in top-B
  - complete coverage  : fraction of theorems whose ENTIRE clustered premise set is in top-B

Hypothesis: selective gating keeps bridge's cross-domain ABILITY while sparing within-domain
goals the distractor dilution -- so it should weakly dominate BOTH pure arms at the population
level.  Whether it actually does depends on the gate's accuracy (also reported: AUC, and
precision/recall/accuracy at the train-chosen threshold).

Still a premise-availability upper bound, not a measured prover success rate or transfer.

Output: results/bridge_retrieval_selective.json
"""

import json
import re
import math
import heapq
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
STEPS = DATA / "tactic_steps.jsonl"
CLUSTERS = DATA / "clusters.json"
OUT = RESULTS / "bridge_retrieval_selective.json"

K_ROUTE = 5
WINDOWS = [10, 25, 50, 100, 200, 500, 1000]
DF_STOP_FRAC = 0.20

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(name):
    toks = set()
    for part in name.replace("_", ".").split("."):
        for m in _CAMEL.findall(part):
            if len(m) >= 2:
                toks.add(m.lower())
    return toks


def load():
    cluster = json.load(open(CLUSTERS))
    premises = collections.defaultdict(set)
    split = {}
    for line in open(STEPS):
        d = json.loads(line)
        fn = d["full_name"]
        split[fn] = d["split"]
        for p in d.get("premises", []):
            if p != fn:
                premises[fn].add(p)
    return cluster, premises, split


def build_priors(cluster, premises, split):
    """top-K bridge neighbours per home cluster + citation popularity, TRAIN only."""
    W = collections.defaultdict(lambda: collections.Counter())
    cite = collections.Counter()
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        c = cluster.get(fn)
        for p in ps:
            cite[p] += 1
        if c is None:
            continue
        seen = set()
        for p in ps:
            d = cluster.get(p)
            if d is None or d == c or d in seen:
                continue
            seen.add(d)
            W[c][d] += 1
    return W, cite


def build_content_index(cluster):
    decl_tokens = {}
    df = collections.Counter()
    for name in cluster:
        t = tokenize(name)
        decl_tokens[name] = t
        for tok in t:
            df[tok] += 1
    N = len(cluster)
    stop = {tok for tok, c in df.items() if c > DF_STOP_FRAC * N}
    idf = {tok: math.log(N / c) for tok, c in df.items() if tok not in stop}
    decl_tokens = {n: (t - stop) for n, t in decl_tokens.items()}
    return decl_tokens, idf, stop


def is_cross_domain(fn, ps, cluster):
    c = cluster.get(fn)
    prem_clusters = {cluster[p] for p in ps if p in cluster}
    return bool(prem_clusters) and (prem_clusters - {c}) != set()


def train_cross_predictor(cluster, premises, split, idf, stop):
    """token t -> P(cross|t) over TRAIN theorems (leakage-free: labels from train premises only).
    Also returns the train base cross-rate and the list of train cross-scores for thresholding."""
    tok_cross = collections.Counter()   # # train theorems with token t that are cross-domain
    tok_total = collections.Counter()   # # train theorems with token t
    n_train = 0
    n_train_cross = 0
    train_examples = []                 # (tokens, is_cross) for thresholding
    for fn, ps in premises.items():
        if split.get(fn) != "train":
            continue
        if cluster.get(fn) is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        toks = (tokenize(fn) - stop) & idf.keys()
        xc = is_cross_domain(fn, ps, cluster)
        n_train += 1
        n_train_cross += int(xc)
        train_examples.append((toks, xc))
        for t in toks:
            tok_total[t] += 1
            if xc:
                tok_cross[t] += 1
    pcross = {t: tok_cross[t] / tok_total[t] for t in tok_total}
    base = n_train_cross / n_train
    return pcross, base, train_examples


def cross_score(toks, pcross, idf, base):
    """IDF-weighted mean of P(cross|t); falls back to base rate when no informative token."""
    num = den = 0.0
    for t in toks:
        w = idf.get(t)
        if w is None or t not in pcross:
            continue
        num += w * pcross[t]
        den += w
    return num / den if den > 0 else base


def choose_threshold(train_examples, pcross, idf, base):
    """Threshold = (1 - base) quantile of TRAIN scores so predicted-positive rate ~ base rate.
    Leakage-free: uses only train scores/labels."""
    scores = sorted(cross_score(toks, pcross, idf, base) for toks, _ in train_examples)
    if not scores:
        return base
    # we want ~base fraction predicted positive -> cut at the (1-base) quantile
    idx = min(len(scores) - 1, int((1.0 - base) * len(scores)))
    return scores[idx]


def auc_and_pr(test_rows, thr):
    """Mann-Whitney AUC of cross-score vs true cross label, plus precision/recall/acc at thr."""
    pos = [s for s, y in test_rows if y]
    neg = [s for s, y in test_rows if not y]
    # AUC via rank-sum
    alls = sorted(((s, y) for s, y in test_rows), key=lambda r: r[0])
    rank = {}
    i = 0
    while i < len(alls):
        j = i
        while j < len(alls) and alls[j][0] == alls[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j):
            rank[k] = avg_rank
        i = j
    sum_pos_ranks = sum(rank[k] for k, (s, y) in enumerate(alls) if y)
    np_, nn_ = len(pos), len(neg)
    auc = (sum_pos_ranks - np_ * (np_ + 1) / 2.0) / (np_ * nn_) if np_ and nn_ else float("nan")
    tp = sum(1 for s, y in test_rows if s >= thr and y)
    fp = sum(1 for s, y in test_rows if s >= thr and not y)
    fn = sum(1 for s, y in test_rows if s < thr and y)
    tn = sum(1 for s, y in test_rows if s < thr and not y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / len(test_rows) if test_rows else 0.0
    return {
        "auc": round(auc, 4),
        "threshold": round(thr, 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "accuracy": round(acc, 4),
        "predicted_positive_rate": round((tp + fp) / len(test_rows), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def content_rank_top(pool, q_tokens, decl_tokens, idf, cite, topn):
    qi = {tok: idf.get(tok, 0.0) for tok in q_tokens}

    def key(name):
        s = 0.0
        tt = decl_tokens.get(name)
        if tt:
            for tok in qi:
                if tok in tt:
                    s += qi[tok]
        return (s, cite.get(name, 0), name)

    return heapq.nlargest(topn, pool, key=key)


def evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop,
             pcross, base, thr):
    members = collections.defaultdict(list)
    for name, cid in cluster.items():
        members[cid].append(name)
    pop_sorted = {cid: sorted(ms, key=lambda n: -cite.get(n, 0)) for cid, ms in members.items()}
    topk_bridge = {c: [d for d, _ in W[c].most_common(K_ROUTE)] for c in W}

    nb = len(WINDOWS)
    arms = ("home_only", "bridge", "selective")
    grps = ("all", "cross", "within")

    def zeros():
        return {"recall": [0.0] * nb, "complete": [0.0] * nb}
    acc = {(arm, g): zeros() for arm in arms for g in grps}
    counts = {g: 0 for g in grps}
    topn = max(WINDOWS)

    test_rows = []  # (cross_score, true_is_cross) for gate diagnostics

    for fn, ps in premises.items():
        if split.get(fn) != "test":
            continue
        c = cluster.get(fn)
        if c is None:
            continue
        prem = {p for p in ps if p in cluster}
        if not prem:
            continue
        total = len(prem)
        xc = (({cluster[p] for p in prem} - {c}) != set())

        q_tokens = tokenize(fn) - stop
        score = cross_score(q_tokens & idf.keys(), pcross, idf, base)
        pred_cross = score >= thr
        test_rows.append((score, xc))

        bridge_pool_cids = {c} | set(topk_bridge.get(c, []))
        home_pool_cids = {c}

        arm_pools = {
            "home_only": home_pool_cids,
            "bridge": bridge_pool_cids,
            "selective": bridge_pool_cids if pred_cross else home_pool_cids,
        }

        counts["all"] += 1
        counts["cross" if xc else "within"] += 1

        for arm, cids in arm_pools.items():
            pool = []
            for cid in cids:
                pool.extend(pop_sorted.get(cid, []))
            ranked = content_rank_top(pool, q_tokens, decl_tokens, idf, cite, topn)
            target_grps = ["all", "cross" if xc else "within"]
            for bi, B in enumerate(WINDOWS):
                topB = set(ranked[:B])
                hit = len(prem & topB)
                rec = hit / total
                full = 1.0 if hit == total else 0.0
                for g in target_grps:
                    acc[(arm, g)]["recall"][bi] += rec
                    acc[(arm, g)]["complete"][bi] += full

    def curves(arm, g):
        n = counts[g]
        a = acc[(arm, g)]
        return {
            "mean_recall": [round(a["recall"][bi] / n, 4) for bi in range(nb)],
            "complete_coverage": [round(a["complete"][bi] / n, 4) for bi in range(nb)],
        }

    out = {
        "metric_note": "selective (gated) bridge routing vs always-home / always-bridge, "
                       "at prover-realistic premise windows; bridge applied IFF a leakage-free "
                       "name-token cross-domain predictor fires",
        "windows": WINDOWS,
        "k_route": K_ROUTE,
        "n_test": counts["all"], "n_cross": counts["cross"], "n_within": counts["within"],
        "frac_cross": round(counts["cross"] / counts["all"], 4),
        "gate": auc_and_pr(test_rows, thr),
        "train_base_cross_rate": round(base, 4),
        "population_all": {arm: curves(arm, "all") for arm in arms},
        "cross_domain": {arm: curves(arm, "cross") for arm in arms},
        "within_domain": {arm: curves(arm, "within") for arm in arms},
    }
    # population-level head-to-head deltas vs home_only (the a-0024 winner)
    h = out["population_all"]["home_only"]
    out["population_uplift_vs_home_only"] = {}
    for arm in ("bridge", "selective"):
        a = out["population_all"][arm]
        out["population_uplift_vs_home_only"][arm] = {
            "mean_recall": [round(a["mean_recall"][bi] - h["mean_recall"][bi], 4) for bi in range(nb)],
            "complete_coverage": [round(a["complete_coverage"][bi] - h["complete_coverage"][bi], 4) for bi in range(nb)],
        }
    return out


def main():
    cluster, premises, split = load()
    W, cite = build_priors(cluster, premises, split)
    decl_tokens, idf, stop = build_content_index(cluster)
    pcross, base, train_examples = train_cross_predictor(cluster, premises, split, idf, stop)
    thr = choose_threshold(train_examples, pcross, idf, base)
    res = evaluate(cluster, premises, split, W, cite, decl_tokens, idf, stop, pcross, base, thr)
    res["params"] = {
        "k_route": K_ROUTE, "windows": WINDOWS, "df_stop_frac": DF_STOP_FRAC,
        "content_ranker": "IDF-weighted query-name token overlap, popularity tiebreak",
        "gate": "P(cross|name token) IDF-weighted mean, threshold = (1-base) train-score quantile",
        "leakage_control": "W, IDF, citation popularity, gate priors+threshold from train/name-vocab only",
    }
    RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
