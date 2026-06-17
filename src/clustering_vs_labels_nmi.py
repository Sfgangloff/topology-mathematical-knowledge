"""
q-0022: What does unsupervised Leiden clustering add BEYOND the Mathlib module
labels? If clustering merely re-derives the labels, NMI ~ 1 and there is no extra
information; if it carves the space differently, where and how?

Compute, over all nodes that carry a top-level Mathlib module label:
  - Leiden partition  L  (data/clusters.json)
  - Module partition  M  (top-level Mathlib module from file path)
  Metrics: NMI, AMI, homogeneity, completeness, V-measure (sklearn).
  Homogeneity ~ "each cluster is one module" (clusters purer than modules?).
  Completeness ~ "each module is one cluster" (does Leiden split modules?).

Structural diagnostics (what clustering ADDS):
  - n Leiden clusters vs n modules.
  - module fragmentation: for each module, # Leiden clusters its nodes spread
    across, and the fraction of the module in its single largest cluster
    (= does Leiden SPLIT a module into sub-domains -> added granularity).
  - cluster mixing: for each Leiden cluster, # modules it draws from and the
    dominant-module fraction (= does Leiden MERGE modules -> bridge zones).
This quantifies the contribution: re-labeling (NMI high) vs sub-structuring
(homogeneity > completeness) vs cross-cutting (low completeness).
"""
import json, pickle
from collections import defaultdict, Counter
from pathlib import Path
from sklearn import metrics

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"


def top_module(module: str):
    parts = module.split(".")
    if "Mathlib" in parts:
        i = parts.index("Mathlib")
        return parts[i + 1] if i + 1 < len(parts) else "Mathlib"
    return None


def main():
    G = pickle.load(open(DATA / "theorem_graph.pkl", "rb"))
    leiden = json.load(open(DATA / "clusters.json"))

    rows = []
    for n in G.nodes():
        m = top_module(G.nodes[n].get("module", ""))
        c = leiden.get(n)
        if m is None or c is None:
            continue
        rows.append((n, str(c), m))

    L = [r[1] for r in rows]
    M = [r[2] for r in rows]
    N = len(rows)

    info = {
        "n_nodes_labelled": N,
        "n_leiden_clusters": len(set(L)),
        "n_modules": len(set(M)),
        "NMI": round(metrics.normalized_mutual_info_score(M, L), 4),
        "AMI": round(metrics.adjusted_mutual_info_score(M, L), 4),
        "homogeneity_cluster_is_one_module": round(metrics.homogeneity_score(M, L), 4),
        "completeness_module_is_one_cluster": round(metrics.completeness_score(M, L), 4),
        "v_measure": round(metrics.v_measure_score(M, L), 4),
        "ARI": round(metrics.adjusted_rand_score(M, L), 4),
    }

    # module fragmentation: how Leiden splits each module
    mod_to_clusters = defaultdict(Counter)
    clu_to_mods = defaultdict(Counter)
    for _, c, m in rows:
        mod_to_clusters[m][c] += 1
        clu_to_mods[c][m] += 1

    frag = []
    for m, cc in mod_to_clusters.items():
        tot = sum(cc.values())
        largest = max(cc.values())
        frag.append({"module": m, "size": tot,
                     "n_leiden_clusters_spanned": len(cc),
                     "frac_in_largest_cluster": round(largest / tot, 3)})
    frag.sort(key=lambda x: x["n_leiden_clusters_spanned"], reverse=True)

    # cluster mixing: how many modules each big cluster merges
    mix = []
    for c, mm in clu_to_mods.items():
        tot = sum(mm.values())
        if tot < 200:  # only sizable clusters
            continue
        largest = max(mm.values())
        mix.append({"cluster": c, "size": tot,
                    "n_modules_merged": len(mm),
                    "dominant_module_frac": round(largest / tot, 3),
                    "dominant_module": mm.most_common(1)[0][0]})
    mix.sort(key=lambda x: x["size"], reverse=True)

    import statistics
    info["module_fragmentation_top10"] = frag[:10]
    info["mean_clusters_per_module"] = round(
        statistics.mean(f["n_leiden_clusters_spanned"] for f in frag), 2)
    info["median_frac_in_largest_cluster"] = round(
        statistics.median(f["frac_in_largest_cluster"] for f in frag), 3)
    info["big_clusters_mixing_top10"] = mix[:10]
    info["n_big_clusters"] = len(mix)
    info["n_big_clusters_mixed_dom_lt50"] = sum(1 for x in mix if x["dominant_module_frac"] < 0.5)

    json.dump(info, open(RES / "clustering_vs_labels_nmi.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in info.items()
                      if not k.endswith("_top10")}, indent=2))
    print("\nfragmentation top5:", info["module_fragmentation_top10"][:5])
    print("\nmixing top5:", info["big_clusters_mixing_top10"][:5])


if __name__ == "__main__":
    main()
