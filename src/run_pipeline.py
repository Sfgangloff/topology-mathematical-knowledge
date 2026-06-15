"""
Full pipeline:
  1. Download data (corpus + tactics)
  2. Build theorem dependency graph
  3. Leiden clustering
  4. Label clusters by Mathlib module
  5. Hierarchical re-clustering
  6. N-gram length sweep (find right granularity)
  7. Bridge detection (entropy × rarity score)
  8. Visualisations
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_data       import download_all, load_corpus, load_tactics
from build_graph     import build as build_graph, save as save_graph, load as load_graph, GRAPH_FILE, TACTICS_FILE
from cluster         import cluster, summarise, save as save_clusters, load as load_clusters, CLUSTER_FILE, DATA_DIR
from label_clusters  import label_clusters, save as save_labels, load as load_labels, report as report_labels, LABELS_FILE
from hierarchical    import recluster, label_sub_clusters, save as save_sub, SUB_FILE, SUB_LABELS
from ngram_analysis  import sweep as ngram_sweep, plot as plot_ngram_sweep
from bridges         import extract_ngram_profiles, find_bridges, save as save_bridges, report as report_bridges
from visualize       import main as make_plots

CLUSTER_SUMMARY_FILE = DATA_DIR / "cluster_summary.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    """Run the full bridge-detection pipeline from raw data download to visualizations.

    Each step checks for cached outputs first; delete the relevant data/ or results/
    file to force recomputation. Step order:
      1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
    (download → graph → clusters → labels → hierarchical → n-gram sweep → bridges → plots)
    """
    # ── 1. Data ───────────────────────────────────────────────────────────────
    download_all()

    # ── 2. Graph ──────────────────────────────────────────────────────────────
    if GRAPH_FILE.exists() and TACTICS_FILE.exists():
        print("Graph cached — loading.")
        G, theorem_tactics = load_graph()
    else:
        print("\n=== Building dependency graph ===")
        corpus  = load_corpus()
        records = load_tactics()
        G, theorem_tactics = build_graph(records, corpus)
        save_graph(G, theorem_tactics)

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    if CLUSTER_FILE.exists() and CLUSTER_SUMMARY_FILE.exists():
        print("Clusters cached — loading.")
        cluster_map = load_clusters()
        with open(CLUSTER_SUMMARY_FILE) as f:
            cluster_summary = json.load(f)
    else:
        print("\n=== Clustering (Leiden) ===")
        cluster_map = cluster(G)
        cluster_summary = summarise(cluster_map, G)
        save_clusters(cluster_map, cluster_summary)

    # ── 4. Cluster labels ─────────────────────────────────────────────────────
    if LABELS_FILE.exists():
        print("Labels cached — loading.")
        labels = load_labels()
    else:
        print("\n=== Labelling clusters ===")
        labels = label_clusters(cluster_map, G)
        save_labels(labels)
    report_labels(labels)

    # ── 5. Hierarchical re-clustering ─────────────────────────────────────────
    if SUB_FILE.exists():
        print("\nSub-clusters cached.")
    else:
        print("\n=== Hierarchical re-clustering ===")
        sub_map    = recluster(G, cluster_map)
        sub_labels = label_sub_clusters(sub_map, G)
        save_sub(sub_map, sub_labels)

    # ── 6. N-gram sweep ───────────────────────────────────────────────────────
    sweep_file = RESULTS_DIR / "ngram_sweep.json"
    if sweep_file.exists():
        print("\nN-gram sweep cached.")
        with open(sweep_file) as f:
            sweep_results = json.load(f)
    else:
        print("\n=== N-gram length sweep ===")
        sweep_results = ngram_sweep(theorem_tactics, cluster_map, max_n=8)
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(sweep_file, "w") as f:
            json.dump(sweep_results, f, indent=2)
        plot_ngram_sweep(sweep_results)

    # ── 7. Bridge detection ───────────────────────────────────────────────────
    # Pick n based on sweep: the n where frac_ge2 first drops below 2%
    # (i.e. cross-cluster patterns become genuinely rare → locally meaningful).
    # We skip n=1 since single tactics are too coarse to be interesting bridges.
    best_n = 3
    for row in sweep_results:
        if row["n"] < 2:
            continue
        if row["frac_ge2"] < 0.02:
            best_n = row["n"]
            break
    print(f"\n=== Bridge detection at n={best_n} and n={best_n+1} ===")

    all_bridges = []
    for n in [best_n, best_n + 1]:
        profiles = extract_ngram_profiles(theorem_tactics, cluster_map, n=n)
        all_bridges += find_bridges(profiles, min_clusters=2, min_count=3)
    all_bridges.sort(key=lambda x: -x["bridge_score"])
    save_bridges(all_bridges)
    report_bridges(all_bridges, top=30)

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    print("\n=== Generating visualisations ===")
    make_plots()

    print("\n✓ Pipeline complete.  Outputs in results/")


if __name__ == "__main__":
    main()
