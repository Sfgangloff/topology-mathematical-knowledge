# Topology of the Mathematical Domain

Experiment code for an unsupervised study of the **structure of mathematics** —
clustering the Lean 4 / Mathlib theorem-dependency graph into domains and
detecting cross-domain **bridge theorems**.

> **The research reasoning lives elsewhere.** Questions, hypotheses, findings,
> and how the project's thinking evolved are maintained as a structured
> reasoning graph in the **research-compiler** database, stream
> **`mathematical-domain-topology`**. To read it, open the research-compiler web
> app or run `rc export paper --stream mathematical-domain-topology`. Each
> analysis below is the node `e-00NN` in that stream.

## Headline

Mathematics' domain structure is recoverable from formal proofs with **no
supervision**: Leiden clustering of Mathlib's dependency graph reproduces named
domains (138 clusters), and an in-degree-filtered cross-cluster premise
criterion detects **operational bridge theorems** (34) that align with
historically celebrated connections (7/8 span distinct clusters, 6/8 loosely
detected). The central open problem is the gap between **operational** bridges
(shared premises — detected) and **conceptual** bridges (shared proof structure,
e.g. Brouwer↔Lawvere — undetected).

## Pipeline / experiment index

`src/run_pipeline.py` orchestrates build → cluster → label → n-gram → bridges;
the validation legs run as standalone scripts. Each row is a node in the
`mathematical-domain-topology` stream.

| Node | Analysis | Script | Output |
| --- | --- | --- | --- |
| `e-0001` | theorem dependency graph | `src/build_graph.py` | `data/theorem_graph.pkl`, `data/theorem_tactics.json` |
| `e-0002` | Leiden clustering + labeling | `src/cluster.py`, `src/label_clusters.py` | `data/clusters.json`, `data/cluster_labels.json` |
| `e-0003` | hierarchical sub-clustering | `src/hierarchical.py` | `data/sub_clusters.json` |
| `e-0004` | tactic n-gram domain-specificity | `src/ngram_analysis.py` (`src/normalize_tactics.py`) | `results/ngram_sweep.json` |
| `e-0005` | n-gram bridge scorer | `src/bridges.py` | `results/bridges.json` |
| `e-0006` | operational bridge theorems | `src/bridge_theorems.py` | `results/bridge_theorems.json` |
| `e-0007` | in-degree depth filter (θ) | `src/depth_filter.py` | `results/bridge_theorems_filtered.json`, `results/indegree_sweep.json` |
| `e-0008` | historical-bridge validation | `src/validate_bridges.py` | `results/validation.json` |
| `e-0009` | edit-distance (conceptual) bridges | `src/edit_distance_bridges.py` | `results/edit_distance_bridges.json` |
| `e-0010` | Tononi–Edelman neural complexity | `src/neural_complexity.py` | `results/neural_complexity.json` |
| — | visualizations | `src/visualize.py` | `results/*.png` |

## Data

`data/` holds the corpus (`corpus.jsonl`, 180,973 declarations; `tactic_steps.jsonl`,
259,580 steps) and derived artifacts (dependency graph, clusters), sourced from
the LeanDojo / Mathlib4 HuggingFace datasets via `src/load_data.py`.

## Running

```bash
pip install -r requirements.txt
python -m src.run_pipeline        # build -> cluster -> label -> n-gram -> bridges
python -m src.validate_bridges    # historical-bridge validation
python -m src.neural_complexity   # cluster neural-complexity
```
