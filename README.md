# Topology of the Mathematical Domain

An unsupervised analysis of the structure of mathematics using formal proofs from
Lean 4 / Mathlib. We build a theorem dependency graph, cluster it into mathematical
domains, and detect "bridge theorems" — results whose proofs draw on multiple distinct
domains, suggesting deeper connections between fields.

**Target venue:** AI for Math workshop (ICML 2026 track).

---

## Research question

Can we recover the topology of mathematics — its domain structure and cross-domain
bridges — from formal proof data alone, without any hand-coded labels?

We find that the answer is largely yes: Leiden community detection on the theorem
dependency graph recovers named Mathlib modules (Topology, Algebra, MeasureTheory, …)
without supervision, and our bridge detectors recover 6 of 8 historically-known
cross-domain connections.

---

## Pipeline overview

```
raw HuggingFace data
        │
        ▼
[1] load_data.py        Download & cache datasets
        │
        ▼
[2] build_graph.py      Directed dependency graph (186K nodes, 309K edges)
        │
        ▼
[3] cluster.py          Leiden clustering → 138 clusters
        │
        ▼
[4] label_clusters.py   Label clusters by dominant Mathlib module
        │
        ▼
[5] hierarchical.py     Re-cluster within each top cluster (sub-domain structure)
        │
        ▼
[6] normalize_tactics.py  Abstract tactic strings → proof-strategy tokens
        │
        ▼
[7] ngram_analysis.py   Sweep n=1..8, find cross-cluster coverage elbow
        │  (parallel)
        ▼
[8] bridges.py          Score bridge n-grams: H_norm × rarity
[8b] bridge_theorems.py  Find theorems with cross-cluster premises
[8c] edit_distance_bridges.py  Find structurally similar proofs across clusters
[8d] depth_filter.py    Filter bridges by premise specificity (in-degree ≤ θ)
        │
        ▼
[9] validate_bridges.py  Check against 8 known historical bridges
[9b] neural_complexity.py  Tononi-Edelman complexity on cluster graph
        │
        ▼
[10] visualize.py       Produce all figures (cluster graph, purity scatter, …)
        │
        ▼
    results/*.png / *.json
```

Run everything with:

```bash
python3 src/run_pipeline.py
```

Additional analyses (not in the main pipeline) run standalone:

```bash
python3 src/bridge_theorems.py       # step 8b
python3 src/edit_distance_bridges.py # step 8c
python3 src/depth_filter.py          # step 8d
python3 src/validate_bridges.py      # step 9
python3 src/neural_complexity.py     # step 9b
```

---

## Source files

| File | Role |
|---|---|
| [`src/load_data.py`](src/load_data.py) | Download and cache LeanDojo datasets from HuggingFace |
| [`src/build_graph.py`](src/build_graph.py) | Build directed theorem dependency graph and tactic sequences |
| [`src/cluster.py`](src/cluster.py) | Leiden community detection on the undirected graph |
| [`src/label_clusters.py`](src/label_clusters.py) | Label clusters by dominant Mathlib module; flag mixed clusters |
| [`src/hierarchical.py`](src/hierarchical.py) | Re-cluster within each top-level cluster |
| [`src/normalize_tactics.py`](src/normalize_tactics.py) | Abstract tactic strings to proof-strategy token sequences |
| [`src/ngram_analysis.py`](src/ngram_analysis.py) | Sweep n-gram lengths, measure cross-cluster coverage |
| [`src/bridges.py`](src/bridges.py) | Score bridge n-gram candidates: H_norm × rarity |
| [`src/bridge_theorems.py`](src/bridge_theorems.py) | Find theorems with premises spanning multiple pure clusters |
| [`src/edit_distance_bridges.py`](src/edit_distance_bridges.py) | Find structurally similar proofs across cluster pairs |
| [`src/depth_filter.py`](src/depth_filter.py) | Filter bridge theorems by cross-cluster premise specificity |
| [`src/validate_bridges.py`](src/validate_bridges.py) | Validate against 8 known historical cross-domain bridges |
| [`src/neural_complexity.py`](src/neural_complexity.py) | Tononi-Edelman complexity proxy on the cluster interaction graph |
| [`src/visualize.py`](src/visualize.py) | Generate all figures |
| [`src/run_pipeline.py`](src/run_pipeline.py) | Orchestrate all steps with caching |

---

## Data sources

| Dataset | HuggingFace ID | Contents |
|---|---|---|
| Tactic benchmark | `cat-searcher/leandojo-benchmark-4-random` | 259K tactic steps across 61K theorems; each step contains the tactic string with `<a>Name</a>` markup identifying premises used |
| Declaration corpus | `zipfit/leandojo_benchmark_4_corpus` | All Mathlib declarations with file paths; used to seed the graph with nodes that have no tactic data |

Both datasets are cached locally in `data/` on first run.

---

## Key findings

### 1. Leiden clustering recovers mathematical domain structure without supervision

Clustering the theorem dependency graph produces 138 communities, 121 of which are
cleanly labelled by a single Mathlib module (Topology, Algebra, MeasureTheory, …).
The 17 remaining mixed clusters are candidate bridge zones — they sit at intersections
of multiple domains.

> **Observation:** Proof dependencies alone encode the domain structure of mathematics.
> The clustering algorithm had no access to theorem names or module labels.

### 2. Large clusters are mixed; small clusters are pure

A clear size-purity anticorrelation exists: large clusters contain theorems from
multiple domains (high entropy), while small clusters are deeply specialised (low entropy).
This is visible in [`results/cluster_purity.png`](results/cluster_purity.png).

Top clusters by size:

| Cluster | Size | Label | Dom. fraction |
|---|---|---|---|
| C0 | 12,513 | Algebra + Analysis + Data | 23% (mixed) |
| C1 | 12,390 | MeasureTheory + Analysis + Topology | 31% (mixed) |
| C2 | 10,389 | Data + Order + Algebra | 30% (mixed) |
| C3 | 8,760 | Algebra + RingTheory + FieldTheory | 33% (mixed) |
| C4 | 8,245 | CategoryTheory | 54% (pure) |
| C5 | 6,372 | LinearAlgebra + Algebra | 31% (mixed) |

### 3. Tactic vocabulary is highly domain-specific

After abstracting tactics to proof-strategy tokens (150K → ~200 unique forms), only
6.2% of unigrams and 0.7% of 3-grams appear in ≥ 2 clusters. Mathematical domains
use genuinely different proof strategies, not just different theorem names. This is
itself a finding: an AI trained only on one domain will generalise poorly across domains.

See [`results/ngram_sweep.png`](results/ngram_sweep.png) and
[`results/ngram_sweep.json`](results/ngram_sweep.json).

### 4. Bridge theorem detection recovers 6/8 known historical connections

Using cross-cluster premise analysis followed by an in-degree depth filter (θ = 2),
our method detects:

| Known bridge | Found in clusters | Detected |
|---|---|---|
| Zorn / Choice / Well-Ordering | YES, spans Order ↔ SetTheory | Loose |
| Galois ↔ Covering spaces | YES | Loose (Grothendieck Galois theorem) |
| Stone duality | YES | Loose |
| Pontryagin duality | YES | Loose |
| Gromov-Hausdorff | YES, spans metric geometry + topology | **Strict** (θ≤2) |
| Spectral ↔ Representation theory | YES | **Strict** (θ≤2) |
| Fixed-point theorems | YES (but undetected) | No — conceptual bridge only |
| Lie algebras ↔ Lie groups | Partial (underrepresented) | No |

**Key insight:** our method detects *operational bridges* (theorems that literally import
results from another domain) but not *conceptual bridges* (same proof structure via
different starting points). The Brouwer ↔ Lawvere fixed-point correspondence is a
conceptual bridge; our edit-distance analysis is needed for that class.

See [`results/bridge_theorems_filtered.json`](results/bridge_theorems_filtered.json),
[`results/validation.json`](results/validation.json).

### 5. LinearAlgebra is the "lingua franca" of proof strategies

Edit-distance analysis across the top 30 clusters finds LinearAlgebra in 20 of the
top 30 most structurally-similar cross-cluster pairs. Proofs in analysis, algebra,
geometry, topology, and category theory all look structurally similar to linear algebra
proofs. This is non-trivial: LinearAlgebra being central in the *dependency* graph is
obvious (it is foundational), but centrality in the *tactic strategy* graph is a new
finding.

See [`results/edit_distance_bridges.json`](results/edit_distance_bridges.json).

### 6. SetTheory has the highest neural complexity

Computing a Tononi-Edelman neural complexity proxy (integration × differentiation)
on the cluster interaction graph:

- **SetTheory** (CN = 1.037): connects broadly to many domains AND maintains a
  specialised proof vocabulary (transfinite induction, cardinal arithmetic, well-orderings).
- **LinearAlgebra** (CN = 0.182, rank 13): high integration but low differentiation —
  it connects everywhere but its proof strategies look the same as its neighbours.

This sharpens the finding above: LinearAlgebra is the most *used* bridge language,
but SetTheory is the most *neurally complex* domain — providing the deepest integration
with the greatest specialisation.

See [`results/neural_complexity.json`](results/neural_complexity.json),
[`results/neural_complexity.png`](results/neural_complexity.png).

---

## Tactic normalization

[`src/normalize_tactics.py`](src/normalize_tactics.py) maps concrete tactic strings
to abstract proof-strategy tokens:

| Concrete tactic | Normalized form | Meaning |
|---|---|---|
| `rw [Finset.sum_comm]` | `rw [@]` | rewrite with named global theorem |
| `rw [h]` | `rw [?]` | rewrite with local hypothesis |
| `simp [Finset.mem_range, hf]` | `simp [@, ?]` | simplify using named theorem + local hyp |
| `intro x` | `intro ?` | introduce local variable |
| `apply le_antisymm` | `apply @` | apply global theorem |
| `norm_cast` | `norm_cast` | keyword kept as-is |

The key design decision: `rw [?]` and `rw [@]` are kept distinct because rewriting
with a local fact vs. a global theorem represents structurally different proof moves.

---

## Bridge scoring

[`src/bridges.py`](src/bridges.py) scores each n-gram `g` with:

```
score(g) = H_norm(g) × rarity(g)
```

where:
- `H_norm(g)` = normalised Shannon entropy of `g`'s cluster distribution — high → `g` is genuinely spread across many clusters (not dominated by one)
- `rarity(g)` = 1 − mean within-cluster relative frequency — high → `g` is locally uncommon

This penalises two failure modes: generic tactics that appear everywhere but with high
frequency (simp, intro → low rarity), and sequences that appear in many clusters but
only in one heavily (low H_norm).

---

## Directory structure

```
.
├── src/                    # all analysis code
│   ├── run_pipeline.py     # entry point — runs everything
│   ├── load_data.py
│   ├── build_graph.py
│   ├── cluster.py
│   ├── label_clusters.py
│   ├── hierarchical.py
│   ├── normalize_tactics.py
│   ├── ngram_analysis.py
│   ├── bridges.py
│   ├── bridge_theorems.py
│   ├── edit_distance_bridges.py
│   ├── depth_filter.py
│   ├── validate_bridges.py
│   ├── neural_complexity.py
│   └── visualize.py
├── data/                   # cached data (generated on first run)
│   ├── corpus.jsonl        # all Mathlib declarations
│   ├── tactic_steps.jsonl  # all tactic steps
│   ├── theorem_graph.pkl   # NetworkX dependency graph
│   ├── theorem_tactics.json
│   ├── clusters.json       # theorem → cluster_id
│   ├── cluster_labels.json # cluster metadata and labels
│   ├── cluster_summary.json
│   ├── sub_clusters.json   # theorem → [top_cluster, sub_cluster]
│   └── sub_cluster_labels.json
├── results/                # analysis outputs
│   ├── cluster_graph.png
│   ├── cluster_purity.png
│   ├── hierarchical_clusters.png
│   ├── ngram_sweep.png / .json
│   ├── bridges.json
│   ├── bridge_theorems.json
│   ├── bridge_theorems_filtered.json
│   ├── edit_distance_bridges.json
│   ├── indegree_sweep.png / .json
│   ├── neural_complexity.png / .json
│   └── validation.json
├── paper/                  # LaTeX source (ICML 2026 format)
│   └── main.tex
├── NOTES.txt               # shared research log
├── requirements.txt
└── README.md
```

---

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `datasets`, `networkx`, `leidenalg`, `python-igraph`, `numpy`,
`matplotlib`, `scikit-learn`, `scipy`, `tqdm`.

Python ≥ 3.11 recommended (uses `list[str]` type hints in function signatures).

---

## Reproducing figures

| Figure | Output file | Script |
|---|---|---|
| Cluster graph (bridge edges) | `results/cluster_graph.png` | `src/visualize.py` |
| Cluster purity scatter | `results/cluster_purity.png` | `src/visualize.py` |
| Hierarchical sub-cluster chart | `results/hierarchical_clusters.png` | `src/visualize.py` |
| N-gram cross-cluster coverage | `results/ngram_sweep.png` | `src/ngram_analysis.py` |
| Premise in-degree sweep | `results/indegree_sweep.png` | `src/depth_filter.py` |
| Neural complexity | `results/neural_complexity.png` | `src/neural_complexity.py` |

All figures are regenerated by `python3 src/run_pipeline.py` (except `indegree_sweep`
and `neural_complexity`, which must be run separately).
