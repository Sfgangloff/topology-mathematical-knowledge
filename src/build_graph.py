"""
Build:
  theorem_graph   — directed NetworkX graph; edge A→B means proof of A uses B
  theorem_tactics — dict theorem_name → ordered list of clean tactic strings

Node attributes: file_path, module (dot-separated Mathlib path fragment).

Dependency edges come from <a>Name</a> anchors extracted by load_data.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_FILE   = DATA_DIR / "theorem_graph.pkl"
TACTICS_FILE = DATA_DIR / "theorem_tactics.json"


def _module_from_path(file_path: str) -> str:
    """Convert a file path like '.../Mathlib/Topology/Basic.lean' to 'Mathlib.Topology.Basic'."""
    p = Path(file_path)
    parts = p.with_suffix("").parts
    try:
        idx = next(i for i, s in enumerate(parts) if s == "Mathlib")
        return ".".join(parts[idx:])
    except StopIteration:
        return str(p.with_suffix(""))


def build(tactic_records: list[dict], corpus: dict[str, dict]) -> tuple[nx.DiGraph, dict]:
    """Build the theorem dependency graph and per-theorem tactic sequences.

    An edge A → B means the proof of A directly references B (A depends on B).
    Nodes are seeded from the corpus first so that theorems with no tactic data
    still appear in the graph with their correct file_path and module attributes.

    Returns:
        G: directed NetworkX graph with node attrs {file_path, module}
        theorem_tactics: {theorem_name: [tactic_str, ...]} in proof order
    """
    # ── collect tactic sequences per theorem ──────────────────────────────────
    theorem_steps: dict[str, list[dict]] = defaultdict(list)
    for rec in tactic_records:
        name = rec.get("full_name", "")
        if name:
            theorem_steps[name].append(rec)

    theorem_tactics: dict[str, list[str]] = {}
    theorem_file: dict[str, str] = {}
    for name, steps in theorem_steps.items():
        theorem_tactics[name] = [s["tactic"] for s in steps if s["tactic"]]
        theorem_file[name] = steps[0].get("file_path", "")

    # ── build graph ───────────────────────────────────────────────────────────
    G = nx.DiGraph()

    # Seed nodes from corpus so even theorems with no tactic data get a node
    for name, meta in corpus.items():
        if not G.has_node(name):
            fp = meta["file_path"]
            G.add_node(name, file_path=fp, module=_module_from_path(fp))

    # Add / update nodes from tactic data (tactic data has exact file paths)
    for name, fp in theorem_file.items():
        G.add_node(name, file_path=fp, module=_module_from_path(fp))

    # Add dependency edges
    for name, steps in tqdm(theorem_steps.items(), desc="Building edges"):
        for step in steps:
            for dep in step.get("premises", []):
                if dep and dep != name:
                    if not G.has_node(dep):
                        # Premise not in corpus — add bare node
                        G.add_node(dep, file_path="", module="")
                    G.add_edge(name, dep)

    return G, theorem_tactics


def save(G: nx.DiGraph, theorem_tactics: dict):
    """Persist the graph (pickle) and tactic sequences (JSON) to data/."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    with open(TACTICS_FILE, "w") as f:
        json.dump(theorem_tactics, f)
    print(f"Graph : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges → {GRAPH_FILE}")
    print(f"Tactics: {len(theorem_tactics):,} theorems → {TACTICS_FILE}")


def load() -> tuple[nx.DiGraph, dict]:
    """Load the graph and tactic sequences from the data/ cache."""
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    with open(TACTICS_FILE) as f:
        theorem_tactics = json.load(f)
    return G, theorem_tactics


if __name__ == "__main__":
    from load_data import load_corpus, load_tactics
    corpus = load_corpus()
    records = load_tactics()
    G, theorem_tactics = build(records, corpus)
    save(G, theorem_tactics)
