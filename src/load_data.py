"""
Load and cache two LeanDojo HuggingFace datasets:

  corpus  — all mathlib4 declarations with their file paths
  tactics — tactic-step benchmark with referenced premises in <a>...</a> tags

Both are saved locally so subsequent runs are instant.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
STEPS_FILE = DATA_DIR / "tactic_steps.jsonl"     # one tactic step per line
CORPUS_FILE = DATA_DIR / "corpus.jsonl"           # one declaration per line

TACTICS_DATASET = "cat-searcher/leandojo-benchmark-4-random"
CORPUS_DATASET  = "zipfit/leandojo_benchmark_4_corpus"

_ANCHOR_RE = re.compile(r"<a>(.*?)</a>")


def _extract_premises(tactic: str) -> list[str]:
    """Pull theorem names out of <a>Name</a> markup in a tactic string."""
    return _ANCHOR_RE.findall(tactic)


def _clean_tactic(tactic: str) -> str:
    """Remove <a>...</a> tags, leaving just the tactic text."""
    return _ANCHOR_RE.sub(lambda m: m.group(1), tactic).strip()


# ── corpus ────────────────────────────────────────────────────────────────────

def download_corpus():
    """Download the Mathlib declaration corpus from HuggingFace and cache as JSONL.

    Each output line is a JSON record {full_name, kind, file_path} for one declaration.
    The corpus seeds the dependency graph with nodes even for theorems that have no
    tactic data in the benchmark split.
    """
    if CORPUS_FILE.exists():
        print(f"Corpus cache exists → {CORPUS_FILE}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading corpus: {CORPUS_DATASET} ...")
    ds = load_dataset(CORPUS_DATASET, split="train", trust_remote_code=True)
    with open(CORPUS_FILE, "w") as f:
        for row in tqdm(ds, desc="corpus"):
            file_path = row["path"]
            for prem in row["premises"]:
                record = {
                    "full_name": prem["full_name"],
                    "kind": prem.get("kind", ""),
                    "file_path": file_path,
                }
                f.write(json.dumps(record) + "\n")
    print(f"Corpus saved → {CORPUS_FILE}")


def load_corpus() -> dict[str, dict]:
    """Load the cached corpus. Returns {full_name: {file_path, kind}}."""
    if not CORPUS_FILE.exists():
        download_corpus()
    out = {}
    with open(CORPUS_FILE) as f:
        for line in f:
            r = json.loads(line)
            out[r["full_name"]] = {"file_path": r["file_path"], "kind": r["kind"]}
    print(f"Corpus: {len(out):,} declarations loaded.")
    return out


# ── tactic steps ──────────────────────────────────────────────────────────────

def download_tactics():
    """Download the LeanDojo tactic-step benchmark and cache as JSONL.

    Each output line is a JSON record {full_name, file_path, tactic, premises, split}.
    The raw tactic string is cleaned (anchor markup removed) and dependency names
    are extracted into the `premises` list for use as graph edges.
    All train/val/test splits are merged into a single file.
    """
    if STEPS_FILE.exists():
        print(f"Tactics cache exists → {STEPS_FILE}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading tactics: {TACTICS_DATASET} ...")
    ds = load_dataset(TACTICS_DATASET, trust_remote_code=True)
    with open(STEPS_FILE, "w") as f:
        for split in ds:
            for row in tqdm(ds[split], desc=split):
                raw_tactic = row.get("tactic", "")
                record = {
                    "full_name": row.get("full_name", ""),
                    "file_path": row.get("file_path", ""),
                    "tactic": _clean_tactic(raw_tactic),
                    "premises": _extract_premises(raw_tactic),
                    "split": split,
                }
                f.write(json.dumps(record) + "\n")
    print(f"Tactics saved → {STEPS_FILE}")


def load_tactics() -> list[dict]:
    """Load the cached tactic steps. Returns a flat list of step records."""
    if not STEPS_FILE.exists():
        download_tactics()
    records = []
    with open(STEPS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Tactics: {len(records):,} tactic steps loaded.")
    return records


# ── combined entry point ──────────────────────────────────────────────────────

def download_all():
    """Download both datasets (corpus and tactics) if not already cached."""
    download_corpus()
    download_tactics()


if __name__ == "__main__":
    download_all()
    load_corpus()
    load_tactics()
