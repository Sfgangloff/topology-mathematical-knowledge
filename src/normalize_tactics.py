"""
Tactic normalization: abstract away variable/hypothesis names and theorem
references, preserving proof *strategy* structure.

Transformation rules (applied left-to-right by regex):
  Qualified names  (contain '.' or start uppercase + length > 3)  →  @
  Short identifiers (lowercase, ≤ 4 chars, not a keyword)          →  ?
  Numeric literals                                                   →  #

Examples:
  rw [Finset.sum_comm]        →  rw [@]
  rw [h]                      →  rw [?]
  rw [← hx]                   →  rw [← ?]
  simp [Finset.mem_range, hf] →  simp [@, ?]
  intro x                     →  intro ?
  apply le_antisymm           →  apply @
  cases' n with n             →  cases' ? with ?
  exact ⟨h, rfl⟩              →  exact ⟨?, rfl⟩
  norm_cast                   →  norm_cast        (bare tactic, unchanged)

The result vocabulary is ~200 unique forms instead of 150,000.
"""

import re
from functools import lru_cache

# Tactic keywords that should never be replaced
KEYWORDS = {
    # bare tactics
    "simp", "ring", "linarith", "omega", "norm_num", "norm_cast", "push_cast",
    "decide", "trivial", "tauto", "aesop", "positivity", "gcongr", "polyrith",
    "constructor", "rfl", "exact", "apply", "rw", "rewrite", "simp_rw",
    "intro", "intros", "rintro", "ext", "ext1", "ext2", "funext", "congr", "convert",
    "use", "left", "right", "cases", "rcases", "obtain",
    "induction", "by_cases", "by_contra", "contrapose",
    "exfalso", "contradiction", "assumption",
    "split_ifs", "split", "subst", "clear",
    "have", "suffices", "show", "change", "refine",
    "dsimp", "simp_all", "ring_nf", "field_simp",
    "revert", "generalize", "specialize", "replace",
    "fin_cases", "interval_cases", "mod_cast",
    "classical", "haveI", "letI", "inferI", "infer_instance",
    "this", "fun", "swap",
    "abel", "group", "linear_combination",
    "with", "at", "only", "using", "all_goals", "any_goals",
    "first", "try", "repeat", "iterate",
    "push_neg", "contrapose!", "by_contra!", "norm_cast",
    "rfl", "and", "or", "not",
    # compound tactic keywords
    "cases'", "induction'", "assumption'", "simp_all_arith",
    "rotate_left", "rotate_right", "split_ifs", "substs",
    "norm_fin", "norm_int", "pull_neg", "module_cast", "simp?",
    "exact?", "apply?", "rw?", "push_cast",
    # proof-structure combinators (domain-agnostic meta-tactics)
    "tfae_have", "tfae_finish", "calc", "conv", "conv_lhs", "conv_rhs",
    "ring1", "ring_nf", "norm_cast", "push_cast", "field_simp",
    "set", "obtain", "rcases",
}

# Qualified names: dot-chains (any case) or CamelCase len≥4
_QUALIFIED = re.compile(
    r'\b(?:'
    r'[A-Za-z][A-Za-z0-9_\']*(?:\.[A-Za-z0-9_\']+)+'    # any.dot.chain
    r'|[A-Z][A-Za-z0-9_\']{3,}'                           # CamelCase len≥4
    r')\b'
)

# Lowercase snake_case theorem names: contain underscore, not a keyword
_SNAKE_NAME = re.compile(r'\b[a-z][a-z0-9\']*(?:_[a-z0-9\']+)+\b')

# Short lowercase identifier: 1-4 chars (likely variable/hypothesis)
_SHORT_ID = re.compile(r'\b([a-z][a-z0-9\']{0,3})\b')

# Numeric literal
_NUM = re.compile(r'\b\d+\b')


@lru_cache(maxsize=32768)
def normalize(tactic: str) -> str:
    """Map a concrete tactic string to an abstract proof-strategy token sequence.

    Replaces theorem names and variable names with abstract tokens (@, ?) while
    preserving tactic keywords. The result vocabulary has ~200 unique forms
    instead of 150,000, enabling meaningful cross-domain n-gram comparison.

    The key distinction preserved: `rw [?]` (rewrite with local hypothesis) vs.
    `rw [@]` (rewrite with a globally-named theorem) — these represent different
    proof moves even though the surface syntax is identical.
    """
    t = " ".join(tactic.split())  # collapse whitespace

    # Step 1: dot-chains and CamelCase → @
    t = _QUALIFIED.sub(lambda m: "@" if m.group(0) not in KEYWORDS else m.group(0), t)

    # Step 2: lowercase snake_case theorem names → @
    t = _SNAKE_NAME.sub(lambda m: "@" if m.group(0) not in KEYWORDS else m.group(0), t)

    # Step 3: short identifiers → ?
    def replace_short(m):
        word = m.group(0)
        if word in KEYWORDS:
            return word
        return "?"
    t = _SHORT_ID.sub(replace_short, t)

    # Step 3: numeric literals → #
    t = _NUM.sub("#", t)

    return t


def normalize_sequence(tactics: list[str]) -> list[str]:
    """Normalize a list of tactic strings in proof order."""
    return [normalize(t) for t in tactics]


if __name__ == "__main__":
    examples = [
        "rw [Finset.sum_comm]",
        "rw [h]",
        "rw [← hx]",
        "simp [Finset.mem_range, hf]",
        "intro x",
        "apply le_antisymm",
        "cases' n with n",
        "exact ⟨h, rfl⟩",
        "norm_cast",
        "constructor",
        "rintro ⟨y, hy⟩",
        "have h := Nat.lt_of_lt_pred h",
        "simp only [Matrix.mul_fin_two, h]",
        "exact hf.continuous.measurable",
        "funext x",
        "ext1",
        "apply Finset.sum_congr rfl",
        "intro j hj",
        "classical infer_instance",
        "split_ifs with h",
        "by_contra! h",
    ]
    print(f"{'Original':<50}  →  {'Normalised'}")
    print("-" * 80)
    for ex in examples:
        print(f"{ex:<50}  →  {normalize(ex)}")
