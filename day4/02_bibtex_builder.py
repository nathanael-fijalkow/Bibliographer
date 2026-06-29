"""
Day 4, Exercise 2: BibTeX Builder

Generates a standards-compliant references.bib file from the curated_library JSON. This step uses NO LLM - all formatting is deterministic Python.

Why separate from the prose generator?
  - BibTeX is structured data: exact author names, exact years, exact titles. An LLM would introduce hallucination risk on the most fact-sensitive part.
  - Citation keys must be stable and consistent across both the .bib file and the prose. Generating them in Python guarantees that.
  - The deterministic step can be run, inspected, and corrected before the expensive LLM step.

Key functions:
  make_cite_key(paper, existing_keys)
    -> LastName + Year, with b/c/... collision resolution
       e.g.: feng2019, feng2019b, feng2019c

  paper_to_bib_entry(paper, cite_key)
    -> @article or @inproceedings BibTeX block

  generate_bib_file(library)
    -> full .bib string + {paper_id: cite_key} map

Output:
  references.bib  - ready to include in your LaTeX project

Run:
    python day4/02_bibtex_builder.py
"""

import json
import re
import os
import sys
import importlib.util as _ilu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def _load_module(name, rel_path):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sm = _load_module("robust_state", "day3/01_robust_state.py")
load_state = _sm.load_state

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")
OUTPUT_DIR = os.path.dirname(os.path.dirname(__file__))


def make_cite_key(paper: dict, existing_keys: set[str]) -> str:
    """
    Generate a BibTeX citation key: LastName + Year.
    Resolves collisions by appending 'b', 'c', ... suffixes.
    Mutates existing_keys in-place so callers share the collision state.
    """
    authors = paper.get("authors", [])
    if authors:
        first_author = authors[0]
        parts = re.split(r"[\s,]+", first_author.strip())
        last_name = re.sub(r"[^a-zA-Z]", "", parts[-1]).lower()
    else:
        last_name = "unknown"

    year = str(paper.get("year", "0000"))[:4]
    base_key = f"{last_name}{year}"

    key = base_key
    suffix_ord = ord("b")
    while key in existing_keys:
        key = base_key + chr(suffix_ord)
        suffix_ord += 1

    existing_keys.add(key)
    return key


def escape_latex(text: str) -> str:
    """Escape the 9 special LaTeX characters that appear in titles/names."""
    for char, escaped in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
    ]:
        text = text.replace(char, escaped)
    return text


def paper_to_bib_entry(paper: dict, cite_key: str) -> str:
    """
    Convert a paper dict to a BibTeX entry.
    Uses @article when a DOI is present (implies journal publication),
    @inproceedings otherwise (conference / preprint default).
    """
    title  = escape_latex(paper.get("title", "Unknown Title"))
    year   = paper.get("year", "0000")
    authors = paper.get("authors", ["Unknown Author"])
    doi    = paper.get("doi", "")
    arxiv  = paper.get("arxiv_id", "")

    author_str = " and ".join(escape_latex(a) for a in authors)

    notes = []
    if arxiv:
        notes.append(f"arXiv:{arxiv}")
    if doi:
        notes.append(f"DOI:{doi}")

    entry_type = "@article" if doi else "@inproceedings"
    lines = [
        f"{entry_type}{{{cite_key},",
        f"  author  = {{{author_str}}},",
        f"  title   = {{{title}}},",
        f"  year    = {{{year}}},",
    ]
    if notes:
        lines.append(f"  note    = {{{'; '.join(notes)}}},")
    lines.append("}")
    return "\n".join(lines)


def generate_bib_file(library: dict) -> tuple[str, dict[str, str]]:
    """
    Build the full .bib file content.

    Returns:
        bib_content   - the complete BibTeX string
        cite_key_map  - {paper_id: cite_key} for use by the prose generator
    """
    existing_keys: set[str] = set()
    cite_key_map: dict[str, str] = {}
    bib_entries: list[str] = []

    for paper_id, paper in library.items():
        key = make_cite_key(paper, existing_keys)
        cite_key_map[paper_id] = key
        bib_entries.append(paper_to_bib_entry(paper, key))

    return "\n\n".join(bib_entries), cite_key_map


# --- Demo

DEMO_LIBRARY = {
    "p001": {
        "paper_id": "p001", "title": "Hypergraph Neural Networks",
        "year": 2019, "authors": ["Y. Feng", "H. You", "Z. Zhang"],
        "abstract": "HGNN framework for data representation via hyperedge convolution.",
        "citation_count": 1200,
    },
    "p002": {
        "paper_id": "p002", "title": "Self-Supervised Contrastive Learning on Graphs",
        "year": 2021, "authors": ["Y. You", "T. Chen"],
        "abstract": "GraphCL applies contrastive learning to graph data with augmentation.",
        "citation_count": 800,
    },
    "p003": {
        "paper_id": "p003", "title": "Sparse Hypergraph Representation Learning",
        "year": 2023, "authors": ["W. Xia"],
        "abstract": "Scalable hypergraph learning via sparse attention mechanisms.",
        "citation_count": 120,
    },
    "p004": {
        "paper_id": "p004", "title": "Contrastive Hypergraph Learning for Recommendation",
        "year": 2023, "authors": ["A. Liu", "B. Zhang"],
        "abstract": "Applies contrastive loss to hypergraph-based recommendation systems.",
        "citation_count": 85,
    },
    "p005": {
        "paper_id": "p005", "title": "Graph Contrastive Learning with Augmentations",
        "year": 2020, "authors": ["Y. You"],
        "abstract": "Systematic study of data augmentation strategies for graph contrastive learning.",
        "citation_count": 950,
    },
}


if __name__ == "__main__":
    state = load_state(STATE_PATH)

    if not state.get("curated_library"):
        print("[demo] Empty library - using built-in demo papers.\n")
        state["curated_library"] = DEMO_LIBRARY

    library = state["curated_library"]
    print(f"[bibtex] Building .bib file for {len(library)} papers...\n")

    bib_content, cite_key_map = generate_bib_file(library)

    print("Citation key assignments:")
    for pid, key in cite_key_map.items():
        title = library[pid]["title"][:50]
        print(f"  {key:20s}  <- {title}")

    bib_path = os.path.join(OUTPUT_DIR, "references.bib")
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib_content)

    print(f"\n[OK] Written: {bib_path}")
    print(f"\nPreview (first entry):\n{'-' * 40}")
    print(bib_content.split("\n\n")[0])
