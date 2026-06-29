"""
Day 4, Exercise 1: theme clusterer

Reads the curated_library and groups papers into 2-4 thematic clusters.
The result is stored as narrative_outline in state.json and used by the
LaTeX generator to structure the related works section.

This is Stage 1 of the multi-stage synthesis pipeline.

Run:
    python day4/01_theme_clusterer.py
"""

import json
import sys
import os
import importlib.util as _ilu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

def _load_module(name, rel_path):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sm = _load_module("robust_state", "day3/01_robust_state.py")
checkpoint = _sm.checkpoint
load_state = _sm.load_state

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")


CLUSTER_SCHEMA = """
{
  "clusters": [
    {
      "name": "<short descriptive cluster name, 3-6 words>",
      "description": "<one sentence: what unifies these papers>",
      "paper_ids": ["<paper_id_1>", "<paper_id_2>", ...]
    }
  ],
  "suggested_reading_order": ["<paper_id>", ...]
}
"""


def build_cluster_prompt(target_topic: str, library: dict) -> list[dict]:
    """
    Build the message list for the clustering call.
    We give the LLM each paper's ID, title, and a short abstract snippet.
    Full abstracts would overflow; titles alone lose too much signal.
    """
    paper_summaries = []
    for pid, paper in library.items():
        abstract = (paper.get("abstract") or "")[:200]
        paper_summaries.append(
            f"  - ID: {pid}\n    Title: {paper['title']}\n    Abstract: {abstract}"
        )

    papers_block = "\n".join(paper_summaries)
    paper_ids_block = ", ".join(f'"{pid}"' for pid in library)

    system = f"""\
You are an expert academic writing assistant specialising in literature reviews.

Your task: group the following papers into 2 to 4 thematic clusters for a
related works section on the topic: "{target_topic}"

Rules:
- Every paper must appear in exactly one cluster.
- Cluster names should reflect shared methodology or subject matter, not just keywords.
- The suggested_reading_order lists all paper IDs in the order a reader should encounter
  them when reading the section from start to finish (foundational -> recent).
- Valid paper IDs: [{paper_ids_block}]
- Respond with ONLY a valid JSON object matching this schema:
{CLUSTER_SCHEMA}
"""

    user = f"Papers to cluster:\n{papers_block}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_clusters(outline: dict, library: dict) -> list[str]:
    errors = []
    known_ids = set(library.keys())
    assigned: set[str] = set()

    for i, cluster in enumerate(outline.get("clusters", [])):
        if "name" not in cluster:
            errors.append(f"Cluster {i}: missing 'name'")
        if "paper_ids" not in cluster or not cluster["paper_ids"]:
            errors.append(f"Cluster {i}: missing or empty 'paper_ids'")
            continue
        for pid in cluster["paper_ids"]:
            if pid not in known_ids:
                errors.append(f"Cluster {i}: unknown paper_id '{pid}'")
            if pid in assigned:
                errors.append(f"Cluster {i}: paper '{pid}' assigned to multiple clusters")
            assigned.add(pid)

    unassigned = known_ids - assigned
    if unassigned:
        errors.append(f"Papers not assigned to any cluster: {unassigned}")

    return errors


def cluster_library(state: dict) -> dict:
    library = state["curated_library"]
    if not library:
        print("[clusterer] curated_library is empty. Run day2 exercises first.")
        return state

    if len(library) < 3:
        print(f"[clusterer] Only {len(library)} paper(s) in library. "
              "Using single cluster.")
        state["narrative_outline"] = {
            "clusters": [{"name": "Related Work", "description": "", "paper_ids": list(library)}],
            "suggested_reading_order": list(library),
        }
        return state

    print(f"[clusterer] Clustering {len(library)} papers...")
    messages = build_cluster_prompt(state["target_topic"], library)
    raw = chat(messages, max_tokens=1200, temperature=0.3)

    outline = _extract_json(raw)
    errors = validate_clusters(outline, library)

    if errors:
        print("[clusterer] Validation errors - attempting repair...")
        for err in errors:
            print(f"  [x] {err}")
        # Repair: assign unassigned papers to the last cluster
        assigned = {pid for c in outline["clusters"] for pid in c.get("paper_ids", [])}
        unassigned = [pid for pid in library if pid not in assigned]
        if unassigned and outline["clusters"]:
            outline["clusters"][-1]["paper_ids"].extend(unassigned)
            print(f"  Assigned {len(unassigned)} unassigned paper(s) to last cluster.")

    state["narrative_outline"] = outline
    return state


if __name__ == "__main__":
    state = load_state(STATE_PATH)

    # If the library is empty, inject demo papers so the exercise runs
    if not state["curated_library"]:
        print("[demo] Injecting sample papers for demo purposes.\n")
        state["curated_library"] = {
            "p001": {
                "paper_id": "p001", "title": "Hypergraph Neural Networks",
                "year": 2019, "authors": ["Y. Feng"],
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

    state = cluster_library(state)
    checkpoint(state, STATE_PATH)

    outline = state["narrative_outline"]
    print(f"\nNarrative outline:")
    for cluster in outline["clusters"]:
        print(f"\n  [{cluster['name']}]")
        print(f"  {cluster.get('description', '')}")
        for pid in cluster["paper_ids"]:
            title = state["curated_library"][pid]["title"]
            print(f"    - {title[:60]}")
    print(f"\nSuggested reading order: {outline.get('suggested_reading_order', [])}")
