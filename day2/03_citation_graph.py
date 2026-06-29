"""
Day 2, Exercise 3: Citation Graph Traversal

The graph traversal tool. Given a validated paper in the curated library,
fetch its references (backward) and/or its citing papers (forward), then
add new candidates to the discovery queue.

This is how the agent autonomously discovers papers it was never told about -
by following the citation network from papers it already trusts.

Key mechanics:
  - Backward = "what did this paper build on?" (foundational work)
  - Forward  = "who cited this paper?" (follow-on / SOTA work)
  - already_seen() prevents re-adding papers already processed

Run:
    python day2/03_citation_graph.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Numbered filenames can't be imported with `import` directly.
# importlib lets us load them by file path.
import importlib.util as _ilu

def _load_day2_module(name):
    spec = _ilu.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "01_api_tools.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_api = _load_day2_module("api_tools")
get_references = _api.get_references
get_citations = _api.get_citations
search_papers = _api.search_papers


# --- Deduplication helpers

def already_seen(paper_id: str, state: dict) -> bool:
    """
    True if this paper_id is already in curated_library, discovery_queue,
    or blacklist. Prevents pointless re-evaluation.
    """
    return (
        paper_id in state["curated_library"]
        or paper_id in state["blacklist"]
        or any(p["paper_id"] == paper_id for p in state["discovery_queue"])
    )


def add_to_queue(papers: list[dict], state: dict, source_paper_id: str) -> int:
    """
    Add papers to discovery_queue, skipping any already seen.
    Returns the number of papers actually added.
    """
    added = 0
    for paper in papers:
        pid = paper.get("paper_id", "")
        if pid and not already_seen(pid, state):
            state["discovery_queue"].append({
                **paper,
                "discovered_via": source_paper_id,
            })
            added += 1
    return added


# --- Graph traversal tool

def traverse_citations(
    paper_id: str,
    direction: str,
    state: dict,
    max_results: int = 15,
) -> dict:
    """
    Expand the citation graph from a paper already in curated_library.

    Args:
        paper_id:    Semantic Scholar paper ID.
        direction:   "backward" (references) or "forward" (citing papers).
        state:       The agent's mutable state dict.
        max_results: Maximum papers to fetch from the API.

    Returns:
        A summary observation dict for injection into the ReAct context.
    """
    if paper_id not in state["curated_library"]:
        return {
            "error": f"Paper '{paper_id}' is not in the curated library. "
                     "Only traverse papers that have been validated."
        }

    if direction not in ("forward", "backward"):
        return {"error": f"direction must be 'forward' or 'backward', got '{direction}'"}

    source_title = state["curated_library"][paper_id].get("title", paper_id)
    print(f"  [Graph] {direction.upper()} traversal from: {source_title[:60]}")

    try:
        if direction == "backward":
            papers = get_references(paper_id, max_refs=max_results)
        else:
            papers = get_citations(paper_id, max_cites=max_results)
    except Exception as exc:
        return {"error": f"API call failed: {exc}"}

    added = add_to_queue(papers, state, source_paper_id=paper_id)

    return {
        "direction": direction,
        "source_paper": paper_id,
        "papers_fetched": len(papers),
        "papers_added_to_queue": added,
        "queue_size_now": len(state["discovery_queue"]),
        "sample": [p["title"] for p in papers[:3]],
    }


# --- BFS utility: expand multiple hops

def bfs_expand(state: dict, max_depth: int = 2, direction: str = "backward") -> dict:
    """
    Breadth-first expansion of the citation graph starting from every paper
    currently in curated_library.

    max_depth=1 fetches direct references.
    max_depth=2 fetches references of references (much larger queue).
    """
    frontier = list(state["curated_library"].keys())
    visited = set(frontier)
    total_added = 0

    for depth in range(1, max_depth + 1):
        print(f"\n  [BFS] Depth {depth}: expanding {len(frontier)} paper(s)...")
        next_frontier = []

        for paper_id in frontier:
            result = traverse_citations(paper_id, direction, state)
            if "error" not in result:
                total_added += result["papers_added_to_queue"]
                # The newly queued papers become the next frontier
                for p in state["discovery_queue"]:
                    pid = p["paper_id"]
                    if pid not in visited:
                        visited.add(pid)
                        next_frontier.append(pid)

        frontier = next_frontier

    return {
        "bfs_depth": max_depth,
        "total_added_to_queue": total_added,
        "queue_size": len(state["discovery_queue"]),
    }


# --- Demo

if __name__ == "__main__":
    # Fetch the seed paper directly by its arXiv ID.
    # S2 accepts "arXiv:XXXX.XXXXX" as a valid paper identifier - much more
    # reliable than a text search, which can surface the wrong paper.
    # "Hypergraph Neural Networks" - Feng et al., AAAI 2019, arXiv:1809.09401
    ARXIV_ID = "arXiv:1809.09401"
    print(f"Fetching seed paper: {ARXIV_ID}\n")

    seed_paper = _api.fetch_paper(ARXIV_ID)
    if not seed_paper or not seed_paper.get("paper_id"):
        print(f"Could not fetch {ARXIV_ID}. Check your network / SEMANTIC_SCHOLAR_API_KEY.")
        raise SystemExit(1)

    paper_id = seed_paper["paper_id"]
    print(f"Found: [{seed_paper['year']}] {seed_paper['title']}")
    print(f"  S2 ID: {paper_id}\n")

    # Build a minimal state with this paper already validated.
    state = {
        "target_topic": "Contrastive learning over sparse hypergraphs",
        "curated_library": {
            paper_id: {**seed_paper, "relevance_score": 0.88},
        },
        "discovery_queue": [],
        "blacklist": [],
    }

    # Backward traversal - what did this paper cite?
    print("Starting backward traversal (papers this work cites)...\n")
    result = traverse_citations(paper_id, "backward", state, max_results=20)

    print(f"\nTraversal result:")
    print(json.dumps(result, indent=2))

    print(f"\nDiscovery queue now has {len(state['discovery_queue'])} papers:")
    for p in state["discovery_queue"][:5]:
        print(f"  [{p.get('year', '?')}] {p['title'][:60]}")
