"""
All agent tools in one place.

Each function follows the same contract:
  - Accepts keyword arguments matching what the LLM will pass
  - Returns a plain dict (the "observation" injected back into context)
  - Never raises — returns {"error": "..."} on failure

Tools are registered in TOOL_REGISTRY and described in TOOL_DESCRIPTIONS.
The loop uses these two structures to validate LLM tool calls and dispatch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.apis import search_papers as _search, get_references, get_citations
from agent.eval import evaluate_paper as _evaluate, RELEVANCE_THRESHOLD, already_seen


def search_papers(query: str, max_results: int = 8, state: dict = None) -> dict:
    """Search Semantic Scholar / arXiv and add new results to discovery_queue."""
    try:
        result = _search(query, max_results=max_results)
        papers = result.get("results", [])
        added = sum(
            1 for p in papers
            if not already_seen(p, state)[0]
            and state["discovery_queue"].append(p) is None
        )
        return {
            "source": result.get("source"), "query": query,
            "total_found": len(papers), "added_to_queue": added,
            "queue_size": len(state["discovery_queue"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def evaluate_paper(paper_id: str, state: dict = None) -> dict:
    """Pop a paper from discovery_queue, score it, route to library or blacklist."""
    target = None
    remaining = []
    for p in state["discovery_queue"]:
        if p.get("paper_id") == paper_id and target is None:
            target = p
        else:
            remaining.append(p)

    if target is None:
        if not state["discovery_queue"]:
            return {"error": "discovery_queue is empty — nothing to evaluate"}
        target = state["discovery_queue"].pop(0)
    else:
        state["discovery_queue"] = remaining

    ev = _evaluate(target, state["target_topic"])
    score, verdict = ev["score"], ev["verdict"]

    if score >= RELEVANCE_THRESHOLD:
        state["curated_library"][target["paper_id"]] = {
            **target, "relevance_score": score,
            "verdict": verdict, "reason": ev.get("reason", ""),
        }
        action = "added_to_library"
    else:
        state["blacklist"].append(target["paper_id"])
        action = "blacklisted"

    return {
        "paper_id": target["paper_id"], "title": target.get("title", ""),
        "score": score, "verdict": verdict, "action": action,
        "library_size": len(state["curated_library"]),
        "queue_remaining": len(state["discovery_queue"]),
    }


def traverse_citations(paper_id: str, direction: str = "backward", state: dict = None) -> dict:
    """Follow the citation graph from a validated paper (backward = references, forward = citing)."""
    if paper_id not in state["curated_library"]:
        return {"error": f"'{paper_id}' not in curated_library. Evaluate it first."}
    try:
        papers = get_references(paper_id, max_refs=15) if direction == "backward" \
                 else get_citations(paper_id, max_cites=15)
        added = sum(
            1 for p in papers
            if not already_seen(p, state)[0]
            and state["discovery_queue"].append({**p, "discovered_via": paper_id}) is None
        )
        return {"direction": direction, "source": paper_id,
                "fetched": len(papers), "added_to_queue": added,
                "queue_size": len(state["discovery_queue"])}
    except Exception as exc:
        return {"error": str(exc)}


def get_state_summary(state: dict = None) -> dict:
    return {
        "iteration":      state.get("iteration", 0),
        "library_size":   len(state["curated_library"]),
        "queue_size":     len(state["discovery_queue"]),
        "blacklist_size": len(state.get("blacklist", [])),
        "library_titles": [p.get("title", "?")[:50] for p in list(state["curated_library"].values())[:5]],
        "queue_titles":   [p.get("title", "?")[:50] for p in state["discovery_queue"][:3]],
    }


def finish(summary: str = "", state: dict = None) -> dict:
    return {"status": "finished", "summary": summary}


TOOL_REGISTRY = {
    "search_papers":      search_papers,
    "evaluate_paper":     evaluate_paper,
    "traverse_citations": traverse_citations,
    "get_state_summary":  get_state_summary,
    "finish":             finish,
}

TOOL_DESCRIPTIONS = [
    {"name": "search_papers",      "args": "query: str, max_results: int = 8",
     "description": "Search academic APIs for papers. Results go to discovery_queue."},
    {"name": "evaluate_paper",     "args": "paper_id: str",
     "description": "Score relevance of a paper from discovery_queue. Routes to library or blacklist."},
    {"name": "traverse_citations", "args": "paper_id: str, direction: 'backward'|'forward'",
     "description": "Crawl citation graph from a validated paper. Adds discovered papers to queue."},
    {"name": "get_state_summary",  "args": "",
     "description": "Get a summary of current state (library size, queue, etc.)."},
    {"name": "finish",             "args": "summary: str = ''",
     "description": "End the search phase. Call when library has >= 5 papers."},
]


def dispatch(tool_name: str, tool_args: dict, state: dict) -> dict:
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool '{tool_name}' does not exist.",
                "available_tools": list(TOOL_REGISTRY.keys()),
                "instruction": "Use only the tools listed above."}
    try:
        return TOOL_REGISTRY[tool_name](**tool_args, state=state)
    except TypeError as exc:
        return {"error": f"Wrong arguments for '{tool_name}': {exc}"}
