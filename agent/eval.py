"""
Paper relevance evaluation and deduplication for the agent.
"""

import hashlib
import re
import sys
import os
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

# ---------------------------------------------------------------------------
# Relevance evaluation
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.65

_EVALUATOR_SYSTEM = """\
You are a strict academic peer reviewer evaluating paper relevance.

Score on three criteria (0.0-1.0 each):
  1. topical_overlap    - Does the paper directly address the research topic?
  2. methodological_fit - Do the methods align with the topic's approach?
  3. contribution_value - Would this paper be cited in a related works section?

Your response must be a JSON object and nothing else.
Start your response with { and end it with }.
Do not include any explanation, preamble, or code fences.

{
  "topical_overlap":    <float>,
  "methodological_fit": <float>,
  "contribution_value": <float>,
  "score":              <average of the three>,
  "verdict":            "highly_relevant"|"relevant"|"borderline"|"irrelevant",
  "reason":             "<one sentence>"
}
"""


def evaluate_paper(paper: dict, target_topic: str) -> dict:
    """Score a paper's relevance to target_topic. Returns dict with score/verdict/reason."""
    authors = paper.get("authors", [])
    user_msg = (
        f"Research topic: {target_topic}\n\n"
        f"Title:    {paper.get('title', 'Unknown')}\n"
        f"Authors:  {', '.join(authors) if authors else 'Unknown'}\n"
        f"Year:     {paper.get('year', '?')}\n"
        f"Abstract: {paper.get('abstract', 'No abstract.')}\n\n"
        "Score this paper's relevance."
    )
    messages = [
        {"role": "system", "content": _EVALUATOR_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw = chat(messages, max_tokens=2048, temperature=0.1)
    try:
        result = _extract_json(raw)
    except ValueError:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            "Output only the JSON object. Start with { end with }. No other text."})
        raw = chat(messages, max_tokens=1024, temperature=0.0)
        result = _extract_json(raw)

    for field in ("topical_overlap", "methodological_fit", "contribution_value"):
        result[field] = max(0.0, min(1.0, float(result.get(field, 0.0))))
    subs = [result["topical_overlap"], result["methodological_fit"], result["contribution_value"]]
    result["score"] = round(sum(subs) / 3, 3)
    s = result["score"]
    result["verdict"] = ("highly_relevant" if s >= 0.80 else
                         "relevant"        if s >= 0.65 else
                         "borderline"      if s >= 0.40 else "irrelevant")
    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_STOP = {"a", "an", "the", "on", "in", "of", "for", "with", "and", "or"}


def _normalise_title(title: str) -> str:
    nfkd = unicodedata.normalize("NFKD", title)
    clean = re.sub(r"[^a-z0-9\s]", " ", nfkd.encode("ascii", "ignore").decode().lower())
    words = [w for w in clean.split() if w and w not in _STOP]
    words.sort()
    return " ".join(words)


def _fingerprint(title: str) -> str:
    return hashlib.sha1(_normalise_title(title).encode()).hexdigest()[:16]


def already_seen(paper: dict, state: dict) -> tuple[bool, str]:
    """Return (True, reason) if the paper is already in state, else (False, '')."""
    pid = paper.get("paper_id", "")
    fp  = _fingerprint(paper.get("title", ""))

    if pid and pid in state["curated_library"]:
        return True, f"'{pid}' already in library"
    if pid and pid in state.get("blacklist", []):
        return True, f"'{pid}' is blacklisted"
    queue_ids = {p.get("paper_id") for p in state.get("discovery_queue", [])}
    if pid and pid in queue_ids:
        return True, f"'{pid}' already in queue"

    lib_fps = {_fingerprint(p.get("title", "")): k for k, p in state["curated_library"].items()}
    if fp in lib_fps:
        return True, f"title fingerprint matches library paper '{lib_fps[fp]}'"
    queue_fps = {_fingerprint(p.get("title", "")) for p in state.get("discovery_queue", [])}
    if fp in queue_fps:
        return True, "title fingerprint matches a paper already in queue"

    return False, ""


def deduplicate_queue(state: dict) -> dict:
    """Remove duplicates from discovery_queue in-place. Returns a summary dict."""
    original = len(state["discovery_queue"])
    clean, removed = [], []
    lib_fps = {_fingerprint(p.get("title", "")) for p in state["curated_library"].values()}
    blacklisted = set(state.get("blacklist", []))
    seen_fps: set[str] = set()

    for paper in state["discovery_queue"]:
        pid = paper.get("paper_id", "")
        fp  = _fingerprint(paper.get("title", ""))
        if pid in blacklisted:
            removed.append((paper["title"], "blacklisted"))
        elif fp in lib_fps:
            removed.append((paper["title"], "already in library"))
        elif fp in seen_fps:
            removed.append((paper["title"], "duplicate within queue"))
        else:
            seen_fps.add(fp)
            clean.append(paper)

    state["discovery_queue"] = clean
    return {"original_size": original, "cleaned_size": len(clean),
            "removed": len(removed), "removed_reasons": removed[:5]}
