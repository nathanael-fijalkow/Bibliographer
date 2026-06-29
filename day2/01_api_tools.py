"""
Day 2, Exercise 1: Academic API Tools

Wraps the Semantic Scholar API (and arXiv as a fallback) into clean Python
functions. Each function returns a normalised paper dict with a fixed schema
to keep downstream code simple and token budgets under control.

These functions become the real tool implementations that replace yesterday's
fake stubs.

Run:
    python day2/01_api_tools.py
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
if _api_key:
    print(f"  [S2] API key loaded ({_api_key[:6]}...)")
else:
    print("  [S2] No API key found - using unauthenticated tier (very low rate limit).")
    print("       Set SEMANTIC_SCHOLAR_API_KEY in .env to increase limits.")

# Minimum seconds between consecutive S2 requests.
# Unauthenticated: ~1 req/s is safe. With key: up to 10 req/s.
REQUEST_DELAY = 1.0 if not _api_key else 0.5

# --- Token budget constants
#
# These control how much information from each paper enters the context window.
# Tune them based on your model's context length and your topic granularity.

ABSTRACT_MAX_CHARS = 400
MAX_AUTHORS = 3
MAX_RESULTS_PER_SEARCH = 10

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "http://export.arxiv.org/api/query"

PAPER_FIELDS = "paperId,title,year,authors,abstract,citationCount,externalIds"
REFERENCE_FIELDS = "paperId,title,year,authors,abstract,citationCount"

# --- HTTP helpers

def _make_request(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """
    GET a URL and return parsed JSON. Retries on rate-limit (429) and transient
    errors with exponential backoff. Always returns a dict (never None).
    """
    headers = {}
    if _api_key:
        headers["x-api-key"] = _api_key

    time.sleep(REQUEST_DELAY)  # polite inter-request delay

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                parsed = json.loads(resp.read().decode())
                # Some S2 responses are legitimately `null` (e.g. on transient
                # overload even after a 200). Treat that as empty.
                return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = backoff ** (attempt + 1)
                print(f"  [API] Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif exc.code == 404:
                print(f"  [API] 404 Not Found: {url}")
                return {}
            else:
                raise
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)

    return {}


def _truncate_paper(raw: dict) -> dict:
    """
    Convert a raw Semantic Scholar paper object into a normalised, token-budget-
    respecting dict. Only fields the agent actually needs are kept.
    """
    abstract = raw.get("abstract") or ""
    authors_raw = raw.get("authors") or []

    return {
        "paper_id": raw.get("paperId", ""),
        "title": raw.get("title", "Unknown Title"),
        "year": raw.get("year"),
        "authors": [a["name"] for a in authors_raw[:MAX_AUTHORS]],
        "abstract": abstract[:ABSTRACT_MAX_CHARS],
        "citation_count": raw.get("citationCount", 0),
        "doi": (raw.get("externalIds") or {}).get("DOI"),
        "arxiv_id": (raw.get("externalIds") or {}).get("ArXiv"),
    }


# --- Semantic Scholar tools

def search_papers_semantic_scholar(query: str, max_results: int = MAX_RESULTS_PER_SEARCH) -> list[dict]:
    """Search Semantic Scholar by keyword. Returns truncated paper list."""
    params = urllib.parse.urlencode({
        "query": query,
        "limit": min(max_results, 100),
        "fields": PAPER_FIELDS,
    })
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?{params}"

    print(f"  [S2] Searching: {query!r}")
    data = _make_request(url)
    papers = data.get("data") or []
    return [_truncate_paper(p) for p in papers]


def fetch_paper(paper_id: str) -> dict:
    """Fetch full metadata for a single paper by its Semantic Scholar ID."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}?fields={PAPER_FIELDS}"
    raw = _make_request(url)
    if not raw:
        return {}
    return _truncate_paper(raw)


def get_references(paper_id: str, max_refs: int = 20) -> list[dict]:
    """
    Backward traversal: return papers that the given paper cites.
    Sorted by citation count descending (most influential first).
    """
    params = urllib.parse.urlencode({"limit": min(max_refs, 100), "fields": REFERENCE_FIELDS})
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/references?{params}"

    data = _make_request(url)
    refs = [item.get("citedPaper", {}) for item in (data.get("data") or [])]
    refs = [_truncate_paper(r) for r in refs if r.get("paperId")]
    refs.sort(key=lambda p: p["citation_count"], reverse=True)
    return refs[:max_refs]


def get_citations(paper_id: str, max_cites: int = 20) -> list[dict]:
    """
    Forward traversal: return papers that cite the given paper.
    Sorted by citation count descending.
    """
    params = urllib.parse.urlencode({"limit": min(max_cites, 100), "fields": REFERENCE_FIELDS})
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/citations?{params}"

    data = _make_request(url)
    cites = [item.get("citingPaper", {}) for item in (data.get("data") or [])]
    cites = [_truncate_paper(c) for c in cites if c.get("paperId")]
    cites.sort(key=lambda p: p["citation_count"], reverse=True)
    return cites[:max_cites]


# --- arXiv fallback

def search_papers_arxiv(query: str, max_results: int = MAX_RESULTS_PER_SEARCH) -> list[dict]:
    """
    Search arXiv via its Atom feed API. Useful when a paper isn't in Semantic
    Scholar or for very recent preprints.
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    })
    url = f"{ARXIV_BASE}?{params}"

    print(f"  [arXiv] Searching: {query!r}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_data = resp.read().decode()

    ns = "http://www.w3.org/2005/Atom"
    root = ET.fromstring(xml_data)

    papers = []
    for entry in root.findall(f"{{{ns}}}entry"):
        arxiv_id_full = (entry.findtext(f"{{{ns}}}id") or "").split("/abs/")[-1]
        abstract = (entry.findtext(f"{{{ns}}}summary") or "").strip().replace("\n", " ")
        authors = [
            a.findtext(f"{{{ns}}}name") or ""
            for a in entry.findall(f"{{{ns}}}author")
        ]
        papers.append({
            "paper_id": f"arxiv:{arxiv_id_full}",
            "title": (entry.findtext(f"{{{ns}}}title") or "").strip(),
            "year": (entry.findtext(f"{{{ns}}}published") or "")[:4],
            "authors": authors[:MAX_AUTHORS],
            "abstract": abstract[:ABSTRACT_MAX_CHARS],
            "citation_count": 0,
            "doi": None,
            "arxiv_id": arxiv_id_full,
        })
    return papers


# --- Unified search (tries Semantic Scholar, falls back to arXiv)

def search_papers(query: str, max_results: int = MAX_RESULTS_PER_SEARCH) -> dict:
    """
    The tool callable by the agent. Tries Semantic Scholar first; falls back
    to arXiv if the result is empty or if the API is unreachable.
    """
    try:
        results = search_papers_semantic_scholar(query, max_results)
        if results:
            return {"source": "semantic_scholar", "query": query, "results": results}
    except Exception as exc:
        print(f"  [S2] Error: {exc}. Falling back to arXiv.")

    results = search_papers_arxiv(query, max_results)
    return {"source": "arxiv", "query": query, "results": results}


# --- Demo

if __name__ == "__main__":
    query = "contrastive learning hypergraph"
    print(f"Searching for: {query!r}\n")

    result = search_papers(query, max_results=3)
    print(f"Source: {result['source']}")
    print(f"Found: {len(result['results'])} papers\n")

    for paper in result["results"]:
        print(f"  [{paper['year']}] {paper['title']}")
        print(f"          Authors: {', '.join(paper['authors'])}")
        print(f"          Citations: {paper['citation_count']}")
        print(f"          Abstract: {paper['abstract'][:100]}...")
        print()
