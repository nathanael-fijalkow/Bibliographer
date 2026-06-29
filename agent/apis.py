"""
Academic API wrappers used by the agent.
Semantic Scholar (primary) with arXiv as a fallback.
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import os

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

REQUEST_DELAY   = 1.0 if not _api_key else 0.5
ABSTRACT_MAX_CHARS = 400
MAX_AUTHORS     = 3

_S2_BASE    = "https://api.semanticscholar.org/graph/v1"
_ARXIV_BASE = "http://export.arxiv.org/api/query"
_FIELDS     = "paperId,title,year,authors,abstract,citationCount,externalIds"
_REF_FIELDS = "paperId,title,year,authors,abstract,citationCount"


def _get(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    headers = {"x-api-key": _api_key} if _api_key else {}
    time.sleep(REQUEST_DELAY)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                parsed = json.loads(r.read().decode())
                return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = backoff ** (attempt + 1)
                print(f"  [API] Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif e.code == 404:
                return {}
            else:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
    return {}


def _normalise(raw: dict) -> dict:
    abstract = raw.get("abstract") or ""
    authors  = raw.get("authors") or []
    return {
        "paper_id":       raw.get("paperId", ""),
        "title":          raw.get("title", "Unknown Title"),
        "year":           raw.get("year"),
        "authors":        [a["name"] for a in authors[:MAX_AUTHORS]],
        "abstract":       abstract[:ABSTRACT_MAX_CHARS],
        "citation_count": raw.get("citationCount", 0),
        "doi":            (raw.get("externalIds") or {}).get("DOI"),
        "arxiv_id":       (raw.get("externalIds") or {}).get("ArXiv"),
    }


def fetch_paper(paper_id: str) -> dict:
    raw = _get(f"{_S2_BASE}/paper/{paper_id}?fields={_FIELDS}")
    return _normalise(raw) if raw else {}


def get_references(paper_id: str, max_refs: int = 20) -> list[dict]:
    params = urllib.parse.urlencode({"limit": min(max_refs, 100), "fields": _REF_FIELDS})
    data = _get(f"{_S2_BASE}/paper/{paper_id}/references?{params}")
    refs = [_normalise(item.get("citedPaper", {}))
            for item in (data.get("data") or [])
            if item.get("citedPaper", {}).get("paperId")]
    refs.sort(key=lambda p: p["citation_count"], reverse=True)
    return refs[:max_refs]


def get_citations(paper_id: str, max_cites: int = 20) -> list[dict]:
    params = urllib.parse.urlencode({"limit": min(max_cites, 100), "fields": _REF_FIELDS})
    data = _get(f"{_S2_BASE}/paper/{paper_id}/citations?{params}")
    cites = [_normalise(item.get("citingPaper", {}))
             for item in (data.get("data") or [])
             if item.get("citingPaper", {}).get("paperId")]
    cites.sort(key=lambda p: p["citation_count"], reverse=True)
    return cites[:max_cites]


def _search_s2(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"query": query, "limit": min(max_results, 100), "fields": _FIELDS})
    print(f"  [S2] Searching: {query!r}")
    data = _get(f"{_S2_BASE}/paper/search?{params}")
    return [_normalise(p) for p in (data.get("data") or [])]


def _search_arxiv(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": max_results})
    print(f"  [arXiv] Searching: {query!r}")
    req = urllib.request.Request(f"{_ARXIV_BASE}?{params}")
    with urllib.request.urlopen(req, timeout=15) as r:
        root = ET.fromstring(r.read().decode())
    ns = "http://www.w3.org/2005/Atom"
    papers = []
    for e in root.findall(f"{{{ns}}}entry"):
        arxiv_id = (e.findtext(f"{{{ns}}}id") or "").split("/abs/")[-1]
        authors  = [a.findtext(f"{{{ns}}}name") or "" for a in e.findall(f"{{{ns}}}author")]
        abstract = (e.findtext(f"{{{ns}}}summary") or "").strip().replace("\n", " ")
        papers.append({
            "paper_id": f"arxiv:{arxiv_id}", "title": (e.findtext(f"{{{ns}}}title") or "").strip(),
            "year": (e.findtext(f"{{{ns}}}published") or "")[:4],
            "authors": authors[:MAX_AUTHORS], "abstract": abstract[:ABSTRACT_MAX_CHARS],
            "citation_count": 0, "doi": None, "arxiv_id": arxiv_id,
        })
    return papers


def search_papers(query: str, max_results: int = 10) -> dict:
    """Search S2 then fall back to arXiv. Returns {source, query, results}."""
    try:
        results = _search_s2(query, max_results)
        if results:
            return {"source": "semantic_scholar", "query": query, "results": results}
    except Exception as exc:
        print(f"  [S2] Error: {exc}. Falling back to arXiv.")
    results = _search_arxiv(query, max_results)
    return {"source": "arxiv", "query": query, "results": results}
