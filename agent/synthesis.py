"""
Synthesis pipeline: cluster -> LaTeX prose -> rigor check.
Self-contained — no imports from day4/.
"""

import json
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

# =============================================================================
# Stage 1: Theme clustering
# =============================================================================

_CLUSTER_SCHEMA = """
{
  "clusters": [
    {
      "name": "<3-6 word cluster name>",
      "description": "<one sentence: what unifies these papers>",
      "paper_ids": ["<paper_id>", ...]
    }
  ],
  "suggested_reading_order": ["<paper_id>", ...]
}
"""


def cluster_library(state: dict) -> dict:
    """Group curated_library papers into 2-4 thematic clusters."""
    library = state["curated_library"]
    if not library:
        return state
    if len(library) < 3:
        state["narrative_outline"] = {
            "clusters": [{"name": "Related Work", "description": "",
                          "paper_ids": list(library)}],
            "suggested_reading_order": list(library),
        }
        return state

    summaries = "\n".join(
        f"  - ID: {pid}\n    Title: {p['title']}\n    Abstract: {(p.get('abstract') or '')[:200]}"
        for pid, p in library.items()
    )
    valid_ids = ", ".join(f'"{pid}"' for pid in library)
    system = (
        f'Group these papers into 2-4 thematic clusters for a related works section on: '
        f'"{state["target_topic"]}"\n\n'
        f'Rules: every paper in exactly one cluster; valid IDs: [{valid_ids}]\n\n'
        f'Your response must be a JSON object and nothing else.\n'
        f'Start your response with {{ and end it with }}.\n'
        f'Do not include any explanation, preamble, or code fences.\n\n'
        f'Schema:\n{_CLUSTER_SCHEMA}'
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Papers:\n{summaries}"}]
    print(f"[clusterer] Clustering {len(library)} papers...")
    raw = chat(messages, max_tokens=2048, temperature=0.3)
    try:
        outline = _extract_json(raw)
    except ValueError:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            "Output only the JSON object. Begin with { on the very first character. "
            "End with }. No markdown, no explanation, no code fence."})
        raw = chat(messages, max_tokens=1024, temperature=0.0)
        outline = _extract_json(raw)

    # Repair: push any unassigned papers into the last cluster
    assigned = {pid for c in outline["clusters"] for pid in c.get("paper_ids", [])}
    unassigned = [pid for pid in library if pid not in assigned]
    if unassigned and outline["clusters"]:
        outline["clusters"][-1]["paper_ids"].extend(unassigned)

    state["narrative_outline"] = outline
    return state


# =============================================================================
# Stage 2: BibTeX + LaTeX prose
# =============================================================================

def _make_cite_key(paper: dict, existing: set[str]) -> str:
    authors = paper.get("authors", [])
    if authors:
        parts = re.split(r"[\s,]+", authors[0].strip())
        last = re.sub(r"[^a-zA-Z]", "", parts[-1]).lower()
    else:
        last = "unknown"
    year = str(paper.get("year", "0000"))[:4]
    base = f"{last}{year}"
    key, n = base, ord("b")
    while key in existing:
        key = base + chr(n); n += 1
    existing.add(key)
    return key


def _escape(text: str) -> str:
    for ch, esc in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                    ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        text = text.replace(ch, esc)
    return text


def _bib_entry(paper: dict, key: str) -> str:
    authors = " and ".join(_escape(a) for a in paper.get("authors", ["Unknown"]))
    doi, arxiv = paper.get("doi", ""), paper.get("arxiv_id", "")
    notes = ([f"arXiv:{arxiv}"] if arxiv else []) + ([f"DOI:{doi}"] if doi else [])
    entry_type = "@article" if doi else "@inproceedings"
    lines = [f"{entry_type}{{{key},",
             f"  author  = {{{authors}}},",
             f"  title   = {{{_escape(paper.get('title', 'Unknown'))}}},",
             f"  year    = {{{paper.get('year', '0000')}}},"]
    if notes:
        lines.append(f"  note    = {{{'; '.join(notes)}}},")
    lines.append("}")
    return "\n".join(lines)


def _build_bib(library: dict) -> tuple[str, dict[str, str]]:
    existing: set[str] = set()
    key_map: dict[str, str] = {}
    entries: list[str] = []
    for pid, paper in library.items():
        k = _make_cite_key(paper, existing)
        key_map[pid] = k
        entries.append(_bib_entry(paper, k))
    return "\n\n".join(entries), key_map


_PROSE_SYSTEM = """\
You are an expert academic writer writing a related works section for a PhD dissertation.
Rules:
  - Use only the cite keys from the provided map. Do not invent new ones.
  - Include \\cite{key} inline. Multiple cites: \\cite{key1,key2}.
  - Do NOT write \\section{}, \\subsection{}, or \\begin{}/\\end{}.
  - Academic register, no first person.
  - Return ONLY the LaTeX paragraph text.
"""


def generate_latex(state: dict) -> tuple[str, str]:
    """
    Generate (latex_content, bib_content) from state.
    Requires state['narrative_outline'] (run cluster_library first).
    """
    library = state["curated_library"]
    outline = state.get("narrative_outline")
    if not library:
        raise ValueError("curated_library is empty.")
    if not outline:
        raise ValueError("narrative_outline is missing. Run cluster_library first.")

    print("[synthesis] Step 1: Building references.bib...")
    bib_content, key_map = _build_bib(library)
    print(f"  {len(key_map)} entries.")

    summaries = "\n".join(
        f"  {key_map.get(pid, pid)}: {library[pid]['title']} ({library[pid].get('year','?')}): "
        f"{(library[pid].get('abstract') or '')[:200]}"
        for pid in library if pid in key_map
    )
    outline_text = "\n\n".join(
        f"Sub-section: {c['name']}\n{c.get('description', '')}\n" +
        "\n".join(f"  - \\cite{{{key_map[pid]}}}: {library[pid]['title']}"
                  for pid in c.get("paper_ids", []) if pid in library and pid in key_map)
        for c in outline.get("clusters", [])
    )
    user_msg = (
        f"Research topic: {state['target_topic']}\n\n"
        f"Outline:\n{outline_text}\n\n"
        f"Citation keys:\n{summaries}"
    )

    print("[synthesis] Step 2: Generating prose...")
    prose = chat([{"role": "system", "content": _PROSE_SYSTEM},
                  {"role": "user", "content": user_msg}],
                 max_tokens=2000, temperature=0.4)

    latex_content = (
        "% Auto-generated by Bibliographer\n"
        f"% Topic: {state['target_topic']}\n\n"
        "\\section{Related Work}\n"
        "\\label{sec:related-work}\n\n"
        + prose + "\n"
    )
    return latex_content, bib_content


# =============================================================================
# Stage 3: Rigor check
# =============================================================================

_CLAIM_RE = re.compile(
    r"([A-Z][^.!?]*(?:\\cite\{[^}]+\})[^.!?]*"
    r"(?:\d+(?:\.\d+)?%?|outperform|exceed|achiev|state.of.the.art|sota|novel|first|superior|improve)"
    r"[^.!?]*[.!?])",
    re.IGNORECASE | re.DOTALL,
)
_CITE_RE = re.compile(r"\\cite\{([^}]+)\}")

_CHECKER_SYSTEM = """\
You are a strict academic fact-checker.
Given a claim from a related works section and the abstract(s) of the cited paper(s),
determine whether the claim is factually supported.

Respond ONLY with JSON:
{
  "status":     "supported"|"not_supported"|"hallucinated"|"misattributed"|"unverifiable",
  "confidence": <0.0-1.0>,
  "issue":      "<null or one sentence>",
  "suggestion": "<null or corrected sentence>"
}
"""


def _cite_key_to_id(library: dict) -> dict[str, str]:
    existing: set[str] = set()
    return {_make_cite_key(p, existing): pid for pid, p in library.items()}


def run_rigor_check(latex_text: str, state: dict) -> list[dict]:
    """Verify factual claims in latex_text against library abstracts."""
    library = state["curated_library"]
    key_to_id = _cite_key_to_id(library)

    claims = []
    for m in _CLAIM_RE.finditer(latex_text):
        sentence = m.group(1).strip()
        keys = [k.strip() for cm in _CITE_RE.finditer(sentence) for k in cm.group(1).split(",")]
        if keys:
            claims.append({"claim_text": sentence, "cite_keys": keys})

    print(f"[rigor] Found {len(claims)} verifiable claim(s).\n")
    findings = []
    for i, claim in enumerate(claims, 1):
        abstracts = []
        for k in claim["cite_keys"]:
            pid = key_to_id.get(k)
            if pid and pid in library:
                p = library[pid]
                abstracts.append(f"[{k}] {p['title']}\nAbstract: {p.get('abstract','N/A')}")
            else:
                abstracts.append(f"[{k}] Not found in library.")
        if not abstracts:
            findings.append({**claim, "status": "unverifiable", "confidence": 0.0,
                              "issue": "No cited papers found.", "suggestion": None})
            continue
        user_msg = f"Claim:\n  {claim['claim_text']}\n\nAbstracts:\n" + "\n\n".join(abstracts)
        raw = chat([{"role": "system", "content": _CHECKER_SYSTEM},
                    {"role": "user", "content": user_msg}],
                   max_tokens=300, temperature=0.1)
        try:
            result = _extract_json(raw)
        except ValueError:
            result = {"status": "unverifiable", "confidence": 0.0,
                      "issue": f"Checker returned non-JSON: {raw[:80]}", "suggestion": None}
        print(f"  [{i}/{len(claims)}] {result.get('status','?')} "
              f"(conf {result.get('confidence', 0):.2f}) — {claim['claim_text'][:60]}...")
        findings.append({**claim, **result})
    return findings


def print_report(findings: list[dict]) -> None:
    statuses = Counter(f["status"] for f in findings)
    print(f"\n{'=' * 60}\nRIGOR CHECK REPORT\n{'=' * 60}")
    print(f"Total claims checked: {len(findings)}")
    icons = {"supported": "[OK]", "not_supported": "[x]",
             "hallucinated": "[!]", "misattributed": "[!]", "unverifiable": "?"}
    for s, n in statuses.most_common():
        print(f"  {icons.get(s,'?')} {s}: {n}")

    problems      = [f for f in findings if f["status"] not in ("supported", "unverifiable")]
    unverifiables = [f for f in findings if f["status"] == "unverifiable"]

    if problems:
        print(f"\nProblems ({len(problems)}):")
        for p in problems:
            print(f"\n  [{p['status'].upper()}] {p['claim_text'][:100]}")
            if p.get("issue"):
                print(f"  Issue: {p['issue']}")
            if p.get("suggestion"):
                print(f"  Fix:   {p['suggestion']}")

    if unverifiables:
        print(f"\nUnverifiable ({len(unverifiables)}) - manual review needed:")
        for u in unverifiables:
            print(f"\n  {u['claim_text'][:100]}")
            if u.get("issue"):
                print(f"  Issue: {u['issue']}")

    if not problems and not unverifiables:
        print("\n[OK] Draft appears factually consistent with library.")
    elif not problems:
        print(f"\n[!] No factual errors, but {len(unverifiables)} claim(s) could not be verified.")
