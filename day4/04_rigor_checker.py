"""
Day 4, Exercise 4: Rigor Checker

A second LLM pass that acts as an adversarial peer reviewer. It reads the generated LaTeX draft and cross-references every factual claim against the original curated_library abstracts.

Two failure modes this catches:
  1. Hallucinated metrics: LLM invents numbers not in the abstract
  2. Attribution errors: LLM assigns a result from Paper A to Paper B

The checker works at the sentence level. It extracts candidate claims (sentences near a \\cite command that contain numbers or superlatives), then verifies each against its cited paper's abstract.

Run:
    python day4/04_rigor_checker.py
"""

import json
import re
import os
import sys
import importlib.util as _ilu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

def _load_module(name, rel_path):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sm = _load_module("robust_state",   "day3/01_robust_state.py")
_bb = _load_module("bibtex_builder", "day4/02_bibtex_builder.py")

load_state    = _sm.load_state
make_cite_key = _bb.make_cite_key

STATE_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")
TEX_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "related_works.tex")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rigor_report.json")


# --- Claim extraction

# A "claim" is a LaTeX sentence that:
#   (a) contains a \cite{} reference, AND
#   (b) contains a number, percentage, or a superlative/comparative word
CLAIM_PATTERN = re.compile(
    r"([A-Z][^.!?]*"
    r"(?:\\cite\{[^}]+\})[^.!?]*"
    r"(?:"
    r"\d+(?:\.\d+)?%?"
    r"|outperform|exceed|achiev|state.of.the.art|sota|novel|first|superior|improve"
    r")[^.!?]*[.!?])",
    re.IGNORECASE | re.DOTALL,
)

CITE_IN_CLAIM = re.compile(r"\\cite\{([^}]+)\}")


def extract_claims(latex_text: str) -> list[dict]:
    """
    Extract verifiable claims from a LaTeX string.
    Returns a list of {claim_text, cite_keys} dicts.
    """
    claims = []
    for match in CLAIM_PATTERN.finditer(latex_text):
        sentence = match.group(1).strip()
        cite_keys = []
        for cite_match in CITE_IN_CLAIM.finditer(sentence):
            keys = [k.strip() for k in cite_match.group(1).split(",")]
            cite_keys.extend(keys)
        if cite_keys:
            claims.append({"claim_text": sentence, "cite_keys": cite_keys})
    return claims


# --- Per-claim verification

CHECKER_SYSTEM = """\
You are a strict academic fact-checker for a PhD dissertation.

You are given:
  1. A sentence from a related works section (the "claim")
  2. The abstract(s) of the paper(s) it cites

Your job: determine whether the claim is factually supported by the abstracts.

Possible statuses:
  "supported"      - the claim is clearly supported by the abstract(s)
  "not_supported"  - the claim makes a statement the abstract does not make
  "hallucinated"   - the claim contains a specific number or fact not in the abstract
  "misattributed"  - the correct fact is present but attributed to the wrong paper
  "unverifiable"   - the abstract is too short or vague to confirm or deny

Respond with ONLY a JSON object:
{
  "status":     "supported" | "not_supported" | "hallucinated" | "misattributed" | "unverifiable",
  "confidence": <float 0.0-1.0>,
  "issue":      "<null or one sentence describing the problem>",
  "suggestion": "<null or suggested corrected sentence>"
}
"""


def verify_claim(claim: dict, library: dict, cite_key_to_id: dict) -> dict:
    """
    Verify a single claim against the abstracts of its cited papers.
    Returns the claim dict augmented with verification results.
    """
    abstracts = []
    for key in claim["cite_keys"]:
        pid = cite_key_to_id.get(key)
        if pid and pid in library:
            paper = library[pid]
            abstracts.append(
                f"Paper [{key}]: {paper['title']}\n"
                f"Abstract: {paper.get('abstract', 'No abstract available.')}"
            )
        else:
            abstracts.append(f"Paper [{key}]: Not found in library.")

    if not abstracts:
        return {**claim, "status": "unverifiable", "confidence": 0.0,
                "issue": "No cited papers found in library.", "suggestion": None}

    user_msg = (
        f"Claim:\n  {claim['claim_text']}\n\n"
        f"Cited paper abstract(s):\n" + "\n\n".join(abstracts)
    )
    messages = [
        {"role": "system", "content": CHECKER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw = chat(messages, max_tokens=300, temperature=0.1)
    try:
        result = _extract_json(raw)
    except ValueError:
        result = {
            "status": "unverifiable", "confidence": 0.0,
            "issue": f"Checker LLM returned non-JSON: {raw[:100]}", "suggestion": None,
        }
    return {**claim, **result}


def build_cite_key_to_id(library: dict) -> dict[str, str]:
    """
    Rebuild the cite_key -> paper_id mapping by re-running the same key
    generation logic as part 2. Called independently so the rigor checker
    does not require the prose generator to have run in the same process.
    """
    existing: set[str] = set()
    return {make_cite_key(paper, existing): pid for pid, paper in library.items()}


# --- Full pipeline

def run_rigor_check(latex_text: str, state: dict) -> list[dict]:
    library = state["curated_library"]
    cite_key_to_id = build_cite_key_to_id(library)

    claims = extract_claims(latex_text)
    print(f"[rigor] Found {len(claims)} verifiable claim(s) in the draft.\n")

    findings = []
    for i, claim in enumerate(claims, 1):
        print(f"  Checking claim {i}/{len(claims)}: {claim['claim_text'][:80]}...")
        result = verify_claim(claim, library, cite_key_to_id)
        status = result.get("status", "unknown")
        icon = {"supported": "[OK]", "not_supported": "[x]", "hallucinated": "[!]",
                "misattributed": "[!]", "unverifiable": "?"}.get(status, "?")
        print(f"  {icon} {status}  [conf: {result.get('confidence', 0):.2f}]")
        if result.get("issue"):
            print(f"    Issue: {result['issue']}")
        findings.append(result)

    return findings


def print_report(findings: list[dict]) -> None:
    from collections import Counter
    statuses = Counter(f["status"] for f in findings)

    print(f"\n{'=' * 60}")
    print("RIGOR CHECK REPORT")
    print(f"{'=' * 60}")
    print(f"Total claims checked: {len(findings)}")
    for status, count in statuses.most_common():
        icon = {"supported": "[OK]", "not_supported": "[x]", "hallucinated": "[!]",
                "misattributed": "[!]", "unverifiable": "?"}.get(status, "?")
        print(f"  {icon} {status}: {count}")

    problems      = [f for f in findings if f["status"] not in ("supported", "unverifiable")]
    unverifiables = [f for f in findings if f["status"] == "unverifiable"]

    if problems:
        print(f"\nProblems found ({len(problems)}):")
        for p in problems:
            print(f"\n  [{p['status'].upper()}]")
            print(f"  Claim: {p['claim_text'][:100]}")
            print(f"  Issue: {p.get('issue', '')}")
            if p.get("suggestion"):
                print(f"  Fix:   {p['suggestion']}")

    if unverifiables:
        print(f"\nUnverifiable claims ({len(unverifiables)}) - manual review needed:")
        for u in unverifiables:
            print(f"\n  [UNVERIFIABLE]")
            print(f"  Claim: {u['claim_text'][:100]}")
            print(f"  Issue: {u.get('issue', '')}")

    if not problems and not unverifiables:
        print("\n[OK] Draft appears factually consistent with library.")
    elif not problems:
        print(f"\n[!] No factual errors detected, but {len(unverifiables)} claim(s) could not be verified.")


if __name__ == "__main__":
    state = load_state(STATE_PATH)

    if not os.path.exists(TEX_PATH):
        print(f"[rigor] {TEX_PATH} not found. Run day4/03_prose_generator.py first.")
        print("[rigor] Using demo draft with a deliberate hallucination instead.\n")
        latex_text = (
            r"\section{Related Work}" + "\n\n"
            r"Hypergraph neural networks were first proposed by \cite{feng2019}, "
            r"who reported a 97.3\% accuracy on the Cora dataset. "
            r"Contrastive learning approaches \cite{you2020} have shown strong "
            r"transfer learning performance across multiple graph benchmarks. "
            r"Recent work on sparse hypergraph learning \cite{xia2023} outperforms "
            r"all prior methods by a significant margin."
        )
    else:
        with open(TEX_PATH, "r", encoding="utf-8") as f:
            latex_text = f.read()

    if not state.get("curated_library"):
        state["curated_library"] = {
            "p001": {
                "paper_id": "p001", "title": "Hypergraph Neural Networks",
                "year": 2019, "authors": ["Y. Feng"],
                "abstract": "We propose HGNN for hyperedge convolution. We report 81.2% on Cora.",
            },
            "p002": {
                "paper_id": "p002", "title": "Graph Contrastive Learning with Augmentations",
                "year": 2020, "authors": ["Y. You", "T. Chen"],
                "abstract": "GraphCL achieves strong semi-supervised results without reporting a single accuracy number on Cora.",
            },
            "p003": {
                "paper_id": "p003", "title": "Sparse Hypergraph Representation Learning",
                "year": 2023, "authors": ["W. Xia"],
                "abstract": "We propose sparse hypergraph attention. We improve over prior work by 2.1 F1 points.",
            },
        }

    findings = run_rigor_check(latex_text, state)
    print_report(findings)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print(f"\n[OK] Full report saved to: {OUTPUT_PATH}")
