"""
Day 4, Exercise 3: Prose Generator

Writes the actual text of the related works section.

This is the second LLM step in the synthesis pipeline. The key design constraint: the LLM writes prose and nothing else. All structured outputs (citation keys, BibTeX entries, section labels) were generated deterministically and are handed to the LLM as read-only inputs.

- Citation keys like \\cite{feng2019} must match references.bib exactly. If the LLM invented them, every key would need post-hoc validation.
- By giving the LLM a pre-approved {cite_key: summary} map, we reduce hallucination to prose style rather than factual structure.
- Python assembles the final .tex file; the LLM only fills in the paragraph text, making the output auditable section by section.

Pipeline:
  1. Load the narrative_outline (clusters from part 1)
  2. Load the cite_key_map from generate_bib_file() (part 2)
  3. Build a prompt that includes the outline and the key map
  4. LLM writes the paragraph text with \\cite{} references
  5. Python wraps it in \\section{} and \\label{}

Output:
  related_works.tex  - ready to \\input{} in your dissertation

Run:
    python day4/03_prose_generator.py
"""

import os
import sys
import importlib.util as _ilu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat

def _load_module(name, rel_path):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sm = _load_module("robust_state",   "day3/01_robust_state.py")
_bb = _load_module("bibtex_builder", "day4/02_bibtex_builder.py")
_tc = _load_module("theme_clusterer","day4/01_theme_clusterer.py")

load_state      = _sm.load_state
generate_bib_file = _bb.generate_bib_file
escape_latex    = _bb.escape_latex
DEMO_LIBRARY    = _bb.DEMO_LIBRARY

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")
OUTPUT_DIR = os.path.dirname(os.path.dirname(__file__))


# --- Prompt

PROSE_SYSTEM = """\
You are an expert academic writer. You are writing the related works section
of a PhD dissertation.

You will be given:
  1. The research topic
  2. A section outline (2-4 sub-themes with paper summaries)
  3. A citation key map: {cite_key: paper_title}

Write the prose for each sub-section. Rules:
  - Use only the cite keys from the provided map. Do not invent new ones.
  - Each sub-section should be 2-4 paragraphs.
  - Include \\cite{key} inline where appropriate. Multiple cites: \\cite{key1,key2}.
  - Do NOT write \\section{}, \\subsection{}, or \\begin{}/\\end{} - only paragraph text.
  - Do NOT include a bibliography - that is generated separately.
  - Academic register: precise, impersonal, no first person.
  - Return ONLY the LaTeX prose text. No JSON, no explanation.
"""


def build_prose_prompt(topic: str, outline: dict, library: dict,
                       cite_key_map: dict) -> list[dict]:
    paper_summaries: dict[str, str] = {}
    for pid, paper in library.items():
        key = cite_key_map.get(pid, pid)
        abstract = (paper.get("abstract") or "")[:200]
        paper_summaries[key] = f"{paper['title']} ({paper.get('year', '?')}): {abstract}"

    outline_text = []
    for cluster in outline.get("clusters", []):
        cluster_papers = [
            f"  - \\cite{{{cite_key_map[pid]}}}: {library[pid]['title']}"
            for pid in cluster.get("paper_ids", [])
            if pid in library and pid in cite_key_map
        ]
        outline_text.append(
            f"Sub-section: {cluster['name']}\n"
            f"Description: {cluster.get('description', '')}\n"
            + "\n".join(cluster_papers)
        )

    cite_key_block = "\n".join(f"  {k}: {v}" for k, v in paper_summaries.items())

    user_msg = (
        f"Research topic: {topic}\n\n"
        f"Outline:\n{'-' * 40}\n" + "\n\n".join(outline_text) +
        f"\n\n{'-' * 40}\n"
        f"Citation key reference:\n{cite_key_block}"
    )

    return [
        {"role": "system", "content": PROSE_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


def generate_latex(state: dict) -> tuple[str, str]:
    """
    Full synthesis: generate prose + assemble the .tex file.
    Also generates the .bib file (calls exercise 2 internally).

    Returns:
        latex_content  - full related_works.tex string
        bib_content    - full references.bib string
    """
    library = state["curated_library"]
    outline = state.get("narrative_outline")

    if not library:
        raise ValueError("curated_library is empty. Run day2/day3 exercises first.")
    if not outline:
        raise ValueError("narrative_outline is missing. Run day4/01_theme_clusterer.py first.")

    print("[prose] Step 1: Generating references.bib (deterministic)...")
    bib_content, cite_key_map = generate_bib_file(library)
    print(f"  Generated {len(cite_key_map)} bib entries.")

    print("[prose] Step 2: Generating prose (LLM)...")
    messages = build_prose_prompt(state["target_topic"], outline, library, cite_key_map)
    prose = chat(messages, max_tokens=2000, temperature=0.4)

    latex_header = (
        "% Auto-generated by Bibliographer - Day 4\n"
        f"% Topic: {state['target_topic']}\n\n"
        "\\section{Related Work}\n"
        "\\label{sec:related-work}\n\n"
    )
    latex_content = latex_header + prose + "\n"
    return latex_content, bib_content


# --- Demo

if __name__ == "__main__":
    state = load_state(STATE_PATH)

    if not state.get("curated_library"):
        print("[demo] Empty library - injecting demo papers.\n")
        state["curated_library"] = DEMO_LIBRARY

    if not state.get("narrative_outline"):
        print("[demo] Running theme clusterer first...")
        _tc.cluster_library(state)

    latex_content, bib_content = generate_latex(state)

    tex_path = os.path.join(OUTPUT_DIR, "related_works.tex")
    bib_path = os.path.join(OUTPUT_DIR, "references.bib")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib_content)

    print(f"\n[OK] Written: {tex_path}")
    print(f"[OK] Written: {bib_path}")
    print(f"\nPreview (first 20 lines of .tex):")
    print("-" * 60)
    for line in latex_content.splitlines()[:20]:
        print(line)
