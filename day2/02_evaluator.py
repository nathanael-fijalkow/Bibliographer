"""
Day 2, Exercise 2: Paper Relevance Evaluator

The agent cannot add every paper it finds to the curated library. This module
provides an LLM-based evaluation step: given a paper's abstract and a target
topic, return a relevance score (0.0-1.0) and a structured verdict.

The evaluation uses a structured scorecard prompt - the LLM must reason about
specific criteria and produce a JSON result with numeric subscores.

Run:
    python day2/02_evaluator.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

# --- Relevance threshold
#
# Papers scoring >= this go into curated_library.
# Papers scoring < this go into blacklist.
# Tune this for your topic: lower = broader review, higher = tighter review.

RELEVANCE_THRESHOLD = 0.65


# --- Evaluation prompt

EVALUATOR_SYSTEM = """\
You are a strict academic peer reviewer. You evaluate whether a paper is
relevant to a given research topic.

You score the paper on three criteria, each from 0.0 to 1.0:

  1. topical_overlap     - Does the paper directly address the research topic?
                           0.0 = completely unrelated, 1.0 = exact match
  2. methodological_fit  - Do the methods used align with the topic's approach?
                           0.0 = incompatible methods, 1.0 = perfect fit
  3. contribution_value  - Would this paper be cited in a related works section?
                           0.0 = definitely not, 1.0 = must-cite

The final score is the average of the three subscores.

Your response must be a JSON object and nothing else.
Start your response with { and end it with }.
Do not include any explanation, reasoning, preamble, or code fences.

{
  "topical_overlap":     <float 0.0-1.0>,
  "methodological_fit":  <float 0.0-1.0>,
  "contribution_value":  <float 0.0-1.0>,
  "score":               <float - average of the three>,
  "verdict":             "highly_relevant" | "relevant" | "borderline" | "irrelevant",
  "reason":              "<one sentence justification>"
}

Verdict thresholds:
  score >= 0.80 -> "highly_relevant"
  score >= 0.65 -> "relevant"
  score >= 0.40 -> "borderline"
  score  < 0.40 -> "irrelevant"
"""


def evaluate_paper(paper: dict, target_topic: str) -> dict:
    """
    Evaluate a paper dict (must have at least 'title' and 'abstract') against
    the target topic.

    Returns a dict with 'score', 'verdict', 'reason', and the three subscores.
    """
    title = paper.get("title", "Unknown Title")
    abstract = paper.get("abstract", "No abstract available.")
    year = paper.get("year", "unknown year")
    authors = paper.get("authors", [])
    author_str = ", ".join(authors) if authors else "Unknown"

    user_message = f"""\
Research topic: {target_topic}

Paper to evaluate:
  Title:    {title}
  Authors:  {author_str}
  Year:     {year}
  Abstract: {abstract}

Score this paper's relevance to the research topic.
"""

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    # 2048 tokens: large models (72B) can generate lengthy preamble before
    # the JSON even when instructed not to, exhausting smaller budgets.
    # Retry once with an even more explicit instruction if still truncated.
    raw = chat(messages, max_tokens=2048, temperature=0.1)
    try:
        result = _extract_json(raw)
    except ValueError:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            "Output only the JSON object. Begin with { on the very first "
            "character. End with }. No markdown, no explanation, no code fence."})
        raw = chat(messages, max_tokens=1024, temperature=0.0)
        result = _extract_json(raw)

    # Validate and clamp subscores
    for field in ("topical_overlap", "methodological_fit", "contribution_value"):
        val = result.get(field, 0.0)
        result[field] = max(0.0, min(1.0, float(val)))

    # Recompute score from subscores in case the model made arithmetic errors
    subscores = [result["topical_overlap"], result["methodological_fit"], result["contribution_value"]]
    result["score"] = round(sum(subscores) / len(subscores), 3)

    # Enforce verdict thresholds regardless of what the model said
    score = result["score"]
    if score >= 0.80:
        result["verdict"] = "highly_relevant"
    elif score >= 0.65:
        result["verdict"] = "relevant"
    elif score >= 0.40:
        result["verdict"] = "borderline"
    else:
        result["verdict"] = "irrelevant"

    return result


def route_paper(paper: dict, evaluation: dict, state: dict) -> dict:
    """
    Based on the evaluation verdict, add the paper to either curated_library
    or blacklist. Mutates and returns the state dict.
    """
    paper_id = paper.get("paper_id", paper.get("title", "unknown"))
    score = evaluation["score"]
    verdict = evaluation["verdict"]

    if score >= RELEVANCE_THRESHOLD:
        state["curated_library"][paper_id] = {
            **paper,
            "relevance_score": score,
            "verdict": verdict,
            "reason": evaluation.get("reason", ""),
        }
        print(f"  [OK] ADDED to library    [{score:.2f}] {paper['title'][:60]}")
    else:
        state["blacklist"].append(paper_id)
        print(f"  [x] BLACKLISTED         [{score:.2f}] {paper['title'][:60]}")

    return state


# --- Demo

if __name__ == "__main__":
    target_topic = "Contrastive learning over sparse hypergraphs"

    test_papers = [
        {
            "paper_id": "s2_001",
            "title": "Hypergraph Neural Networks",
            "year": 2019,
            "authors": ["Yifan Feng", "Haoxuan You", "Zizhao Zhang"],
            "abstract": (
                "In this paper, we present a hypergraph neural network (HGNN) framework for "
                "data representation learning, which can encode high-order data correlation "
                "in a hypergraph structure. Specially, a hyperedge convolution operation is "
                "designed to handle the data correlation during representation learning."
            ),
        },
        {
            "paper_id": "s2_002",
            "title": "Attention Is All You Need",
            "year": 2017,
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "abstract": (
                "We propose a new simple network architecture, the Transformer, based solely "
                "on attention mechanisms, dispensing with recurrence and convolutions entirely."
            ),
        },
        {
            "paper_id": "s2_003",
            "title": "Contrastive Learning on Hypergraphs for Recommendation",
            "year": 2023,
            "authors": ["Wei Xia", "Chao Huang", "Yong Xu"],
            "abstract": (
                "We propose a contrastive learning framework for hypergraph-based recommendation "
                "that exploits high-order correlations through sparse hyperedge construction. "
                "Our method outperforms state-of-the-art baselines on four benchmark datasets."
            ),
        },
    ]

    state = {"curated_library": {}, "blacklist": []}

    print(f"Target topic: {target_topic!r}\n")
    print(f"{'-' * 60}")

    for paper in test_papers:
        print(f"\nEvaluating: {paper['title']}")
        evaluation = evaluate_paper(paper, target_topic)
        print(f"  Score:    {evaluation['score']:.2f}")
        print(f"  Verdict:  {evaluation['verdict']}")
        print(f"  Reason:   {evaluation['reason']}")
        state = route_paper(paper, evaluation, state)

    print(f"\n{'=' * 60}")
    print(f"Curated library: {len(state['curated_library'])} papers")
    print(f"Blacklist:        {len(state['blacklist'])} papers")
