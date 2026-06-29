"""
Day 3, Exercise 2: Human-in-the-Loop Interrupt

Research agents should not have unilateral authority over relevance decisions.
This module adds an explicit user pause: when a paper's LLM relevance score
falls in the "borderline" zone, execution stops and the researcher decides.

Key design choices:
  - Only borderline papers (configurable score range) trigger the interrupt
  - Clear papers (high/low score) pass through automatically
  - The interrupt shows the full abstract, not just the title
  - Ctrl+C saves state and exits cleanly (the run can be resumed)

Run:
    python day3/02_human_in_loop.py
"""

import json
import signal
import sys
import os
import importlib.util as _ilu

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def _load_module(name, rel_path):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sm   = _load_module("robust_state", "day3/01_robust_state.py")
_eval = _load_module("evaluator",    "day2/02_evaluator.py")

checkpoint        = _sm.checkpoint
load_state        = _sm.load_state
advance_iteration = _sm.advance_iteration
evaluate_paper    = _eval.evaluate_paper

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")

# --- Interrupt thresholds
#
# Papers with score in [BORDERLINE_LOW, BORDERLINE_HIGH) get paused for review.
# Papers with score >= BORDERLINE_HIGH are accepted automatically.
# Papers with score <  BORDERLINE_LOW are rejected automatically.

BORDERLINE_LOW  = 0.40
BORDERLINE_HIGH = 0.65


# --- SIGINT handler (Ctrl+C)

_current_state = None

def _handle_interrupt(signum, frame):
    print("\n\n[Interrupt] Saving state before exit...")
    if _current_state is not None:
        checkpoint(_current_state, STATE_PATH)
        print(f"[Interrupt] State saved at iteration {_current_state.get('iteration', 0)}.")
    print("[Interrupt] You can resume the run by restarting the script.")
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_interrupt)


# --- Decision logic

def auto_decide(paper: dict, evaluation: dict, state: dict) -> str:
    """
    Route a paper automatically when the score is outside the borderline zone.
    Returns 'accepted', 'rejected', or 'borderline'.
    """
    score = evaluation["score"]

    if score >= BORDERLINE_HIGH:
        state["curated_library"][paper["paper_id"]] = {
            **paper,
            "relevance_score": score,
            "verdict": evaluation["verdict"],
            "reason": evaluation.get("reason", ""),
            "added_by": "auto",
        }
        print(f"  AUTO [OK] [{score:.2f}] {paper['title'][:60]}")
        return "accepted"

    if score < BORDERLINE_LOW:
        state["blacklist"].append(paper["paper_id"])
        print(f"  AUTO [x] [{score:.2f}] {paper['title'][:60]}")
        return "rejected"

    return "borderline"


def human_decide(paper: dict, evaluation: dict, state: dict) -> str:
    """
    Present a borderline paper to the researcher and wait for y/n/s input.
    Returns 'accepted', 'rejected', or 'saved_for_later'.
    """
    print(f"\n{'-' * 60}")
    print(f"BORDERLINE PAPER - Your decision needed")
    print(f"{'-' * 60}")
    print(f"Title:   {paper['title']}")
    print(f"Authors: {', '.join(paper.get('authors', []))}")
    print(f"Year:    {paper.get('year', 'unknown')}")
    print(f"Score:   {evaluation['score']:.2f}  ({evaluation['verdict']})")
    print(f"Reason:  {evaluation.get('reason', '')}")
    print(f"\nAbstract:\n  {paper.get('abstract', 'No abstract available.')}")
    print()

    while True:
        choice = input("  Include this paper? [y]es / [n]o / [s]ave for later: ").strip().lower()
        if choice in ("y", "yes"):
            state["curated_library"][paper["paper_id"]] = {
                **paper,
                "relevance_score": evaluation["score"],
                "verdict": "accepted_by_human",
                "reason": evaluation.get("reason", ""),
                "added_by": "human",
            }
            print(f"  [OK] Added to library.")
            return "accepted"

        elif choice in ("n", "no"):
            state["blacklist"].append(paper["paper_id"])
            print(f"  [x] Blacklisted.")
            return "rejected"

        elif choice in ("s", "save"):
            if "pending_review" not in state:
                state["pending_review"] = []
            state["pending_review"].append({**paper, "score": evaluation["score"]})
            print(f"  [saved] Saved for later review.")
            return "saved_for_later"

        else:
            print("  Please type y, n, or s.")


def process_queue(state: dict) -> dict:
    """
    Drain the discovery_queue: evaluate each paper and route it.
    Pauses for human input on borderline papers.
    Checkpoints after every paper - the human pause can take a long time.
    """
    global _current_state
    _current_state = state

    queue = list(state["discovery_queue"])
    state["discovery_queue"] = []

    print(f"\nProcessing {len(queue)} papers from discovery queue...\n")

    stats = {"auto_accepted": 0, "auto_rejected": 0, "human_accepted": 0,
             "human_rejected": 0, "saved_for_later": 0}

    for i, paper in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] Evaluating: {paper['title'][:60]}")

        evaluation = evaluate_paper(paper, state["target_topic"])
        status = auto_decide(paper, evaluation, state)

        if status == "borderline":
            human_result = human_decide(paper, evaluation, state)
            stats[f"human_{human_result}"] = stats.get(f"human_{human_result}", 0) + 1
        elif status == "accepted":
            stats["auto_accepted"] += 1
        elif status == "rejected":
            stats["auto_rejected"] += 1

        checkpoint(state, STATE_PATH)

    print(f"\n{'=' * 60}")
    print("Queue processing complete.")
    print(f"  Auto accepted:   {stats['auto_accepted']}")
    print(f"  Auto rejected:   {stats['auto_rejected']}")
    print(f"  Human accepted:  {stats.get('human_accepted', 0)}")
    print(f"  Human rejected:  {stats.get('human_rejected', 0)}")
    print(f"  Saved for later: {stats.get('saved_for_later', 0)}")
    print(f"  Library size now: {len(state['curated_library'])}")

    return state


# --- Demo

if __name__ == "__main__":
    state = load_state(STATE_PATH)

    state["discovery_queue"] = [
        {
            "paper_id": "test_high",
            "title": "Contrastive Learning on Sparse Hypergraphs via Dual-Channel Encoding",
            "year": 2024,
            "authors": ["Alice Zhang", "Bob Chen"],
            "abstract": "We propose a contrastive self-supervised method specifically designed for sparse hypergraph structures, achieving state-of-the-art results on node classification and link prediction benchmarks.",
        },
        {
            "paper_id": "test_low",
            "title": "Stochastic Gradient Descent: A Survey",
            "year": 2021,
            "authors": ["Carol Williams"],
            "abstract": "We survey variants of stochastic gradient descent in deep learning including Adam, RMSProp, and Adagrad, with convergence analysis.",
        },
        {
            "paper_id": "test_borderline",
            "title": "Graph Contrastive Learning with Augmentations",
            "year": 2020,
            "authors": ["Yuning You", "Tianlong Chen"],
            "abstract": "GraphCL applies contrastive learning to graph-structured data using four types of graph augmentations. It shows strong performance on various semi-supervised and transfer learning benchmarks.",
        },
    ]

    print("=== Human-in-the-Loop Demo ===")
    print(f"Topic: {state['target_topic']}\n")
    print("Papers with score >= 0.65 -> auto-accepted")
    print("Papers with score <  0.40 -> auto-rejected")
    print("Papers in between         -> you decide\n")

    state = process_queue(state)
    advance_iteration(state, STATE_PATH)
