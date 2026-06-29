"""
Day 1, Exercise 2: ReAct loop with error recovery

Two concepts in one script, building on each other:

PART 1 - The ReAct loop
  ReAct (Reason, Act, Observe) is the backbone of most LLM agents.
  The agent thinks (Reason), picks a tool call (Act), runs it, then feeds the
  result back into the conversation as a user turn (Observe). The messages list
  IS the agent's working memory for the session.

  Key mechanics:
    - Conversation history encodes everything the agent has seen and done
    - Tool outputs are injected as {"role": "user", "content": observation}
    - The agent drives termination by calling the `finish` tool

PART 2 - Hallucination recovery
  When the model calls a tool that does not exist, the agent must not crash.
  Instead it injects a structured error back into the conversation so the model
  can self-correct on the next turn. A retry budget (MAX_CONSECUTIVE_ERRORS)
  prevents an infinite loop if the model is persistently confused.

  Key mechanics:
    - A tight tool registry (only 3 tools) makes hallucinations observable
    - build_hallucination_error() produces a corrective observation
    - consecutive_errors tracks how many bad turns in a row have occurred
    - A hallucination log lets you audit which tools the model invented

Run:
    python day1/02_react_loop.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json

MAX_STEPS = 10
MAX_CONSECUTIVE_ERRORS = 3


# --- Toy tool implementations
#
# These return fake data. Day 2 replaces them with real API calls.
# The registry is intentionally small - only 3 tools - to make hallucinations
# observable. A model tempted to call `search_google_scholar` or
# `get_citation_count` will be caught.

def search_papers(query: str, max_results: int = 3) -> dict:
    fake_papers = [
        {
            "paper_id": "paper_001",
            "title": "Hypergraph Neural Networks for Semi-Supervised Classification",
            "year": 2019,
            "abstract_snippet": "We propose HGNN, a general framework for data fitting with hypergraph structure...",
        },
        {
            "paper_id": "paper_002",
            "title": "Self-Supervised Contrastive Learning on Graphs",
            "year": 2021,
            "abstract_snippet": "GraphCL applies contrastive learning to graph-structured data via augmentation...",
        },
        {
            "paper_id": "paper_003",
            "title": "Sparse Hypergraph Representation Learning",
            "year": 2023,
            "abstract_snippet": "We address scalability in hypergraph learning via sparse attention mechanisms...",
        },
    ]
    return {"query": query, "results": fake_papers[:max_results]}


def evaluate_paper(paper_id: str) -> dict:
    scores = {
        "paper_001": {"score": 0.72, "verdict": "relevant",       "reason": "HGNN directly relates to hypergraph learning"},
        "paper_002": {"score": 0.65, "verdict": "relevant",       "reason": "Contrastive learning on graphs is closely related"},
        "paper_003": {"score": 0.88, "verdict": "highly_relevant","reason": "Sparse hypergraphs are the exact target domain"},
    }
    return scores.get(
        paper_id,
        {"score": 0.1, "verdict": "irrelevant", "reason": "Paper not found in mock database"},
    )


def finish(summary: str = "") -> dict:
    return {"status": "finished", "summary": summary}


TOOL_REGISTRY = {
    "search_papers":  search_papers,
    "evaluate_paper": evaluate_paper,
    "finish":         finish,
}

TOOL_DESCRIPTIONS = [
    {"name": "search_papers",  "args": "query: str, max_results: int = 3",
     "description": "Search for academic papers matching a keyword query."},
    {"name": "evaluate_paper", "args": "paper_id: str",
     "description": "Score a paper's relevance to the target topic (0.0-1.0)."},
    {"name": "finish",         "args": "summary: str = ''",
     "description": "End the session. Call this when you have enough information."},
]


# --- System prompt
#
# The framing as "expert academic assistant" is intentional: it signals a broad
# academic domain, which tempts some models to reach for tools they "know"
# from training (e.g. search_semantic_scholar, get_impact_factor).

def build_system_prompt(topic: str) -> str:
    tool_block = "\n".join(
        f"  - {t['name']}({t['args']}): {t['description']}"
        for t in TOOL_DESCRIPTIONS
    )
    return f"""\
You are an expert academic research assistant for PhD students.
You help find and evaluate papers for literature reviews on: "{topic}"

You have access to these tools ONLY:
{tool_block}

Respond with ONLY a JSON object each turn:
{{
  "reasoning": "<your thinking about what to do next>",
  "action": {{
    "tool": "<tool_name>",
    "args": {{ "<arg>": "<value>" }}
  }}
}}

No prose. No markdown fences. Only the JSON object.
Call `finish` when you have searched for at least 2 queries and evaluated at least 2 papers.
"""


# --- PART 1: Core loop
#
# This is the plain version without error recovery. Read this first to
# understand the basic Reason -> Act -> Observe cycle, then read run_react_loop()
# below to see how error handling layers on top.

def run_react_loop_basic(topic: str) -> None:
    """Plain ReAct loop. Crashes on bad JSON, ignores unknown tools."""
    messages = [
        {"role": "system", "content": build_system_prompt(topic)},
        {"role": "user",   "content": f"Begin: {{'curated_library': {{}}, 'queue': []}}"},
    ]
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- STEP {step}")
        raw = chat(messages, max_tokens=512, temperature=0.2)
        parsed = _extract_json(raw)   # raises ValueError on bad JSON
        messages.append({"role": "assistant", "content": json.dumps(parsed)})

        action = parsed.get("action", {})
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})
        print(f"[Action]: {tool_name}({tool_args})")

        if tool_name == "finish":
            print(f"Finished: {tool_args.get('summary', '')}")
            return

        observation = TOOL_REGISTRY.get(tool_name, lambda **_: {"error": "unknown tool"})(**tool_args)
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})


# --- PART 2: Hallucination recovery
#
# Three additions over the basic loop:
#   1. build_hallucination_error() - structured error message that tells the
#      model exactly what went wrong and lists valid alternatives
#   2. consecutive_errors counter - abort if the model keeps failing
#   3. hallucination_log - audit which invented tools were called

def build_hallucination_error(bad_tool: str) -> dict:
    """
    Produce a corrective error observation.

    Wording matters: too terse ("invalid tool") gives no signal; too verbose
    wastes context. The sweet spot names the error, lists valid options, and
    re-states the constraint.
    """
    return {
        "error": f"Tool '{bad_tool}' does not exist in this agent.",
        "available_tools": list(TOOL_REGISTRY.keys()),
        "instruction": (
            f"'{bad_tool}' is not in the list of available tools. "
            "Please choose from the available_tools list only."
        ),
    }


def run_react_loop(topic: str) -> None:
    """
    Full ReAct loop with JSON-parse recovery and hallucination handling.
    Produces a post-run hallucination report.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(topic)},
        {"role": "user",   "content": f"Begin: {{'curated_library': {{}}, 'queue': []}}"},
    ]

    consecutive_errors = 0
    hallucination_log: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'-' * 50}")
        print(f"STEP {step}  (consecutive errors: {consecutive_errors})")

        # Abort if the model is stuck in an error loop
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"[ABORT] {MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                  "Check system prompt or model choice.")
            break

        raw = chat(messages, max_tokens=512, temperature=0.2)
        print(f"[LLM raw]: {raw[:300]}")

        # -- Parse: recover from bad JSON without crashing
        try:
            parsed = _extract_json(raw)
        except ValueError:
            consecutive_errors += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": json.dumps({
                "error": "Response was not valid JSON.",
                "instruction": "You must respond with ONLY a JSON object. No other text.",
            })})
            continue

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        print(f"[Reasoning]: {parsed.get('reasoning', '')[:120]}")

        action = parsed.get("action", {})
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})
        print(f"[Action]:    {tool_name}({tool_args})")

        # -- Finish
        if tool_name == "finish":
            print(f"\n{'=' * 50}")
            print("Agent finished successfully.")
            print(f"Summary: {tool_args.get('summary', '(none)')}")
            break

        # -- Hallucination check: is this tool in the registry?
        if tool_name not in TOOL_REGISTRY:
            consecutive_errors += 1
            hallucination_log.append(tool_name)
            error_obs = build_hallucination_error(tool_name)
            print(f"[HALLUCINATION] Model called: '{tool_name}'")
            messages.append({"role": "user", "content": json.dumps(error_obs)})
            continue

        # -- Valid tool: execute and reset error counter
        consecutive_errors = 0
        try:
            observation = TOOL_REGISTRY[tool_name](**tool_args)
        except TypeError as exc:
            observation = {"error": f"Bad arguments for '{tool_name}': {exc}"}
            consecutive_errors += 1

        print(f"[Observation]: {json.dumps(observation)[:200]}")
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})

    # -- Post-run report
    print(f"\n{'=' * 50}")
    print("HALLUCINATION REPORT")
    print(f"  Total steps: {step}")
    print(f"  Hallucinated tools: {len(hallucination_log)}")
    if hallucination_log:
        for name, count in Counter(hallucination_log).most_common():
            print(f"    '{name}' called {count} time(s)")
    else:
        print("  None detected - model stayed within the tool registry.")


if __name__ == "__main__":
    topic = "Contrastive learning over sparse hypergraphs"
    run_react_loop(topic)
