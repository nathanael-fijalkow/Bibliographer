"""
Day 1, Exercise 1: schema parser

Given a research topic, ask the LLM to output a structured execution plan. The plan is a JSON object listing which tools the agent should call first, and in what order, to bootstrap a literature search.

Key lessons:
  - Crafting a system prompt that produces reliable JSON
  - Parsing and validating LLM output against an expected schema
  - Handling the cases where the model ignores your format instructions

Run:
    python day1/01_schema_parser.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.llm_client import chat, _extract_json


# --- Schema definition
#
# We define the schema as a string so we can embed it verbatim in the system
# prompt. This makes the contract explicit to both the LLM and the reader.

PLAN_SCHEMA = """
{
  "reasoning": "<why these steps make sense for this topic>",
  "steps": [
    {
      "step": 1,
      "tool": "<tool_name>",
      "args": { "<arg_name>": "<arg_value>" },
      "goal": "<one sentence: what we hope to learn from this step>"
    }
  ],
  "estimated_papers_needed": <integer>
}
"""

AVAILABLE_TOOLS = [
    {
        "name": "search_papers",
        "description": "Search an academic API by keyword. Args: query (str), max_results (int).",
    },
    {
        "name": "evaluate_paper",
        "description": "Score a paper's relevance to the target topic. Args: paper_id (str).",
    },
    {
        "name": "traverse_citations",
        "description": "Get all papers that cite or are cited by a given paper. Args: paper_id (str), direction ('forward'|'backward').",
    },
    {
        "name": "finish",
        "description": "End the planning phase and return the plan.",
    },
]


def build_system_prompt(tools: list[dict]) -> str:
    tool_descriptions = "\n".join(
        f"  - {t['name']}: {t['description']}" for t in tools
    )
    return f"""\
You are a research planning assistant for PhD students.
Your job is to produce an initial execution plan for a literature search agent.

Available tools:
{tool_descriptions}

You MUST respond with ONLY a valid JSON object matching this exact schema:
{PLAN_SCHEMA}

Rules:
- Do not include markdown fences, prose, or any text outside the JSON object.
- The "steps" array must contain between 2 and 5 steps.
- Only use tool names from the available tools list above.
- "estimated_papers_needed" must be a reasonable integer (5-50 for most topics).
"""


def validate_plan(plan: dict) -> list[str]:
    """
    Returns a list of validation errors. Empty list = valid.
    """
    errors = []
    required_top = {"reasoning", "steps", "estimated_papers_needed"}
    missing = required_top - set(plan.keys())
    if missing:
        errors.append(f"Missing top-level fields: {missing}")

    valid_tools = {t["name"] for t in AVAILABLE_TOOLS}

    for i, step in enumerate(plan.get("steps", [])):
        required_step = {"step", "tool", "args", "goal"}
        missing_step = required_step - set(step.keys())
        if missing_step:
            errors.append(f"Step {i}: missing fields {missing_step}")
        if step.get("tool") not in valid_tools:
            errors.append(
                f"Step {i}: unknown tool '{step.get('tool')}'. "
                f"Valid: {sorted(valid_tools)}"
            )

    count = plan.get("estimated_papers_needed")
    if not isinstance(count, int) or count < 1:
        errors.append(f"estimated_papers_needed must be a positive integer, got: {count!r}")

    return errors


def generate_plan(target_topic: str) -> dict:
    messages = [
        {"role": "system", "content": build_system_prompt(AVAILABLE_TOOLS)},
        {
            "role": "user",
            "content": f"Generate an execution plan for the following research topic:\n\n{target_topic}",
        },
    ]

    print(f"[schema_parser] Calling LLM for topic: {target_topic!r}")
    raw = chat(messages, max_tokens=800, temperature=0.3)

    print(f"\n[schema_parser] Raw LLM output:\n{raw}\n")

    plan = _extract_json(raw)

    errors = validate_plan(plan)
    if errors:
        print("[schema_parser] Validation errors:")
        for err in errors:
            print(f"  [x] {err}")
        raise ValueError(f"Plan failed validation: {errors}")

    print("[schema_parser] Plan validated successfully.")
    return plan


if __name__ == "__main__":
    topic = "Contrastive learning over sparse hypergraphs"

    plan = generate_plan(topic)

    print("\n" + "=" * 60)
    print("EXECUTION PLAN")
    print("=" * 60)
    print(json.dumps(plan, indent=2))
    print(f"\nEstimated papers needed: {plan['estimated_papers_needed']}")
    print(f"Steps planned: {len(plan['steps'])}")
