"""
The main agent loop.

This is a ReAct loop with:
  - State machine: state.json is the single source of truth
  - Hallucination recovery: unknown tools inject an error observation
  - Human-in-the-loop: borderline papers pause for y/n input
  - Checkpointing: state saved after every iteration
  - Termination: finish tool OR max iterations OR queue-empty condition

After the search loop ends, the synthesis pipeline runs (Day 4):
  cluster -> generate LaTeX -> rigor check
"""

import json
import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.llm_client import chat, _extract_json
from agent.state import checkpoint, advance, STATE_PATH
from agent.tools import TOOL_REGISTRY, TOOL_DESCRIPTIONS, dispatch
from agent.synthesis import cluster_library, generate_latex, run_rigor_check, print_report

MAX_ITERATIONS = 30
MIN_LIBRARY_SIZE = 5   # stop searching when we have this many validated papers
MAX_CONSECUTIVE_ERRORS = 3

_state_ref = None

def _handle_interrupt(signum, frame):
    print("\n\n[Interrupt] Saving state and exiting...")
    if _state_ref is not None:
        checkpoint(_state_ref, STATE_PATH)
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_interrupt)


# --- System prompt

def build_system_prompt(topic: str) -> str:
    tool_block = "\n".join(
        f"  - {t['name']}({t['args']}): {t['description']}"
        for t in TOOL_DESCRIPTIONS
    )
    return f"""\
You are an autonomous research agent building a literature review for a PhD student.

Research topic: "{topic}"

Your goal: find and validate 5-15 highly relevant academic papers.

Available tools:
{tool_block}

Strategy (follow in order):
  1. Call search_papers with 2-3 different keyword queries based on the topic.
  2. Call evaluate_paper for each paper in the discovery queue.
  3. For each validated paper in curated_library, call traverse_citations TWICE:
       traverse_citations(paper_id, direction="backward")  <- what it cites (foundational)
       traverse_citations(paper_id, direction="forward")   <- who cites it (follow-on work)
     You MUST do at least one backward AND one forward traversal before finishing.
  4. Repeat steps 2-3 until you have >= {MIN_LIBRARY_SIZE} validated papers.
  5. Call finish with a brief summary.

On every turn, respond with ONLY a valid JSON object:
{{
  "reasoning": "<your thinking about what to do next>",
  "action": {{
    "tool": "<tool_name>",
    "args": {{ "<arg>": "<value>" }}
  }}
}}

No prose, no markdown fences. Only the JSON object.
"""


# --- Search loop

def run_search_loop(state: dict) -> dict:
    global _state_ref
    _state_ref = state

    messages = [
        {"role": "system", "content": build_system_prompt(state["target_topic"])},
        {"role": "user",   "content": json.dumps({
            "event": "start",
            "state_summary": {
                "library_size": len(state["curated_library"]),
                "queue_size": len(state["discovery_queue"]),
                "seed_keywords": state["seed_keywords"],
            },
        })},
    ]

    consecutive_errors = 0
    traversals_backward = 0
    traversals_forward  = 0

    # Keep only the last N round-trips in context so the conversation doesn't
    # grow indefinitely (some providers fail on very long histories).
    MAX_CONTEXT_PAIRS = 8

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'-' * 55}")
        print(f"ITERATION {iteration}  |  library={len(state['curated_library'])}  queue={len(state['discovery_queue'])}")

        # -- Terminal condition check
        if (len(state["curated_library"]) >= MIN_LIBRARY_SIZE
                and len(state["discovery_queue"]) == 0):
            print(f"\n[loop] Terminal condition: library has {len(state['curated_library'])} papers and queue is empty.")
            break

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"[loop] {MAX_CONSECUTIVE_ERRORS} consecutive errors. Aborting search phase.")
            break

        # -- Prune old context (keep system + start header, then last N pairs)
        header, tail = messages[:2], messages[2:]
        if len(tail) > MAX_CONTEXT_PAIRS * 2:
            tail = tail[-(MAX_CONTEXT_PAIRS * 2):]
            messages = header + tail

        # -- THINK
        raw = chat(messages, max_tokens=1024, temperature=0.2)

        # -- PARSE
        try:
            parsed = _extract_json(raw)
            consecutive_errors = 0
        except ValueError as exc:
            consecutive_errors += 1
            # Only append the assistant turn if it has content — some providers
            # (e.g. Cohere) reject messages with empty or whitespace-only content.
            if raw.strip():
                messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": json.dumps({
                "error": "Response was not valid JSON.",
                "instruction": "Respond with ONLY the JSON object described in the system prompt.",
            })})
            continue

        messages.append({"role": "assistant", "content": json.dumps(parsed)})

        reasoning = parsed.get("reasoning", "")
        action = parsed.get("action", {})
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})

        print(f"[Reasoning]: {reasoning[:120]}")
        print(f"[Action]:    {tool_name}({tool_args})")

        # -- ACT
        if tool_name == "finish":
            # Require at least one backward and one forward traversal before finishing.
            missing = []
            if traversals_backward == 0:
                missing.append("backward (traverse_citations with direction='backward')")
            if traversals_forward == 0:
                missing.append("forward (traverse_citations with direction='forward')")
            if missing and state["curated_library"]:
                print(f"[loop] Blocked finish: traversal not done yet.")
                messages.append({"role": "user", "content": json.dumps({
                    "error": "Cannot finish yet.",
                    "reason": "You must explore the citation graph before finishing.",
                    "missing": missing,
                    "instruction": (
                        "Call traverse_citations on a validated paper for each missing direction, "
                        "then evaluate any new papers that appear in the queue."
                    ),
                })})
                continue
            print(f"\n[loop] Agent called finish: {tool_args.get('summary', '')}")
            break

        observation = dispatch(tool_name, tool_args, state)
        if "error" in observation:
            consecutive_errors += 1

        # Track citation traversals for the finish guard
        if tool_name == "traverse_citations" and "error" not in observation:
            direction = tool_args.get("direction", "")
            if direction == "backward":
                traversals_backward += 1
            elif direction == "forward":
                traversals_forward += 1

        print(f"[Observation]: {json.dumps(observation)[:200]}")

        # -- OBSERVE (inject result)
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})

        # -- CHECKPOINT
        advance(state, STATE_PATH)

    print(f"\n[loop] Search phase complete.")
    print(f"  Library: {len(state['curated_library'])} papers")
    print(f"  Blacklist: {len(state['blacklist'])} papers")
    return state


# --- Synthesis pipeline

def run_synthesis(state: dict) -> None:
    if len(state["curated_library"]) < 2:
        print("\n[synthesis] Not enough papers for synthesis (need >= 2). Skipping.")
        return

    print(f"\n{'=' * 55}")
    print("SYNTHESIS PIPELINE")
    print(f"{'=' * 55}")

    print("\n[Stage 1] Clustering papers into themes...")
    state = cluster_library(state)
    checkpoint(state, STATE_PATH)

    print("\n[Stage 2] Generating LaTeX...")
    out_dir = os.path.dirname(os.path.dirname(__file__))
    latex_content, bib_content = generate_latex(state)

    tex_path = os.path.join(out_dir, "related_works.tex")
    bib_path = os.path.join(out_dir, "references.bib")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib_content)
    print(f"  [OK] {tex_path}")
    print(f"  [OK] {bib_path}")

    print("\n[Stage 3] Running rigor check...")
    findings = run_rigor_check(latex_content, state)
    print_report(findings)

    report_path = os.path.join(out_dir, "rigor_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print(f"  [OK] {report_path}")
