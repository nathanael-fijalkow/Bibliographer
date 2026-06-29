"""
Bibliographer Agent - Entry Point

Usage:
    python agent/main.py
    python agent/main.py --reset
    python agent/main.py --topic "Graph neural networks for drug discovery"
    python agent/main.py --max-iterations 15 --no-synthesis

Run from the repo root directory.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.state import load_state, reset as reset_state, STATE_PATH
from agent.loop import run_search_loop, run_synthesis


def main():
    parser = argparse.ArgumentParser(description="Bibliographer: autonomous literature review agent")
    parser.add_argument("--reset", action="store_true",
                        help="Clear state.json and start from scratch")
    parser.add_argument("--topic", type=str, default=None,
                        help="Override the research topic (implies --reset)")
    parser.add_argument("--max-iterations", type=int, default=30,
                        help="Maximum search loop iterations (default: 30)")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="Skip the Day 4 synthesis pipeline after search")
    args = parser.parse_args()

    # Override loop constant if requested
    if args.max_iterations != 30:
        import agent.loop as _loop
        _loop.MAX_ITERATIONS = args.max_iterations

    # State initialisation
    if args.reset or args.topic:
        print(f"[main] Starting fresh{' with topic: ' + args.topic if args.topic else ''}.")
        state = reset_state(topic=args.topic)
    else:
        state = load_state(STATE_PATH)
        if state["iteration"] > 0:
            print(f"[main] Resuming from iteration {state['iteration']}.")
            print(f"       Library: {len(state['curated_library'])} papers, "
                  f"Queue: {len(state['discovery_queue'])} papers.")
        else:
            print(f"[main] Starting new run.")
            print(f"       Topic: {state['target_topic']}")

    print()

    # Search phase
    state = run_search_loop(state)

    # Synthesis phase
    if not args.no_synthesis:
        run_synthesis(state)
    else:
        print("\n[main] Skipping synthesis (--no-synthesis).")

    print(f"\n{'=' * 55}")
    print("Run complete.")
    print(f"  Final library: {len(state['curated_library'])} papers")
    print(f"  Iterations:    {state['iteration']}")
    print(f"  State saved:   {STATE_PATH}")

    if not args.no_synthesis and len(state["curated_library"]) >= 2:
        out_dir = os.path.dirname(os.path.dirname(__file__))
        print(f"  LaTeX output:  {os.path.join(out_dir, 'related_works.tex')}")
        print(f"  Bibliography:  {os.path.join(out_dir, 'references.bib')}")
        print(f"  Rigor report:  {os.path.join(out_dir, 'rigor_report.json')}")


if __name__ == "__main__":
    main()
