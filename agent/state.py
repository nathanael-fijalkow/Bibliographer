"""
Agent state management. Thin wrapper around day3/01_state_manager.py
that knows where the canonical state.json lives relative to this file.
"""

import json
import os
import time

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state.json")


DEFAULT_TOPIC = "Contrastive learning over sparse hypergraphs"
DEFAULT_KEYWORDS = [
    "contrastive learning",
    "hypergraph neural networks",
    "sparse representation",
    "self-supervised learning on graphs",
]


def default_state(topic: str | None = None) -> dict:
    custom_topic = topic and topic != DEFAULT_TOPIC
    return {
        "target_topic": topic or DEFAULT_TOPIC,
        # When a custom topic is given, leave keywords empty so the agent
        # derives its own queries from the topic instead of using stale examples.
        "seed_keywords": [] if custom_topic else DEFAULT_KEYWORDS,
        "curated_library": {},
        "discovery_queue": [],
        "blacklist": [],
        "pending_review": [],
        "narrative_outline": None,
        "iteration": 0,
        "last_updated": None,
    }


def load_state(path: str = STATE_PATH) -> dict:
    if not os.path.exists(path):
        return default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Fill in keys that may be missing from older checkpoints
        for key, val in default_state().items():
            if key not in state:
                state[key] = val
        # Repair type mismatches — curated_library must be a dict, not a list
        if not isinstance(state.get("curated_library"), dict):
            print("[state] curated_library was not a dict — resetting to {}.")
            state["curated_library"] = {}
        if not isinstance(state.get("discovery_queue"), list):
            state["discovery_queue"] = []
        if not isinstance(state.get("blacklist"), list):
            state["blacklist"] = []
        return state
    except (json.JSONDecodeError, OSError):
        print("[state] Corrupt checkpoint - starting fresh.")
        return default_state()


def checkpoint(state: dict, path: str = STATE_PATH) -> None:
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def advance(state: dict, path: str = STATE_PATH) -> dict:
    state["iteration"] += 1
    checkpoint(state, path)
    return state


def reset(topic: str | None = None, path: str = STATE_PATH) -> dict:
    fresh = default_state(topic)
    checkpoint(fresh, path)
    return fresh
