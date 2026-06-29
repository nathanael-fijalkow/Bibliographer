"""
Day 3, Exercise 1: Robust State Management

Two problems that every persistent agent must solve, presented together because
they are two sides of the same coin: keeping the library consistent over time.

PART A: Atomic checkpointing
  After every loop iteration, state.json on disk must be consistent with what
  the agent believes. A crash at any point must leave the agent able to resume
  from the last complete iteration without data loss or corruption.

  The atomic write pattern:
    1. Serialise state to JSON
    2. Write to state.json.tmp  (crash here -> old file intact)
    3. os.replace()             (atomic on POSIX; nearly atomic on Windows)
    4. Old file is never partially overwritten

  Why not just open(path, 'w')? If the process dies mid-write, you get a
  truncated file. os.replace() swaps file handles atomically at the OS level.

PART B: Title deduplication
  Semantic Scholar and arXiv return slightly different strings for the same
  paper:
    "HGNN: Hypergraph Neural Networks for Learning with Hypergraph Structure"
    "Hypergraph Neural Networks (HGNN)"
    "hypergraph neural networks"

  A naive string-equality check treats these as three different papers.
  The fingerprint approach normalises titles to a canonical form, then hashes
  them. The 16-char SHA-1 prefix is the dedup key.

  Normalisation pipeline:
    1. Unicode NFKD (handles accents, ligatures, fullwidth chars)
    2. ASCII stripping (removes diacritics)
    3. Lowercase
    4. Remove punctuation
    5. Remove stop words ("a", "the", "on", ...)
    6. Sort words and rejoin (handles word-order variations)

Run:
    python day3/01_robust_state.py
"""

import json
import os
import sys
import time
import unicodedata
import re
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state.json")


# =============================================================================
# PART A: Atomic checkpointing
# =============================================================================

def default_state() -> dict:
    return {
        "target_topic": "Contrastive learning over sparse hypergraphs",
        "seed_keywords": [
            "contrastive learning",
            "hypergraph neural networks",
            "sparse representation",
            "self-supervised learning on graphs",
        ],
        "curated_library": {},
        "discovery_queue": [],
        "blacklist": [],
        "narrative_outline": None,
        "iteration": 0,
        "last_updated": None,
    }


def load_state(path: str = STATE_PATH) -> dict:
    """
    Load state from disk. Returns default state if the file is missing or
    corrupt. Never crashes - the agent must always be able to start.
    """
    if not os.path.exists(path):
        print(f"[state] No checkpoint found at {path}. Starting fresh.")
        return default_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        iteration = state.get("iteration", 0)
        last = state.get("last_updated", "unknown")
        print(f"[state] Loaded checkpoint: iteration={iteration}, last_updated={last}")
        return state
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[state] Checkpoint corrupted ({exc}). Starting fresh.")
        return default_state()


def checkpoint(state: dict, path: str = STATE_PATH) -> None:
    """
    Write state to disk atomically via a temp file + os.replace().
    The old checkpoint is intact until the new one is fully written.
    """
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def advance_iteration(state: dict, path: str = STATE_PATH) -> dict:
    """
    Increment the loop counter and checkpoint. Call at the END of every agent
    loop iteration, after all state mutations for that step are complete.
    """
    state["iteration"] += 1
    checkpoint(state, path)
    print(f"[state] Checkpoint saved. Iteration: {state['iteration']}")
    return state


def reset_state(path: str = STATE_PATH) -> dict:
    if os.path.exists(path):
        confirm = input(f"Reset state at {path}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return load_state(path)
    fresh = default_state()
    checkpoint(fresh, path)
    print("[state] State reset.")
    return fresh


# =============================================================================
# PART B: Title deduplication
# =============================================================================

STOP_WORDS = {"a", "an", "the", "on", "in", "of", "for", "with", "and", "or"}


def normalise_title(title: str) -> str:
    """
    Canonical form of a paper title for deduplication purposes.
    See module docstring for the full pipeline.
    """
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())
    words = [w for w in clean.split() if w and w not in STOP_WORDS]
    words.sort()
    return " ".join(words)


def title_fingerprint(title: str) -> str:
    """
    16-char SHA-1 prefix of the normalised title. Used as the duplicate key.
    Not used for security - SHA-1 is just a compact, stable hash for strings.
    """
    return hashlib.sha1(normalise_title(title).encode()).hexdigest()[:16]


def normalise_authors(authors: list[str]) -> frozenset[str]:
    """Last-name-only normalisation for secondary duplicate checking."""
    result = set()
    for name in authors:
        clean = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        parts = re.split(r"[,\s]+", clean.lower().strip())
        if parts:
            result.add(parts[0])
    return frozenset(result)


def already_seen(paper: dict, state: dict) -> tuple[bool, str]:
    """
    Check if a paper is already in any part of the state (library, blacklist,
    queue). Checks by paper_id first (exact), then title fingerprint.

    Returns:
        (True, reason_string) if duplicate
        (False, "") if new
    """
    paper_id    = paper.get("paper_id", "")
    fingerprint = title_fingerprint(paper.get("title", ""))

    if paper_id and paper_id in state["curated_library"]:
        return True, f"paper_id '{paper_id}' already in curated_library"
    if paper_id and paper_id in state.get("blacklist", []):
        return True, f"paper_id '{paper_id}' is blacklisted"

    queue_ids = {p.get("paper_id") for p in state.get("discovery_queue", [])}
    if paper_id and paper_id in queue_ids:
        return True, f"paper_id '{paper_id}' already in discovery_queue"

    lib_fingerprints = {
        title_fingerprint(p.get("title", "")): pid
        for pid, p in state["curated_library"].items()
    }
    if fingerprint in lib_fingerprints:
        return True, f"title fingerprint matches library paper '{lib_fingerprints[fingerprint]}'"

    queue_fps = {title_fingerprint(p.get("title", "")) for p in state.get("discovery_queue", [])}
    if fingerprint in queue_fps:
        return True, "title fingerprint matches a paper already in discovery_queue"

    return False, ""


def deduplicate_queue(state: dict) -> dict:
    """
    Clean the discovery_queue in one pass: remove papers whose fingerprint
    already appears in curated_library or blacklist, and remove intra-queue
    duplicates. Mutates state in-place and returns a summary.
    """
    original_size = len(state["discovery_queue"])
    clean_queue: list[dict] = []
    removed: list[tuple[str, str]] = []

    lib_fingerprints = {
        title_fingerprint(p.get("title", ""))
        for p in state["curated_library"].values()
    }
    blacklisted_ids = set(state.get("blacklist", []))
    seen_fingerprints: set[str] = set()

    for paper in state["discovery_queue"]:
        pid = paper.get("paper_id", "")
        fp  = title_fingerprint(paper.get("title", ""))

        if pid in blacklisted_ids:
            removed.append((paper["title"], "blacklisted"))
        elif fp in lib_fingerprints:
            removed.append((paper["title"], "already in library"))
        elif fp in seen_fingerprints:
            removed.append((paper["title"], "duplicate within queue"))
        else:
            seen_fingerprints.add(fp)
            clean_queue.append(paper)

    state["discovery_queue"] = clean_queue
    return {
        "original_size": original_size,
        "cleaned_size": len(clean_queue),
        "removed": len(removed),
        "removed_reasons": removed[:5],
    }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== PART A: Atomic Checkpointing Demo ===\n")

    state = load_state()
    print(f"Starting at iteration {state['iteration']}\n")

    for i in range(3):
        fake_id = f"paper_{state['iteration']:03d}"
        state["curated_library"][fake_id] = {
            "paper_id": fake_id,
            "title": f"Simulated Paper {state['iteration']}",
            "year": 2024,
        }
        print(f"Loop {i+1}: added {fake_id} to library")
        advance_iteration(state)

    print(f"\nFinal state: {state['iteration']} iterations, "
          f"{len(state['curated_library'])} papers")

    print("\nSimulating restart (reload from disk)...")
    recovered = load_state()
    assert recovered["iteration"] == state["iteration"], "Checkpoint mismatch!"
    print(f"Recovered {len(recovered['curated_library'])} papers - state is consistent.\n")

    # -------------------------------------------------------------------------

    print("=== PART B: Deduplication Demo ===\n")

    print("Pairs the normaliser collapses to the same fingerprint:")
    matching_pairs = [
        ("Hypergraph Neural Networks.",    "hypergraph neural networks"),
        ("On Hypergraph Neural Networks",  "Hypergraph Neural Networks"),
        ("The Hypergraph Neural Networks Framework", "Hypergraph Neural Networks Framework"),
    ]
    for a, b in matching_pairs:
        fa, fb = title_fingerprint(a), title_fingerprint(b)
        ok = "[OK]" if fa == fb else "[x]"
        print(f"  {ok}  '{a}'")
        print(f"       vs '{b}'")

    diff_title = "A Completely Different Paper on Attention Mechanisms"
    print(f"\nGenuinely different title has unique fingerprint: "
          f"{title_fingerprint(diff_title) != title_fingerprint(matching_pairs[0][0])}")

    print("\nDeduplication run:")
    demo_state = {
        "curated_library": {
            "s2_hgnn": {
                "paper_id": "s2_hgnn",
                "title": "HGNN: Hypergraph Neural Networks for Learning with Hypergraph Structure",
                "year": 2019,
            }
        },
        "discovery_queue": [
            {"paper_id": "arxiv_hgnn", "title": "Hypergraph Neural Networks (HGNN)", "year": 2019},
            {"paper_id": "new_paper",  "title": "Sparse Contrastive Learning on Hypergraphs", "year": 2023},
            {"paper_id": "dup_new",    "title": "Sparse Contrastive Learning on Hypergraphs", "year": 2023},
        ],
        "blacklist": [],
    }
    print(f"Queue before: {len(demo_state['discovery_queue'])} papers")
    result = deduplicate_queue(demo_state)
    print(f"Queue after:  {len(demo_state['discovery_queue'])} papers")
    print(f"Removed: {result['removed']}")
    for title, reason in result["removed_reasons"]:
        print(f"  - {title[:50]} -> {reason}")
