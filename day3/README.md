# Day 3: state tracking, de-duplication, and persistent memory

## Learning objectives

- Checkpoint the agent's state to disk after every loop iteration
- Normalise paper titles and authors to detect duplicates across APIs
- Add a human-in-the-loop interrupt for uncertain papers

---

## Persistence

Running 30 iterations of an agent is a big risk: if the process crashes at step 29, you lose everything. More subtly: without persistence, the agent cannot be restarted with partial progress, cannot be inspected mid-run, and cannot be audited after the run.

After every iteration:

```
1. Execute the loop body
2. Update state in memory
3. Write state.json to disk atomically
4. Increment the iteration counter
```

"Atomically" here means: write to a temp file, then `os.replace()` it over the real file. This prevents a partial write from corrupting the checkpoint if the process is killed mid-write.

---

## Memory Write-Back pattern

In database terms, the agent operates with a **write-through cache**: every change to the in-memory `state` dict is immediately reflected on disk. This is more conservative than **write-back** (buffer changes and flush periodically), but for an agent with 5-50 iterations per run, the performance difference is irrelevant and the safety benefit is large.

```python
# Every tool call that modifies state calls this:
def add_to_library(paper, state):
    state["curated_library"][paper["paper_id"]] = paper
    checkpoint(state)   # <- immediate write-back
```

The checkpoint function also records `iteration` and a `last_updated` timestamp so you can tell exactly when a run was interrupted.

---

## Cache Invalidation

The blacklist is a negative cache: it stores paper IDs we've already decided are irrelevant. Without it, the agent might re-evaluate the same irrelevant paper on every run (especially via citation traversal, which surfaces the same popular papers repeatedly).

The de-duplication check - `already_seen()` - is the cache hit test. It checks three places:
1. `curated_library` (already validated and kept)
2. `blacklist` (already validated and rejected)
3. `discovery_queue` (currently awaiting evaluation)

A cache miss means the paper is genuinely new. Only then does it enter the evaluation pipeline.

---

## State recovery

When the agent starts, it should check whether a checkpoint already exists:

```python
if os.path.exists("state.json"):
    state = json.load(open("state.json"))
    print(f"Resuming from iteration {state['iteration']}")
else:
    state = default_state()
```

This transforms the agent from a one-shot script into a **resumable process**. You can kill it after 10 iterations, inspect the state, and resume from exactly where you stopped.

---

## Title normalisation

The same paper can appear in multiple API responses with slightly different titles:

- `"HGNN: Hypergraph Neural Networks"` (Semantic Scholar)
- `"Hypergraph Neural Networks (HGNN)"` (arXiv)
- `"hypergraph neural networks"` (a citing paper's reference list, lowercased)

Naively checking `if title in library` misses all three variations. The normalisation strategy:

1. Lowercase the title
2. Remove punctuation and accents (Unicode normalisation)
3. Strip common prefixes ("a ", "an ", "the ")
4. Sort the word tokens and hash them

Two titles with the same normalised hash are treated as duplicates. This catches 95% of real-world variations without a full string-similarity library.
