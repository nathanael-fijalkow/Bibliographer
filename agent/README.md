# Full integrated agent

A self-contained agent that crawls citation networks, evaluates papers against
a research topic, and synthesises a `related_works.tex` LaTeX snippet.

## Files

```
agent/
  main.py        CLI entry point (--reset, --topic, --max-iterations, --no-synthesis)
  loop.py        ReAct loop: THINK -> PARSE -> ACT -> OBSERVE -> CHECKPOINT
  state.py       load / checkpoint / advance (writes state.json atomically)
  tools.py       tool implementations + TOOL_REGISTRY + dispatch()
  apis.py        Semantic Scholar + arXiv API wrappers
  eval.py        LLM relevance evaluation + title deduplication
  synthesis.py   cluster -> LaTeX prose -> rigor check
```

The agent is **self-contained**: `agent/` imports nothing from `day1/`-`day4/`.
The day* directories are teaching exercises; this directory is the production integration.

## Running

Run from the **repo root**:

```bash
uv run python agent/main.py
uv run python agent/main.py --reset
uv run python agent/main.py --reset --topic "Your dissertation topic"
uv run python agent/main.py --max-iterations 20
uv run python agent/main.py --no-synthesis   # skip LaTeX generation
```

## What it produces

| File | Contents |
|---|---|
| `state.json` | Agent state - resumes from here on next run |
| `related_works.tex` | LaTeX section ready to `\input{}` |
| `references.bib` | BibTeX bibliography |
| `rigor_report.json` | Per-claim fact-check findings |

## Loop termination

The agent stops when:
1. Library has >= 5 papers AND the discovery queue is empty, OR
2. The maximum iteration count is reached, OR
3. The model calls `finish` (only allowed after at least one backward + one forward traversal)

After the search loop, the synthesis pipeline runs automatically unless `--no-synthesis` is set.
