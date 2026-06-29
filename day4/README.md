# Day 4: narrative synthesis & LaTeX generation

## Learning Objectives

- Cluster a curated paper library into thematic groups using an LLM
- Generate a syntactically valid LaTeX `\section{Related Works}` block
- Run a second LLM instance as a "rigor checker" that cross-validates every citation claim

---

## Multi-stage synthesis

Single-shot writing ("here are 20 papers, write my related works section") produces weak output.

A solution is staged synthesis:

```
Stage 1 - CLUSTER:   Group papers into 2-4 thematic sub-sections
Stage 2 - OUTLINE:   Write a sentence-level plan for each sub-section
Stage 2 - DRAFT:    Build BibTeX deterministically, then LLM writes prose (exercises 2-3)
Stage 3 - CHECK:    A second LLM pass validates every factual claim (exercise 4)
```

Each stage gets its own prompt. Each stage's output is the next stage's input. This decomposition is more token-efficient and produces more coherent text than a monolithic prompt.

Exercises 2 and 3 are intentionally split: generating deterministic BibTeX (no LLM) and writing prose (LLM) are separate concerns. The split makes clear that structured data should never pass through an LLM.

---

## Template interpolation vs. free generation

For the LaTeX block, you have two strategies:

**Free generation**: Give the LLM the paper list and ask it to write LaTeX directly. Fast, but the model may hallucinate `\cite` keys, mangle bib entries, or invent facts.

**Template interpolation**: Pre-build the `.bib` file and the citation key map from your JSON data (no LLM involved). Then give the LLM only the *prose* task, constraining it to use only the pre-validated keys from the map.

We use template interpolation. The `.bib` file is generated programmatically from `curated_library`. The LLM is only asked to write the human-readable prose. LaTeX structure and bibliography entries never pass through an LLM.

---

## The rigor checker

The rigor checker is a second LLM call that acts as an adversarial peer reviewer. It is given:
1. The generated LaTeX draft
2. The original `curated_library` JSON

Its job: for every factual claim in the draft (e.g., "Smith et al. achieved 94% accuracy"), find the corresponding paper in the library, read its abstract, and check whether the claim is supported.

This catches two common LLM failure modes:
- **Hallucinated metrics**: The model invents a number that sounds plausible but isn't in the abstract
- **Attribution errors**: The model assigns a result from Paper A to Paper B

The checker outputs a JSON list of `{claim, paper_id, status, correction}`. The agent then either patches the draft automatically (for clear corrections) or flags them for human review.
