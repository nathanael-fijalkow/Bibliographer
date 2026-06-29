# Day 1: Structured thought

## Learning objectives

- Force an LLM to output syntactically predictable JSON without using a framework
- Build a manual ReAct (Reason Act Observe) loop from scratch
- Intercept hallucinated tool calls and iterate

---

## Why JSON?

An LLM is a text-in, text-out function. To build an agent, you need the text it outputs to be **machine-readable**. The canonical solution is to constrain the output format to JSON.

This is harder than it sounds. LLMs routinely:
- Wrap JSON in Markdown code fences (` ```json ... ``` `)
- Add a sentence of explanation before or after the JSON block
- Use single quotes instead of double quotes
- Hallucinate field names that weren't in your schema
- Produce valid JSON that doesn't match your expected schema

We will first build the plumbing that handles all of these failure modes.

---

## The system prompt

The system prompt contains something like this:

```
You must respond with ONLY a JSON object matching this schema:
{
  "reasoning": "<your thinking>",
  "action": { "tool": "<tool_name>", "args": { ... } }
}
Do not add any text before or after the JSON.
```

The system prompt defines the grammar of your agent's outputs. The parsing code defines the grammar checker. They must be designed together.

---

## ReAct

ReAct is the main pattern for tool-using agents:

```
THOUGHT:   The model reasons about what to do next
ACTION:    The model names a tool and its arguments
OBSERVATION: The tool runs and returns a result
(repeat)
```

In the original paper, thoughts and actions were free text. We make them structured JSON so the outer loop can parse them deterministically.

The conversation history grows with each loop:
```
[system] You are a research assistant. Tools: [...]
[user]   State: {...}
[assistant] {"reasoning": "I should search for ...", "action": {"tool": "search_papers", ...}}
[user]   Observation: {"papers": [...]}        <- we inject this
[assistant] {"reasoning": "Paper X looks relevant...", "action": {"tool": "evaluate_paper", ...}}
...
```

The `[user]` observation injection is the key mechanic. The model never "runs" tools, a Python interpreter does, and you feed the results back in.

---

## Hallucinated tools

When you give a model a list of available tools and it calls one that isn't on the list, that is a **hallucinated tool call**. This happens more than you'd expect, especially with smaller models.

The recovery pattern is:

```python
if parsed["action"]["tool"] not in REGISTERED_TOOLS:
    # Don't crash. Inject an error observation and retry.
    observation = {
        "error": f"Tool '{parsed['action']['tool']}' does not exist.",
        "available_tools": list(REGISTERED_TOOLS.keys()),
        "instruction": "Choose one of the available tools."
    }
    messages.append({"role": "user", "content": json.dumps(observation)})
    continue  # back to the top of the loop
```

This mirrors how humans correct mistakes: you don't fire someone for trying the wrong approach, you tell them what went wrong and let them try again.
