"""
Unified LLM client: Hugging Face Inference API or LiteLLM.

Set LLM_BACKEND=huggingface (default) or LLM_BACKEND=litellm in your .env.

Usage:
    from shared.llm_client import chat

    response = chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is contrastive learning?"},
    ])
    print(response)  # plain string
"""

import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

BACKEND = os.getenv("LLM_BACKEND", "huggingface").lower()


# --- Hugging Face Inference API

def _hf_chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """
    Calls the HuggingFace Inference API using huggingface_hub.InferenceClient.

    The InferenceClient exposes an OpenAI-compatible .chat.completions.create()
    interface for instruction-tuned models hosted on HF serverless inference.
    """
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. See .env.example.")

    model    = model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    provider = os.getenv("HF_PROVIDER") or None   # None -> HF picks automatically
    client   = InferenceClient(token=token, provider=provider)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=kwargs.get("max_tokens", 1024),
        temperature=kwargs.get("temperature", 0.2),
    )
    return response.choices[0].message.content


# --- LiteLLM

def _litellm_chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """
    Routes the call through LiteLLM, which supports OpenAI, Anthropic, Groq,
    Ollama, and 100+ other providers using the same interface.

    The model string follows LiteLLM's convention:
        gpt-4o-mini                      -> OpenAI
        anthropic/claude-haiku-4-5       -> Anthropic
        groq/llama3-8b-8192              -> Groq
        ollama/mistral                   -> local Ollama
    """
    import litellm

    model = model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=kwargs.get("max_tokens", 1024),
        temperature=kwargs.get("temperature", 0.2),
    )
    return response.choices[0].message.content


# --- Public interface

def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    """
    Send a list of chat messages to the configured LLM and return the reply.

    Args:
        messages:  OpenAI-format message list, e.g.
                   [{"role": "system", "content": "..."}, {"role": "user", ...}]
        model:     Override the model set in .env (optional).
        **kwargs:  Passed through: max_tokens, temperature, etc.

    Returns:
        The assistant's reply as a plain string.
    """
    if BACKEND == "litellm":
        return _litellm_chat(messages, model=model, **kwargs)
    return _hf_chat(messages, model=model, **kwargs)


def chat_json(messages: list[dict], model: str | None = None, **kwargs) -> dict:
    """
    Like chat(), but extracts and parses the first JSON object in the reply.

    Many exercises in this course require the LLM to output a JSON object.
    This helper centralises the extraction logic so individual scripts don't
    each need to handle `json.JSONDecodeError`.

    Raises:
        ValueError: if no valid JSON object is found in the response.
    """
    raw = chat(messages, model=model, **kwargs)
    return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from text that may contain prose around it.

    LLMs often wrap JSON in markdown fences (```json ... ```) or add a
    sentence before/after. This function strips that scaffolding.
    """
    # Strip markdown code fences
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Found JSON-shaped text but it failed to parse:\n{candidate}\n\nError: {exc}"
                    )

    raise ValueError(f"Unbalanced braces in LLM response:\n{text}")
