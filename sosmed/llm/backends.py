"""
LLM backends: OpenRouter, Anthropic, OpenAI, Ollama.
"""

import json
import os
import re
import sys
from typing import Any

from ..utils import log, BOLD, RESET, DEFAULT_OPENROUTER_MODEL, DEFAULT_OPENROUTER_BASE


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Robustly extract a JSON array from LLM text."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        # Wrapped in an object — find the array
        for v in data.values():
            if isinstance(v, list):
                return v
        return []
    except json.JSONDecodeError:
        pass
    # Last resort: find first [ ... ] block
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())
    log("ERROR", "Could not parse JSON from LLM response")
    return []


def openrouter(
    system: str,
    user: str,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE,
) -> list[dict[str, Any]]:
    """Call OpenRouter (OpenAI-compatible API). Default: free model."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenRouter → {BOLD}{model}{RESET}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/ai-video-clipper",
            "X-Title": "AI Video Clipper",
        },
    )
    raw = resp.choices[0].message.content or ""
    clips = _parse_llm_json(raw)
    log("OK", f"OpenRouter returned {len(clips)} clips")
    return clips


def anthropic(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call Anthropic Claude."""
    try:
        import anthropic
    except ImportError:
        log("ERROR", "anthropic SDK not installed.  pip install anthropic")
        sys.exit(1)

    log("LLM", f"Anthropic → {BOLD}claude-sonnet-4-20250514{RESET}")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text
    clips = _parse_llm_json(raw)
    log("OK", f"Anthropic returned {len(clips)} clips")
    return clips


def openai(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call OpenAI GPT-4o."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenAI → {BOLD}gpt-4o{RESET}")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    clips = _parse_llm_json(raw)
    log("OK", f"OpenAI returned {len(clips)} clips")
    return clips


def ollama(system: str, user: str) -> list[dict[str, Any]]:
    """Call local Ollama instance."""
    try:
        import requests
    except ImportError:
        log("ERROR", "requests not installed.  pip install requests")
        sys.exit(1)

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    log("LLM", f"Ollama (local) → {BOLD}{model}{RESET}")
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    clips = _parse_llm_json(raw)
    log("OK", f"Ollama returned {len(clips)} clips")
    return clips


def call_llm(
    system: str,
    user: str,
    api_key: str | None = None,
    llm_model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call the best available LLM backend and return raw parsed clips.

    Backend priority:
      1. OpenRouter  (OPENROUTER_API_KEY) — default free model
      2. Anthropic   (ANTHROPIC_API_KEY)
      3. OpenAI      (OPENAI_API_KEY)
      4. Ollama      (local, no key needed)
    """
    or_key = api_key or os.getenv("OPENROUTER_API_KEY")
    ant_key = os.getenv("ANTHROPIC_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")

    if or_key:
        model = llm_model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        return openrouter(system, user, or_key, model=model)
    elif ant_key:
        return anthropic(system, user, ant_key)
    elif oai_key:
        return openai(system, user, oai_key)
    else:
        log("WARN", "No API key found. Trying local Ollama …")
        return ollama(system, user)
