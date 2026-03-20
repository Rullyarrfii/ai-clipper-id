"""
LLM backends: OpenRouter, Anthropic, OpenAI, Ollama.
"""

import json
import os
import re
import sys
import time
from typing import Any, Callable

from ..utils import log, BOLD, RESET, DEFAULT_OPENROUTER_MODEL, DEFAULT_OPENROUTER_BASE


def _retry_on_rate_limit(
    api_call_fn: Callable,
    max_retries: int = 5,
    initial_wait: float = 2.0,
) -> str:
    """
    Retry API call on rate limit errors with exponential backoff.
    
    Args:
        api_call_fn: Callable that makes the API call (no args)
        max_retries: Maximum number of retries
        initial_wait: Initial wait time in seconds before first retry
    
    Returns:
        Response text from successful API call
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return api_call_fn()
        except Exception as e:
            # Check for rate limit error (429)
            is_rate_limit = (
                "429" in str(e) or 
                "RateLimitError" in type(e).__name__ or
                "rate" in str(e).lower()
            )
            
            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                raise
            
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                log("WARN", f"Rate limited (429). Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                log("ERROR", f"Rate limit persisted after {max_retries} retries")
                raise


def _parse_llm_json(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Robustly extract a JSON array from LLM text.
    
    Returns:
        (parsed_data, success) — success=False if parsing failed
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data, True
        # Wrapped in an object — find the array
        for v in data.values():
            if isinstance(v, list):
                return v, True
        return [], True
    except json.JSONDecodeError:
        pass
    # Last resort: find first [ ... ] block
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), True
        except json.JSONDecodeError:
            pass
    return [], False


def _retry_on_json_failure(
    api_call_fn,
    system: str,
    user: str,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    Retry LLM call with modified prompts if JSON parsing fails.
    
    Args:
        api_call_fn: Callable that takes (system, user) and returns raw response text
        system: System prompt
        user: User prompt
        max_retries: Number of retries (1 = no retry, just one attempt)
    
    Returns:
        Parsed clips, or empty list if all attempts fail
    """
    retry_prompts = [
        "",  # First attempt: original prompt
        "\n\n[RETRY: Please respond with ONLY a valid JSON array, no other text. Start with [ and end with ].]",
        "\n\n[RETRY 2: Output ONLY valid JSON array. Example format: [{\"start\": 0, \"end\": 10, \"reason\": \"...\"}, ...].]",
    ]
    
    for attempt in range(min(max_retries, len(retry_prompts))):
        modified_user = user + retry_prompts[attempt]
        raw = api_call_fn(system, modified_user)
        clips, success = _parse_llm_json(raw)
        
        if success:
            return clips
        
        if attempt < max_retries - 1:
            log("WARN", f"JSON parse failed (attempt {attempt + 1}), retrying with modified prompt...")
    
    log("ERROR", "Could not parse JSON from LLM response after retries")
    return []


def openrouter(
    system: str,
    user: str,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE,
) -> list[dict[str, Any]]:
    """Call OpenRouter (OpenAI-compatible API). Default: free model. With retry on JSON parse failure and rate limits."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenRouter → {BOLD}{model}{RESET}")
    
    def _call_openrouter(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call with rate limit retry."""
        def api_call():
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=8192,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/ai-video-clipper",
                    "X-Title": "AI Video Clipper",
                },
                extra_body={
                    "reasoning": {
                        "effort": "high"
                    }
                }
            )
            reasoning = getattr(resp.choices[0].message, "reasoning", None)
            if reasoning:
                log("LLM-REASONING", "Reasoning generated for this response.")
            return resp.choices[0].message.content or ""
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_openrouter, system, user, max_retries=2)
    log("OK", f"OpenRouter returned {len(clips)} clips")
    return clips


def anthropic(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call Anthropic Claude. With retry on JSON parse failure and rate limits."""
    try:
        import anthropic
    except ImportError:
        log("ERROR", "anthropic SDK not installed.  pip install anthropic")
        sys.exit(1)

    log("LLM", f"Anthropic → {BOLD}claude-sonnet-4-20250514{RESET}")
    
    def _call_anthropic(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call with rate limit retry."""
        def api_call():
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_anthropic, system, user, max_retries=2)
    log("OK", f"Anthropic returned {len(clips)} clips")
    return clips



def openai(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call OpenAI GPT-4o. With retry on JSON parse failure and rate limits."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenAI → {BOLD}gpt-4o{RESET}")
    
    def _call_openai(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call with rate limit retry."""
        def api_call():
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={
                    "reasoning": {
                        "effort": "high"
                    }
                }
            )
            reasoning = getattr(resp.choices[0].message, "reasoning", None)
            if reasoning:
                log("LLM-REASONING", "Reasoning generated for this response.")
            return resp.choices[0].message.content or ""
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_openai, system, user, max_retries=2)
    log("OK", f"OpenAI returned {len(clips)} clips")
    return clips



def ollama(system: str, user: str) -> list[dict[str, Any]]:
    """Call local Ollama instance. With retry on JSON parse failure and rate limits."""
    try:
        import requests
    except ImportError:
        log("ERROR", "requests not installed.  pip install requests")
        sys.exit(1)

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    log("LLM", f"Ollama (local) → {BOLD}{model}{RESET}")
    
    def _call_ollama(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call with rate limit retry."""
        def api_call():
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_ollama, system, user, max_retries=2)
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
