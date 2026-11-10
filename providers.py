"""External AI providers: Groq (LLM, OpenAI-compatible) and Tavily (web search).

Both have free tiers that HARD-STOP at their limits (no card on file → no bill),
and the app adds its own daily caps well below them (see limits.py). Any
provider rate-limit / quota response is raised as ``ProviderLimit`` so the web
app can show the friendly limit card instead of an error.

Local dev without keys: set SEARCHIQ_FAKE_PROVIDERS=1 to get canned answers and
search results (ignored on Vercel). SEARCHIQ_SIMULATE_GROQ_429=1 makes every
Groq call fail with a simulated 429 (also local-only) — used by the tests.
"""

from __future__ import annotations

import os
import threading
import time

import requests

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Fast tier. Groq retired llama-3.1-8b-instant for free/developer keys on
# 2026-08-16 and recommends openai/gpt-oss-20b as its replacement — override with
# GROQ_FAST_MODEL if your account still serves another fast model.
FAST_MODEL = (os.environ.get("GROQ_FAST_MODEL") or "openai/gpt-oss-20b").strip()
DEEP_MODEL = (os.environ.get("GROQ_DEEP_MODEL") or "openai/gpt-oss-120b").strip()


def model_label(model: str) -> str:
    """Short UI label for a Groq model id, e.g. 'GPT-OSS 20B', 'GPT-OSS 120B'."""
    m = (model or "").lower()
    known = {
        "openai/gpt-oss-120b": "GPT-OSS 120B",
        "openai/gpt-oss-20b": "GPT-OSS 20B",
    }
    if m in known:
        return known[m]
    return model.split("/")[-1]


FAST_LABEL = model_label(FAST_MODEL)
DEEP_LABEL = model_label(DEEP_MODEL)

# Groq on-demand list prices, USD per 1M tokens (input, output) — used ONLY for
# the dashboard's labelled cost *estimate* (the app itself runs on the free tier).
# Source: console.groq.com/docs/models, checked 2026-11-09.
LIST_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# The live rewrite + re-check (the check acting, arm D) run on a separate
# prepaid OpenRouter key, pinned to Groq with fallbacks off, so a flagged
# question doesn't run out of the free key's per-minute token limit. Same
# prompts and model as locked eval arm D; override with SEARCHIQ_REWRITE_MODEL.
REWRITE_MODEL = (os.environ.get("SEARCHIQ_REWRITE_MODEL") or FAST_MODEL).strip()
TAVILY_URL = "https://api.tavily.com/search"
HTTP_TIMEOUT_S = 60


class ProviderLimit(Exception):
    """A provider refused for quota/rate reasons (HTTP 429, or our own budget)."""

    def __init__(self, provider: str, message: str = ""):
        super().__init__(message or f"{provider} rate limit reached")
        self.provider = provider


class ProviderError(Exception):
    """Any other provider failure (auth, network, 5xx)."""


def _local_only(flag: str) -> bool:
    return os.environ.get(flag) == "1" and not os.environ.get("VERCEL")


def fake_mode() -> bool:
    return _local_only("SEARCHIQ_FAKE_PROVIDERS")


def configured() -> bool:
    return fake_mode() or bool(os.environ.get("GROQ_API_KEY") and os.environ.get("TAVILY_API_KEY"))


# --------------------------------------------------------------------------- #
# Groq chat completions
# --------------------------------------------------------------------------- #
_http = threading.local()


def _session() -> requests.Session:
    s = getattr(_http, "s", None)
    if s is None:
        s = _http.s = requests.Session()
    return s


# Optional hook: limits.py registers a function(model, total_tokens) so every
# call's token usage is counted against the app's daily per-model budget.
usage_hook = None
# Optional hook: function(model) → raises ProviderLimit when today's budget for
# that model is spent (checked BEFORE the call).
budget_check = None


def _parse_reset(v: str) -> float:
    """Groq reset headers look like '47.2s', '1m2.5s' or '850ms'."""
    import re as _re
    total = 0.0
    for num, unit in _re.findall(r"([\d.]+)(ms|m|s|h)", v or ""):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def chat(model: str, messages: list[dict], max_tokens: int = 800,
         system: str | None = None, effort: str = "low", max_wait: float = 6, on_wait=None) -> str:
    """One chat completion on Groq; returns the assistant text (see chat_full)."""
    return chat_full(model, messages, max_tokens=max_tokens, system=system, effort=effort,
                     max_wait=max_wait, on_wait=on_wait)[0]


def openrouter_chat(model: str, messages: list[dict], max_tokens: int = 800,
                    system: str | None = None, effort: str = "low", temperature: float = 0.4) -> str:
    """One completion via OpenRouter, pinned to Groq with fallbacks off.

    Raises ProviderError on ANY failure (no key, out of credit, expired key, a
    non-Groq provider, network) so the caller can fall back to the free key."""
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    if fake_mode():
        if os.environ.get("SEARCHIQ_FAKE_OPENROUTER_DOWN") == "1":
            raise ProviderError("OpenRouter unavailable (simulated)")
        last = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        return _fake_answer(model, last, system or "")
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ProviderError("OPENROUTER_API_KEY not set")
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "provider": {"order": ["Groq"], "allow_fallbacks": False}}
    if _is_reasoning_model(model):
        body["reasoning"] = {"effort": effort, "exclude": True}
    try:
        resp = _session().post(f"{OPENROUTER_URL}/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {key}"}, timeout=HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach OpenRouter: {exc.__class__.__name__}") from exc
    if resp.status_code >= 400:
        raise ProviderError(f"OpenRouter error {resp.status_code}")
    data = resp.json()
    if data.get("error"):
        raise ProviderError(f"OpenRouter error: {str(data['error'])[:120]}")
    if data.get("provider") != "Groq":
        raise ProviderError(f"OpenRouter served {model} from {data.get('provider')!r}, not Groq")
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    if not text:
        raise ProviderError("OpenRouter returned an empty answer")
    return text


def chat_full(model: str, messages: list[dict], max_tokens: int = 800,
              system: str | None = None, effort: str = "low",
              temperature: float = 0.4, max_wait: float = 6, on_wait=None) -> tuple[str, dict]:
    """One chat completion on Groq → (text, usage). usage has prompt_tokens /
    completion_tokens / total_tokens. The live app uses temperature 0.4; the
    eval passes 0 for repeatability.

    gpt-oss models are reasoning models: ``reasoning_effort`` (low|medium|high)
    sets how long they think, and include_reasoning=False keeps the thinking out
    of the response. Reasoning tokens count toward max_tokens AND the free-tier
    token budget, so the effort is kept low by default.
    """
    if budget_check:
        budget_check(model)
    if _local_only("SEARCHIQ_SIMULATE_GROQ_429"):
        raise ProviderLimit("groq", "Simulated Groq 429 (SEARCHIQ_SIMULATE_GROQ_429=1)")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    if fake_mode():
        last = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        text = _fake_answer(model, last, system or "")
        if usage_hook:
            usage_hook(model, 500)
        return text, {"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500}

    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": temperature}
    if _is_reasoning_model(model):
        body["reasoning_effort"] = effort
        body["include_reasoning"] = False
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    resp = None
    for attempt in range(2):
        try:
            resp = _session().post(
                f"{GROQ_BASE_URL}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {key}"}, timeout=HTTP_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Could not reach Groq: {exc.__class__.__name__}") from exc
        if resp.status_code != 429:
            break
        # Per-minute token limits reset fast: wait once if Groq says it's short.
        try:
            wait = float(resp.headers.get("retry-after", "0"))
        except ValueError:
            wait = 0
        if not wait:  # per-minute token limit: Groq says when the bucket refills
            wait = _parse_reset(resp.headers.get("x-ratelimit-reset-tokens", ""))
        if attempt == 0 and 0 < wait <= max_wait:
            if on_wait:
                on_wait(True)
            time.sleep(wait)
            if on_wait:
                on_wait(False)
            continue
        raise ProviderLimit("groq", "Groq free-tier rate limit reached")
    if resp.status_code == 429:
        raise ProviderLimit("groq", "Groq free-tier rate limit reached")
    if resp.status_code >= 400:
        raise ProviderError(f"Groq error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    usage = data.get("usage") or {}
    if usage_hook:
        usage_hook(model, int(usage.get("total_tokens") or 0))
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    return text, {k: int(usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


# --------------------------------------------------------------------------- #
# Tavily search
# --------------------------------------------------------------------------- #
def search(query: str, max_results: int = 3) -> list[dict]:
    """Basic Tavily search (1 credit). Returns [{title, url, content, published_date}]."""
    if fake_mode():
        return [
            {"title": f"Example source {i} for {query[:40]}", "url": f"https://example.com/{i}",
             "content": f"Fake search snippet {i} about {query}.", "published_date": None}
            for i in range(1, max_results + 1)
        ]
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    try:
        resp = _session().post(
            TAVILY_URL,
            json={"query": query[:400], "search_depth": "basic", "max_results": max_results,
                  "include_answer": False, "include_raw_content": False},
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach Tavily: {exc.__class__.__name__}") from exc
    if resp.status_code in (429, 432, 433):  # rate limit / plan or credit limit
        raise ProviderLimit("tavily", "Tavily free-tier limit reached")
    if resp.status_code >= 400:
        raise ProviderError(f"Tavily error {resp.status_code}: {resp.text[:200]}")
    out = []
    for r in resp.json().get("results", [])[:max_results]:
        out.append({
            "title": r.get("title") or r.get("url"),
            "url": r.get("url"),
            "content": (r.get("content") or "")[:1200],
            "published_date": r.get("published_date"),
        })
    return out


# --------------------------------------------------------------------------- #
# Fake provider output (local dev only)
# --------------------------------------------------------------------------- #
def _fake_answer(model: str, prompt: str, system: str) -> str:
    label = model_label(model)
    if "SearchIQ's editor" in system:
        return ("Einstein won only one Nobel Prize: the 1921 Nobel Prize in Physics, awarded in 1922 [1][2]. "
                "There was no second prize.")
    if "fact checker" in system and "second Nobel" in prompt:  # demo of a flagged premise
        return ('{"premise":{"ok":false,"note":"Einstein won only one Nobel Prize (Physics, 1921)."},"claims":['
                '{"side":"A","text":"Einstein won one Nobel Prize, in Physics","status":"backed","sources":[1],"note":""},'
                '{"side":"B","text":"Einstein won a second Nobel Prize in 1922","status":"contradicted","sources":[2],'
                '"note":"Source 2 says the 1921 prize was awarded in 1922; there was no second prize."}],'
                '"disagreements":[{"topic":"Number of Nobel Prizes","A":"One (1921)","B":"Two (second in 1922)",'
                '"supported":"A","sources":[1,2]}]}')
    if "fact checker" in system:
        return ('{"premise":{"ok":true,"note":""},"claims":['
                '{"side":"A","text":"Fake claim one","status":"backed","sources":[1],"note":""},'
                '{"side":"B","text":"Fake claim one","status":"backed","sources":[1],"note":""},'
                '{"side":"B","text":"Fake claim two","status":"unsupported","sources":[],"note":""}],'
                '"disagreements":[]}')
    if "follow-up questions" in prompt:
        return "What are the key details?\nHow has this changed recently?\nWhere can I learn more?"
    if "clarifying question" in prompt:
        return "Happy to help! What will you use it for, and is it for personal use or a team?"
    if "Rewrite the query" in prompt:
        return prompt.split("Original query:", 1)[-1].split("\n", 1)[0].strip() + " (clarified)"
    style = "detailed" if "DETAILED" in system else "concise"
    return (f"**[{label} · {style}]** Placeholder answer for: {prompt.splitlines()[-1][:120]}\n\n"
            "- Grounded in the provided sources [1]\n- Generated in fake-provider mode [2]")
