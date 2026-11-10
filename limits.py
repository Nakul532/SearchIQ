"""Free-tier guard rails: daily caps, provider token budgets, and the answer cache.

Everything is stored in two small tables (usage_counters, answer_cache) in the
same database as the app data, so it works across stateless Vercel functions.
Days roll over at midnight UTC.

Caps (all overridable via environment variables):
  GUEST_DAILY_CAP        3    live searches per guest per day (cookie id)
  GUEST_IP_DAILY_CAP    10    ... per hashed IP — survives clearing cookies
                              (higher than 3 so shared networks/NATs aren't starved)
  USER_DAILY_CAP        10    live searches per signed-in Google user per day
  SITE_DAILY_CAP        25    live searches site-wide per day   (~750 Tavily/mo)
  SITE_MONTHLY_CAP     900    Tavily credits per month          (free tier: 1,000)
  FAST_DAILY_TOKENS 150000    Groq tokens/day on the fast model (free: 200K TPD)
  DEEP_DAILY_TOKENS 150000    Groq tokens/day on the deep model (free: 200K TPD)

A "live search" = one query that runs a Tavily search + fresh Groq answers.
Cache hits, A/B picks, reasons and follow-up chips never count against a cap.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re

import providers


def _cap(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


GUEST_DAILY_CAP = _cap("GUEST_DAILY_CAP", 3)
GUEST_IP_DAILY_CAP = _cap("GUEST_IP_DAILY_CAP", 10)
USER_DAILY_CAP = _cap("USER_DAILY_CAP", 10)
SITE_DAILY_CAP = _cap("SITE_DAILY_CAP", 25)
SITE_MONTHLY_CAP = _cap("SITE_MONTHLY_CAP", 900)
TOKEN_BUDGETS = {
    providers.FAST_MODEL: _cap("FAST_DAILY_TOKENS", 150_000),
    providers.DEEP_MODEL: _cap("DEEP_DAILY_TOKENS", 150_000),
}

CACHE_TTL_DAYS = _cap("CACHE_TTL_DAYS", 7)

# The three example chips on the welcome screen — always served from cache
# (pinned, never expire).
EXAMPLE_QUERIES = (
    "Who founded Nvidia, and in what year?",
    "What was Apple's total net sales in fiscal year 2023?",
    "When did Microsoft complete its acquisition of Activision Blizzard?",
)

# Time-sensitive queries are never served from (or written to) the cache, so
# users don't get stale news/prices.
_TIME_SENSITIVE = re.compile(
    r"\b(today|todays|today's|tonight|yesterday|tomorrow|latest|current|currently|"
    r"now|right now|news|breaking|this week|this month|this year|recent|recently|"
    r"price|prices|pricing|stock|stocks|score|scores|weather|live)\b",
    re.I,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS usage_counters (
        key        TEXT PRIMARY KEY,   -- e.g. 'd:2026-11-09:site', 'm:2026-11:tavily'
        count      INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS answer_cache (
        key        TEXT PRIMARY KEY,   -- normalized query text
        query      TEXT NOT NULL,      -- as first asked
        payload    TEXT NOT NULL,      -- JSON: kind + answers/sources/routing (or clarify)
        pinned     INTEGER NOT NULL DEFAULT 0,  -- 1 = example chip, never expires
        created_at TEXT NOT NULL
    )""",
]


# --------------------------------------------------------------------------- #
# Time helpers (UTC)
# --------------------------------------------------------------------------- #
def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _day() -> str:
    return _now().strftime("%Y-%m-%d")


def _month() -> str:
    return _now().strftime("%Y-%m")


def resets_at() -> str:
    """ISO timestamp of the next UTC midnight (when daily caps reset)."""
    tomorrow = (_now() + _dt.timedelta(days=1)).date()
    return _dt.datetime.combine(tomorrow, _dt.time(0, 0), tzinfo=_dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #
def _get(conn, key: str) -> int:
    row = conn.execute("SELECT count FROM usage_counters WHERE key = ?", (key,)).fetchone()
    return int(row[0]) if row else 0


def _incr(conn, key: str, by: int = 1) -> None:
    conn.execute(
        "INSERT INTO usage_counters (key, count, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET count = count + excluded.count, "
        "updated_at = excluded.updated_at",
        (key, by, _now().isoformat(timespec="seconds")),
    )
    conn.commit()


def hash_ip(ip: str | None) -> str:
    """Salted, truncated hash — raw IPs are never stored."""
    salt = os.environ.get("FLASK_SECRET_KEY", "dev-salt")
    return hashlib.sha256(f"{salt}|{ip or 'unknown'}".encode()).hexdigest()[:24]


def _keys(identity: str, signed_in: bool, ip_hash: str) -> dict[str, str]:
    d, m = _day(), _month()
    keys = {"site": f"d:{d}:site", "month": f"m:{m}:tavily"}
    if signed_in:
        keys["user"] = f"d:{d}:user:{identity}"
    else:
        keys["guest"] = f"d:{d}:guest:{identity}"
        keys["ip"] = f"d:{d}:ip:{ip_hash}"
    return keys


def check_live_search(conn, identity: str, signed_in: bool, ip_hash: str) -> dict | None:
    """Return a limit dict if this caller may NOT run a live search now, else None."""
    k = _keys(identity, signed_in, ip_hash)
    if _get(conn, k["month"]) >= SITE_MONTHLY_CAP or _get(conn, k["site"]) >= SITE_DAILY_CAP:
        return limit_info("site")
    if signed_in:
        if _get(conn, k["user"]) >= USER_DAILY_CAP:
            return limit_info("user")
    elif _get(conn, k["guest"]) >= GUEST_DAILY_CAP or _get(conn, k["ip"]) >= GUEST_IP_DAILY_CAP:
        return limit_info("guest")
    for model, budget in TOKEN_BUDGETS.items():
        if _get(conn, f"d:{_day()}:tokens:{model}") >= budget:
            return limit_info("site")
    return None


def record_search_spend(conn) -> None:
    """Count one Tavily credit (site-wide daily + monthly) — called when it's spent."""
    _incr(conn, f"d:{_day()}:site")
    _incr(conn, f"m:{_month()}:tavily")


def record_caller_search(conn, identity: str, signed_in: bool, ip_hash: str) -> None:
    """Count one live search against the caller (only once an answer was delivered)."""
    k = _keys(identity, signed_in, ip_hash)
    if signed_in:
        _incr(conn, k["user"])
    else:
        _incr(conn, k["guest"])
        _incr(conn, k["ip"])


def remaining(conn, identity: str, signed_in: bool, ip_hash: str) -> dict:
    """Searches left today for this caller (min of their cap and the site cap)."""
    k = _keys(identity, signed_in, ip_hash)
    site_left = max(0, min(SITE_DAILY_CAP - _get(conn, k["site"]),
                           SITE_MONTHLY_CAP - _get(conn, k["month"])))
    if signed_in:
        mine = USER_DAILY_CAP - _get(conn, k["user"])
        cap = USER_DAILY_CAP
    else:
        mine = min(GUEST_DAILY_CAP - _get(conn, k["guest"]),
                   GUEST_IP_DAILY_CAP - _get(conn, k["ip"]))
        cap = GUEST_DAILY_CAP
    return {"remaining": max(0, min(mine, site_left)), "cap": cap, "signed_in": signed_in,
            "resets_at": resets_at()}


def install_token_budget(conn_factory) -> None:
    """Wire providers.chat() to count tokens and refuse once a model's budget is spent."""

    def _hook(model: str, tokens: int) -> None:
        if tokens <= 0:
            return
        conn = conn_factory()
        try:
            _incr(conn, f"d:{_day()}:tokens:{model}", tokens)
        finally:
            conn.close()

    def _check(model: str) -> None:
        budget = TOKEN_BUDGETS.get(model)
        if not budget:
            return
        conn = conn_factory()
        try:
            if _get(conn, f"d:{_day()}:tokens:{model}") >= budget:
                raise providers.ProviderLimit("groq", f"Daily token budget for {model} reached")
        finally:
            conn.close()

    providers.usage_hook = _hook
    providers.budget_check = _check


# --------------------------------------------------------------------------- #
# Limit card payload
# --------------------------------------------------------------------------- #
_LIMIT_COPY = {
    "guest": ("You've used today's {cap} free searches.",
              "Free searches reset at midnight UTC. Meanwhile, try a cached example below."),
    "user": ("You've used today's {cap} searches.",
             "Your searches reset at midnight UTC. Meanwhile, try a cached example below."),
    "site": ("SearchIQ has reached today's free search limit.",
             "This demo runs entirely on free tiers, so live searches are capped "
             "site-wide. They reset at midnight UTC. Meanwhile, try a cached example below."),
}


def limit_info(reason: str, examples: list[dict] | None = None) -> dict:
    """The friendly limit card shown instead of an error."""
    cap = {"guest": GUEST_DAILY_CAP, "user": USER_DAILY_CAP}.get(reason, SITE_DAILY_CAP)
    title, body = _LIMIT_COPY.get(reason, _LIMIT_COPY["site"])
    return {
        "type": "limit",
        "reason": reason,
        "title": title.format(cap=cap),
        "message": body,
        "resets_at": resets_at(),
        "examples": examples if examples is not None else [],
    }


# --------------------------------------------------------------------------- #
# Answer cache
# --------------------------------------------------------------------------- #
def normalize(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    return q.strip(" ?!.\"'")


_EXAMPLE_KEYS = {normalize(q) for q in EXAMPLE_QUERIES}


def is_example(query: str) -> bool:
    return normalize(query) in _EXAMPLE_KEYS


def is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE.search(query or ""))


def cacheable(query: str) -> bool:
    return is_example(query) or not is_time_sensitive(query)


def cache_get(conn, query: str) -> dict | None:
    """Cached payload (+ cached_at) for this query, or None (expired/absent/uncacheable)."""
    if not cacheable(query):
        return None
    row = conn.execute(
        "SELECT payload, pinned, created_at FROM answer_cache WHERE key = ?",
        (normalize(query),),
    ).fetchone()
    if not row:
        return None
    payload, pinned, created_at = row
    if not pinned:
        try:
            age = _now() - _dt.datetime.fromisoformat(created_at)
        except ValueError:
            return None
        if age > _dt.timedelta(days=CACHE_TTL_DAYS):
            return None
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    data["cached_at"] = created_at
    return data


def cache_put(conn, query: str, payload: dict) -> None:
    if not cacheable(query):
        return
    conn.execute(
        "INSERT INTO answer_cache (key, query, payload, pinned, created_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET query = excluded.query, payload = excluded.payload, "
        "pinned = excluded.pinned, created_at = excluded.created_at",
        (normalize(query), query, json.dumps(payload), 1 if is_example(query) else 0,
         _now().isoformat(timespec="seconds")),
    )
    conn.commit()


def cached_examples(conn) -> list[dict]:
    """The example chips that are actually in the cache (for the limit card)."""
    out = []
    for q in EXAMPLE_QUERIES:
        row = conn.execute("SELECT created_at, payload FROM answer_cache WHERE key = ?",
                           (normalize(q),)).fetchone()
        # An A/B example cached before the v2 agent (no "check") is regenerated on
        # its next request, so don't offer it as a free cached answer yet.
        if row and not ('"kind": "ab"' in (row[1] or "") and '"check"' not in (row[1] or "")):
            out.append({"query": q, "cached_at": row[0]})
    return out
