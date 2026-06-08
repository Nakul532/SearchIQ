"""Search Evaluation Agent.

A command-line agent that:
  1. Takes a user search query.
  2. Generates an answer with the Anthropic API (Claude).
  3. Collects thumbs up/down feedback + an optional comment.
  4. Persists every query, answer, and piece of feedback to a local SQLite DB.
  5. Learns over time: queries that get bad feedback are rephrased and retried,
     and accumulated feedback is fed back into the system prompt as "lessons".
  6. Tracks a North Star Metric: Task Completion Rate (% of queries that
     eventually earned positive feedback).
  7. Prints a dashboard after every 5 queries.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python search_agent.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

import anthropic

# Load a local .env (if present) so ANTHROPIC_API_KEY / DB_PATH etc. can live in
# a file instead of being exported each session. Loaded here at import time —
# app.py imports this module, so both the web app and the CLI pick it up before
# any environment variable is read. Real environment vars take precedence over
# .env values (override=False), and a missing python-dotenv is non-fatal.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# --------------------------------------------------------------------------- #
# Model routing tiers
#
# Pipeline: EVERY query runs web search first (always — the retrieval layer
# never changes). A rule-based pre-processor (classify_query) then sizes the
# query and picks which model AGGREGATES the same search results into the answer:
#   simple  → Haiku  (factual / definitional / single facts) — fast
#   medium  → Haiku  (comparisons / explanations / how-does-X-work) — balanced
#   complex → Sonnet (analysis / research / multi-step reasoning / synthesis) — deep
# Routing simple/medium queries to Haiku avoids an expensive Sonnet call on the
# bulk of traffic; the dashboard surfaces the model mix and the cost saved.
# --------------------------------------------------------------------------- #
HAIKU_MODEL = "claude-haiku-4-5-20251001"   # fast, cheap
SONNET_MODEL = "claude-sonnet-4-6"          # capable, for complex reasoning

# Kept for the rephrase step (a small, cheap classification-style call).
MODEL = HAIKU_MODEL

# Per-tier routing config — model, badge icon, and the one-word "mode" label.
TIERS = {
    "simple":  {"model": HAIKU_MODEL,  "icon": "⚡", "mode": "Fast"},
    "medium":  {"model": HAIKU_MODEL,  "icon": "🔍", "mode": "Balanced"},
    "complex": {"model": SONNET_MODEL, "icon": "🧠", "mode": "Deep"},
}

# Rough per-query cost estimate, used only for the "cost saved vs always-Sonnet"
# dashboard stat (we don't meter real tokens). Assumes a typical aggregation call
# that includes web-search results in the prompt.
AVG_INPUT_TOKENS = 3000
AVG_OUTPUT_TOKENS = 800
MODEL_PRICING = {  # USD per 1M tokens: (input, output)
    HAIKU_MODEL: (1.0, 5.0),
    SONNET_MODEL: (3.0, 15.0),
}


def estimate_query_cost(model: str) -> float:
    """Estimated USD cost of one aggregation call on ``model`` (see notes above)."""
    pin, pout = MODEL_PRICING.get(model, MODEL_PRICING[SONNET_MODEL])
    return AVG_INPUT_TOKENS / 1e6 * pin + AVG_OUTPUT_TOKENS / 1e6 * pout


def _resolve_db_path() -> str:
    """Pick where the SQLite DB lives, honoring a persistent volume if present.

    Priority:
      1. DB_PATH env var — explicit full path to the .db file.
      2. RAILWAY_VOLUME_MOUNT_PATH — Railway injects this when a volume is
         attached; the DB goes inside it as search_agent.db so data survives
         redeploys/restarts.
      3. Local file next to this script (default for local dev).
    The parent directory is created if needed.
    """
    explicit = os.environ.get("DB_PATH")
    volume = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if explicit:
        path = explicit
    elif volume:
        path = os.path.join(volume, "search_agent.db")
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_agent.db")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


DB_PATH = _resolve_db_path()
MAX_RETRIES = 2          # how many rephrase-and-retry rounds per failed query
DASHBOARD_EVERY = 5      # show the dashboard after this many queries
LESSON_CONTEXT_LIMIT = 8 # how many recent feedback lessons to feed the model

# --------------------------------------------------------------------------- #
# Reinforcement-learning reward signals
# --------------------------------------------------------------------------- #
# Numeric reward attached to every interaction outcome. Positive = the user was
# served well; negative = they were not. Aggregated into the Reward dashboard and
# used to learn a policy (which model + answer style earns the most reward).
REWARD_SCORES = {
    "ab_winner":    1.0,   # the answer the user picked in an A/B comparison
    "ab_loser":    -0.5,   # the answer they passed over
    "thumbs_up":    1.0,   # 👍 on a single (non-A/B) answer
    "thumbs_down": -1.0,   # 👎 on a single answer
    "followup":     0.2,   # user asked a follow-up — engaged enough to continue
}
# Bonus reward added when the user explains WHY they preferred an answer.
REASON_BONUS = {
    "more accurate":        0.3,
    "clearer explanation":  0.2,
    "better sources":       0.1,
}
# A/B side → answer style. A is concise/factual, B is detailed/analytical
# (matches STYLE_A / STYLE_B and the lessons winning_style convention).
STYLE_OF_SIDE = {"A": "concise", "B": "detailed"}
POLICY_EVERY = 5         # re-evaluate the policy after every N reward-bearing queries
PROFILE_EVERY = 5        # rebuild a user's personality profile after every N queries

# Anthropic's built-in server-side web search tool. Runs the search on
# Anthropic's infrastructure, feeds results into Claude's context, and returns
# citations — the RAG retrieval layer. GA, no beta header needed.
# allowed_callers=["direct"] disables the tool's programmatic-tool-calling
# (dynamic-filtering) path, which Haiku 4.5 does not support — without it, a
# Haiku request with this tool 400s. Direct calling works on every routed model.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    # Fewer search rounds → fewer results injected into the prompt → far fewer
    # tokens (the tool has no per-result cap), which keeps each answer fast.
    "max_uses": 2,
    "allowed_callers": ["direct"],
}
MAX_SOURCES = 3          # cap how many sources we surface per answer (top 3)
ANSWER_TIMEOUT_S = 90    # hard per-answer generation timeout (each A/B draft)


# --------------------------------------------------------------------------- #
# Query pre-processor — rule-based complexity classifier + model router
# --------------------------------------------------------------------------- #
# Signal words, checked in priority order (complex beats medium beats simple).
# Lower-cased substring matching against the query.
_COMPLEX_SIGNALS = (
    "analyze", "analyse", "analysis", "evaluate", "assess", "research",
    "implication", "impact of", "consequence", "pros and cons", "trade-off",
    "tradeoff", "compare and contrast", "in depth", "step by step", "step-by-step",
    "strategy", "architect", "design a", "design an", "framework for",
    "why does", "why do", "why is", "why are", "how would", "what would happen",
    "deep dive", "critique", "synthesize", "synthesis", "rank ", "recommend",
    "news", "latest news",  # news synthesis is a deep, multi-source task
)
_MEDIUM_SIGNALS = (
    "compare", "difference between", "vs ", " vs.", "versus", "explain",
    "how do", "how does", "how to", "how can", "latest", "current", "today",
    "recent", "this year", "update", "best ", "should i", "review",
    "guide", "tutorial", "examples of", "trends",
    # Live-data lookups — short, but a balanced (vs simple) framing fits better.
    "price", "stock", "weather", "score", "near me", "schedule", "release date",
)
_SIMPLE_STARTS = (
    "what is", "what's", "whats", "who is", "who's", "who was", "when is",
    "when did", "when was", "where is", "where was", "define", "definition of",
    "capital of", "how many", "how much", "what year", "what time", "meaning of",
)


def classify_query(query: str) -> dict:
    """Rule-based pre-processor: size a query and pick the AGGREGATION model.

    Web search always runs (the retrieval layer never changes) — this only
    decides which model turns the same search results into the answer. No API
    call; uses length, question words, and signal phrases. Returns a routing
    dict (see build_routing). Priority: complex > medium > simple > fallback.
    """
    q = (query or "").lower().strip()
    word_count = len(q.split())

    def has(signals: tuple[str, ...]) -> bool:
        return any(s in q for s in signals)

    # Complex: analysis / research / multi-step reasoning, "why", news synthesis,
    # or a long multi-part question.
    if has(_COMPLEX_SIGNALS) or word_count > 24 or q.count("?") > 1:
        complexity = "complex"
    # Medium: comparisons, explanations, "how does X work", "difference between".
    elif has(_MEDIUM_SIGNALS):
        complexity = "medium"
    # Simple: short factual/definitional lookups, "what is X", "who is X".
    elif q.startswith(_SIMPLE_STARTS) or word_count <= 6:
        complexity = "simple"
    # Fallback: when in doubt, the balanced tier.
    else:
        complexity = "medium"

    return build_routing(complexity)


def build_routing(complexity: str) -> dict:
    """Assemble a routing dict (model + UI badge) from a complexity tier.

    Web search is always on; the badge encodes tier, model, and mode, e.g.
    "⚡ Simple · Haiku · Fast" / "🔍 Medium · Haiku · Balanced" /
    "🧠 Complex · Sonnet · Deep".
    """
    tier = TIERS.get(complexity, TIERS["medium"])
    model = tier["model"]
    model_label = "Sonnet" if model.startswith("claude-sonnet") else "Haiku"
    badge = f"{tier['icon']} {complexity.capitalize()} · {model_label} · {tier['mode']}"
    return {
        "complexity": complexity,
        "model": model,
        "web_search": True,        # always — retrieval layer is constant
        "model_label": model_label,
        "mode": tier["mode"],
        "icon": tier["icon"],
        "badge": badge,
    }


def ab_routings(base_routing: dict) -> tuple[dict, dict]:
    """Per-side routing for the A/B pair.

    For COMPLEX queries, split the models so the user sees a good answer fast:
    Answer A uses Haiku (fast, concise) and Answer B uses Sonnet (thorough,
    detailed). For simpler tiers both sides keep the (policy-biased) base routing.
    The complex split intentionally overrides the policy's model for the A/B pair —
    that latency strategy is the whole point of the split.
    """
    if base_routing.get("complexity") == "complex":
        tier = TIERS["complex"]
        a = dict(base_routing)
        a.update({
            "model": HAIKU_MODEL,
            "model_label": "Haiku",
            "badge": f"{tier['icon']} Complex · Haiku · {tier['mode']}",
        })
        a.pop("policy_biased", None)  # this side is fixed to Haiku by design
        b = build_routing("complex")  # Sonnet
        return a, b
    return dict(base_routing), dict(base_routing)


def _reasoning_kwargs(model: str) -> dict:
    """Per-model thinking/effort params.

    Adaptive thinking and the effort parameter are supported on Sonnet 4.6 (and
    Opus) but ERROR on Haiku 4.5 — so only attach them for Sonnet. Haiku runs
    plain, which is exactly what the cheap/fast tier wants.
    """
    if model.startswith("claude-sonnet"):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    return {}


# --------------------------------------------------------------------------- #
# Clarifying questions — disambiguate vague queries before answering
# --------------------------------------------------------------------------- #
# Generic "vague" terms that signal an under-specified query.
_VAGUE_TERMS = {"best", "good", "top", "help", "recommend", "recommendation",
                "recommendations", "tips", "advice", "ideas", "options"}
# Category → a natural, conversational fallback question (used when the Haiku call
# to phrase the question is unavailable). First matching category wins.
_CLARIFY_CATEGORIES = [
    (("tool", "tools", "software", "app", "apps", "platform", "framework",
      "library", "stack", "service"),
     "Happy to help you find the right fit! Could you tell me a bit more about "
     "what you're looking for — what would you use it for, and is this for personal "
     "use or for a team?"),
    (("learn", "study", "course", "tutorial", "book", "books"),
     "I'd love to point you in the right direction — what's your current level with "
     "this, and is it for work or personal interest?"),
    (("buy", "cheap", "budget", "price", "phone", "laptop", "car", "gift"),
     "Sure! To narrow it down, what's your rough budget and what will you mainly "
     "use it for?"),
    (("help", "advice", "tips", "ideas"),
     "Of course — could you tell me a bit more about what you're trying to do, and "
     "what you've already tried?"),
]
_CLARIFY_DEFAULT = (
    "Could you tell me a bit more about what you're looking for? A little extra "
    "context will help me give you a more useful answer."
)


def _clarify_category(query: str) -> tuple[bool, str | None]:
    """Rule-based ambiguity check (no API). Returns (ambiguous, fallback_question).

    Ambiguous = very short (≤3 words) or a vague generic term in a short query.
    Specific multi-word queries (even with 'best') are left alone. The fallback
    question is chosen from the query's category.
    """
    q = (query or "").lower().strip()
    words = [w.strip(".,!?\"'()") for w in q.split()]
    word_count = len(words)
    has_vague = any(w in _VAGUE_TERMS for w in words)

    ambiguous = word_count <= 3 or (has_vague and word_count <= 4)
    if not ambiguous:
        return False, None
    for keywords, question in _CLARIFY_CATEGORIES:
        if any(w in keywords for w in words):
            return True, question
    return True, _CLARIFY_DEFAULT


def clarifying_question(client: anthropic.Anthropic, query: str) -> str | None:
    """If the query is ambiguous, return ONE natural clarifying question to ask.

    The agent asks this in the chat like a person would; the user answers in their
    own words (no chips/buttons), and that reply — together with this question, both
    kept in the conversation history — gives the model the context to answer the
    original query. The question is phrased by Haiku for a human touch and tailored
    to the query; on any API failure it falls back to a templated question for the
    query's category. Returns None when the query is specific enough to answer.
    """
    ambiguous, fallback = _clarify_category(query)
    if not ambiguous:
        return None
    prompt = (
        f'A user sent this short, ambiguous request: "{query}"\n\n'
        "Before answering, ask them ONE warm, natural clarifying question — the way a "
        "helpful person would in conversation — to learn what they actually need. "
        "Reference what they asked about and weave in one or two concrete angles "
        "(for example: the purpose, personal vs. professional use, budget, or skill "
        "level) so it's easy to answer. Keep it to one or two friendly sentences. "
        "Reply with ONLY the question — no preamble, no quotes, no lists."
    )
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip().strip('"').strip()
        if text:
            return text
    except anthropic.APIError:
        pass
    return fallback


def sanitize_env_secret(name: str) -> str | None:
    """Strip non-ASCII chars (e.g. a pasted U+2028) and whitespace from a secret.

    HTTP headers must be ASCII, so a stray Unicode character in the API key
    raises "'ascii' codec can't encode character ...". Anthropic keys are pure
    ASCII, so dropping non-ASCII bytes can only remove paste artifacts, never a
    valid key character. The cleaned value is written back to the environment so
    every client (web app and CLI) uses it.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    cleaned = "".join(ch for ch in raw if ord(ch) < 128).strip()
    if cleaned != raw:
        os.environ[name] = cleaned
        print(f"Note: removed non-ASCII/whitespace characters from {name}.")
    return cleaned or None


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def init_db(conn: sqlite3.Connection) -> None:
    """Create the interactions table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            query_group     INTEGER NOT NULL,  -- groups an original query + its retries
            original_query  TEXT    NOT NULL,
            effective_query TEXT    NOT NULL,  -- rephrased query, or the original
            answer          TEXT    NOT NULL,
            feedback        TEXT,              -- 'up' | 'down' | NULL (skipped)
            comment         TEXT,
            attempt         INTEGER NOT NULL,  -- 1 = first try, 2+ = retry
            created_at      TEXT    NOT NULL
        )
        """
    )
    # A/B answer comparisons — two answers per query, plus the user's preference.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comparisons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            query_group     INTEGER NOT NULL,
            session_id      TEXT,              -- per-browser session (see localStorage)
            conversation_id TEXT,              -- which chat this belongs to
            original_query  TEXT    NOT NULL,
            effective_query TEXT    NOT NULL,  -- after a clarifying chip is appended
            answer_a        TEXT    NOT NULL,
            answer_b        TEXT    NOT NULL,
            sources_a       TEXT,              -- JSON list of {url,title}
            sources_b       TEXT,
            routing         TEXT,              -- JSON routing decision (model used)
            preferred       TEXT,              -- 'A' | 'B' | NULL (not yet chosen)
            reason          TEXT,              -- why the user preferred it
            lessons_applied TEXT,              -- JSON list of lesson ids applied here
            created_at      TEXT    NOT NULL
        )
        """
    )
    # Lessons the system has learned from preferences — the visible learning loop.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lessons (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,                -- lessons are scoped per session/user
            topic         TEXT    NOT NULL,   -- extracted query topic ('tools', 'python', …)
            winning_style TEXT    NOT NULL,   -- 'A' (concise) | 'B' (detailed)
            reason        TEXT,               -- 'more accurate' | 'clearer explanation' | 'better sources'
            text          TEXT    NOT NULL,   -- plain-English lesson shown in the UI
            applied_count INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT    NOT NULL
        )
        """
    )
    # Full conversation transcript — one row per user/assistant turn, linked by
    # conversation_id and scoped to a session. Powers memory + the history sidebar.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            conversation_id TEXT    NOT NULL,
            role            TEXT    NOT NULL,  -- 'user' | 'assistant'
            content         TEXT    NOT NULL,
            sources         TEXT,              -- JSON list of {url,title} (assistant turns)
            created_at      TEXT    NOT NULL
        )
        """
    )

    # Reinforcement-learning reward log — one row per reward signal (see
    # REWARD_SCORES). Powers the Reward dashboard and the learned policy.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rewards (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER,            -- the interaction this reward is for (may be NULL)
            query_group    INTEGER,            -- groups rewards by query (for per-query totals)
            session_id     TEXT,
            signal         TEXT    NOT NULL,   -- 'ab_winner' | 'ab_loser' | 'thumbs_up' | … | 'reason_bonus'
            score          REAL    NOT NULL,   -- the numeric reward
            model          TEXT,               -- 'Haiku' | 'Sonnet' (which model produced the answer)
            style          TEXT,               -- 'concise' | 'detailed' | NULL (single/followup)
            query_type     TEXT,               -- 'simple' | 'medium' | 'complex'
            created_at     TEXT    NOT NULL
        )
        """
    )
    # Learned policy updates — each row is one re-evaluation that picked the
    # highest-reward model+style+query_type pattern and adjusted behavior to it.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_updates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            queries_at  INTEGER,               -- # of reward-bearing queries at update time
            model       TEXT,                  -- highest-reward model
            style       TEXT,                  -- highest-reward answer style
            query_type  TEXT,                  -- query type the pattern applies to
            avg_reward  REAL,                  -- the pattern's average reward
            multiplier  REAL,                  -- reward vs. the other patterns (e.g. 2.0 = 2x)
            text        TEXT    NOT NULL,       -- plain-English policy statement
            created_at  TEXT    NOT NULL
        )
        """
    )

    # Authenticated Google accounts — identity persists across devices for 2 weeks.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            google_id       TEXT PRIMARY KEY,
            name            TEXT,
            email           TEXT,
            profile_picture TEXT,
            created_at      TEXT    NOT NULL,
            last_login      TEXT
        )
        """
    )
    # Per-user personality profile, rebuilt every few queries from their own data.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            google_id       TEXT PRIMARY KEY,
            expertise_level TEXT,              -- beginner | intermediate | expert
            preferred_style TEXT,              -- concise | detailed
            top_topics      TEXT,              -- JSON list of subjects
            preferred_model TEXT,              -- Haiku | Sonnet (highest-reward)
            summary         TEXT,              -- the system-prompt sentence
            queries_at      INTEGER,           -- # of the user's queries at build time
            updated_at      TEXT    NOT NULL
        )
        """
    )

    # Migrations: add columns to pre-existing tables (idempotent).
    def _ensure(table: str, col: str, decl: str) -> None:
        existing = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    _ensure("interactions", "sources", "TEXT")
    _ensure("interactions", "routing", "TEXT")
    _ensure("interactions", "session_id", "TEXT")
    _ensure("comparisons", "session_id", "TEXT")
    _ensure("comparisons", "conversation_id", "TEXT")
    # Per-side routing — A and B can use different models (Haiku A / Sonnet B on
    # complex queries), so the reward system can attribute the winner to its model.
    _ensure("comparisons", "routing_a", "TEXT")
    _ensure("comparisons", "routing_b", "TEXT")
    _ensure("lessons", "session_id", "TEXT")
    # Policy is now learned per user (scoped by the originating session_id/google_id).
    _ensure("policy_updates", "session_id", "TEXT")
    conn.commit()


# The app's data tables — used by restore_from_sql to clear the DB before a
# full import (the backup is a sqlite `.dump`, whose CREATE TABLE statements
# would otherwise collide with init_db's tables). NOTE: `users` is deliberately
# excluded so accounts survive a reset/restore; user_profiles is derived data
# and regenerates, so it's included.
DATA_TABLES = ("interactions", "comparisons", "lessons", "messages",
               "rewards", "policy_updates", "user_profiles")


def restore_from_sql(conn: sqlite3.Connection, sql_text: str) -> dict[str, int]:
    """Restore the database from a sqlite `.dump` script (e.g. backup.sql).

    Drops the existing app tables, then replays the dump so the data matches the
    backup exactly. Returns a row count per table after the import. Idempotent:
    running it twice yields the same final state.
    """
    # Drop first so the dump's CREATE TABLE statements don't hit "table exists".
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in DATA_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()

    # executescript() commits any pending transaction, then runs the whole dump
    # (which wraps its INSERTs in its own BEGIN/COMMIT).
    conn.executescript(sql_text)
    conn.commit()

    # Re-apply column migrations in case the dump predates a newer column.
    init_db(conn)

    counts: dict[str, int] = {}
    for table in DATA_TABLES:
        try:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = 0
    return counts


def next_query_group(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(query_group), 0) FROM interactions").fetchone()
    return int(row[0]) + 1


def record_interaction(
    conn: sqlite3.Connection,
    *,
    query_group: int,
    original_query: str,
    effective_query: str,
    answer: str,
    feedback: str | None,
    comment: str | None,
    attempt: int,
    sources: list[dict] | None = None,
    routing: dict | None = None,
    session_id: str | None = None,
) -> int:
    """Insert a completed interaction (with feedback); return its row id."""
    cur = conn.execute(
        """
        INSERT INTO interactions
            (query_group, original_query, effective_query, answer,
             feedback, comment, attempt, created_at, sources, routing, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_group,
            original_query,
            effective_query,
            answer,
            feedback,
            comment,
            attempt,
            _dt.datetime.now().isoformat(timespec="seconds"),
            json.dumps(sources or []),
            json.dumps(routing) if routing else None,
            session_id,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_answer(
    conn: sqlite3.Connection,
    *,
    query_group: int,
    original_query: str,
    effective_query: str,
    answer: str,
    attempt: int,
    sources: list[dict] | None = None,
    routing: dict | None = None,
) -> int:
    """Insert an answer (with its web sources) with no feedback yet; return row id.

    Used by the web UI, where feedback arrives in a later request.
    """
    cur = conn.execute(
        """
        INSERT INTO interactions
            (query_group, original_query, effective_query, answer,
             feedback, comment, attempt, created_at, sources, routing)
        VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
        """,
        (
            query_group,
            original_query,
            effective_query,
            answer,
            attempt,
            _dt.datetime.now().isoformat(timespec="seconds"),
            json.dumps(sources or []),
            json.dumps(routing) if routing else None,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_feedback(
    conn: sqlite3.Connection,
    interaction_id: int,
    feedback: str | None,
    comment: str | None = None,
) -> None:
    """Attach feedback (and an optional comment) to an existing interaction."""
    conn.execute(
        "UPDATE interactions SET feedback = ?, comment = ? WHERE id = ?",
        (feedback, comment, interaction_id),
    )
    conn.commit()


def attempts_in_group(conn: sqlite3.Connection, query_group: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM interactions WHERE query_group = ?", (query_group,)
    ).fetchone()
    return int(row[0])


def get_answer(conn: sqlite3.Connection, interaction_id: int) -> str:
    row = conn.execute(
        "SELECT answer FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    return row[0] if row else ""


def get_sources(conn: sqlite3.Connection, interaction_id: int) -> list[dict]:
    """Return the stored web sources for an interaction (the original retrieval)."""
    row = conn.execute(
        "SELECT sources FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def recent_lessons(conn: sqlite3.Connection, limit: int = LESSON_CONTEXT_LIMIT) -> list[str]:
    """Return recent negative-feedback comments to learn from."""
    rows = conn.execute(
        """
        SELECT effective_query, comment
        FROM interactions
        WHERE feedback = 'down' AND comment IS NOT NULL AND TRIM(comment) <> ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [f'For a query like "{q}", a user complained: {c}' for q, c in rows]


# --------------------------------------------------------------------------- #
# A/B comparisons + the visible learning loop
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "best", "good", "top", "vs", "versus", "between", "difference", "compare",
    "should", "can", "do", "does", "i", "my", "me", "we", "you", "your", "with",
    "about", "explain", "tell", "give", "list", "some", "any", "this", "that",
    "it", "be", "use", "using", "vs.", "&",
}
# Plain-English descriptions used to phrase a learned lesson.
_STYLE_DESC = {
    "A": "concise, factual",
    "B": "detailed, analytical",
}
_OTHER_STYLE_DESC = {
    "A": "long, detailed ones",
    "B": "brief summaries",
}
_REASON_DESC = {
    "more accurate": "they were more accurate",
    "clearer explanation": "they explained things more clearly",
    "better sources": "they had better sources",
}


def extract_topic(query: str) -> str:
    """Pull a short topic keyword from a query (for matching lessons to queries).

    Strips question words / stopwords and returns the most salient remaining
    word, e.g. "best tools for python" -> "tools". Falls back to the whole
    (trimmed) query when nothing salient remains.
    """
    words = [w.strip(".,!?\"'()") for w in (query or "").lower().split()]
    salient = [w for w in words if w and w not in _STOPWORDS and len(w) > 2]
    return salient[0] if salient else (query or "").lower().strip()[:40]


def record_comparison(
    conn: sqlite3.Connection,
    *,
    query_group: int,
    original_query: str,
    effective_query: str,
    answers: dict,
    routing: dict,
    lessons_applied: list[int],
    session_id: str | None = None,
    conversation_id: str | None = None,
    routing_a: dict | None = None,
    routing_b: dict | None = None,
) -> int:
    """Persist a freshly generated A/B pair (no preference yet); return its id.

    ``routing`` is the base classification (used for the UI badge); ``routing_a`` /
    ``routing_b`` are the per-side routings (they differ on complex queries) and
    default to ``routing`` so the reward system can attribute the winner correctly.
    """
    cur = conn.execute(
        """
        INSERT INTO comparisons
            (query_group, session_id, conversation_id, original_query, effective_query,
             answer_a, answer_b, sources_a, sources_b, routing, routing_a, routing_b,
             preferred, reason, lessons_applied, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
        """,
        (
            query_group,
            session_id,
            conversation_id,
            original_query,
            effective_query,
            answers["a"]["answer"],
            answers["b"]["answer"],
            json.dumps(answers["a"]["sources"]),
            json.dumps(answers["b"]["sources"]),
            json.dumps(routing),
            json.dumps(routing_a or routing),
            json.dumps(routing_b or routing),
            json.dumps(lessons_applied or []),
            _dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_comparison(conn: sqlite3.Connection, comparison_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT id, query_group, session_id, conversation_id, original_query, effective_query,
               answer_a, answer_b, sources_a, sources_b, routing, routing_a, routing_b,
               preferred, reason
        FROM comparisons WHERE id = ?
        """,
        (comparison_id,),
    ).fetchone()
    if not row:
        return None
    keys = ["id", "query_group", "session_id", "conversation_id", "original_query",
            "effective_query", "answer_a", "answer_b", "sources_a", "sources_b",
            "routing", "routing_a", "routing_b", "preferred", "reason"]
    return dict(zip(keys, row))


# --------------------------------------------------------------------------- #
# Conversation transcript (memory + history sidebar)
# --------------------------------------------------------------------------- #
def save_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    """Append one turn to the conversation transcript; return its row id."""
    cur = conn.execute(
        """
        INSERT INTO messages (session_id, conversation_id, role, content, sources, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            conversation_id,
            role,
            content,
            json.dumps(sources) if sources else None,
            _dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def conversation_history(
    conn: sqlite3.Connection, conversation_id: str, limit: int = 10
) -> list[dict]:
    """Return the last ``limit`` turns as [{role, content}] for Claude's context.

    This is what makes follow-ups work ("tell me more about point 2"): the prior
    user/assistant turns are replayed to the model before the new question.
    """
    rows = conn.execute(
        """
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    return [{"role": r, "content": c} for (r, c) in reversed(rows)]


def load_conversation(conn: sqlite3.Connection, session_id: str, conversation_id: str) -> list[dict]:
    """Return the full transcript of a conversation (for rendering when reopened).

    Scoped by session_id so one browser can't read another's history.
    """
    rows = conn.execute(
        """
        SELECT role, content, sources, created_at FROM messages
        WHERE session_id = ? AND conversation_id = ?
        ORDER BY id ASC
        """,
        (session_id, conversation_id),
    ).fetchall()
    out = []
    for role, content, sources, created in rows:
        try:
            src = json.loads(sources) if sources else []
        except (json.JSONDecodeError, TypeError):
            src = []
        out.append({"role": role, "content": content, "sources": src, "created_at": created})
    return out


def list_conversations(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """List a session's conversations for the sidebar (newest activity first).

    Title = the first user message; sorted by the latest message in each chat.
    """
    rows = conn.execute(
        """
        SELECT conversation_id,
               MIN(CASE WHEN role = 'user' THEN id END) AS first_user_id,
               MAX(created_at) AS last_at,
               COUNT(*) AS n
        FROM messages
        WHERE session_id = ?
        GROUP BY conversation_id
        ORDER BY MAX(id) DESC
        """,
        (session_id,),
    ).fetchall()
    out = []
    for conv_id, first_user_id, last_at, n in rows:
        title = ""
        if first_user_id is not None:
            t = conn.execute("SELECT content FROM messages WHERE id = ?", (first_user_id,)).fetchone()
            title = t[0] if t else ""
        out.append({
            "conversation_id": conv_id,
            "title": title or "New conversation",
            "last_at": last_at,
            "message_count": int(n),
        })
    return out


def set_preference(conn: sqlite3.Connection, comparison_id: int, preferred: str) -> None:
    conn.execute(
        "UPDATE comparisons SET preferred = ? WHERE id = ?", (preferred, comparison_id)
    )
    conn.commit()


def set_reason(conn: sqlite3.Connection, comparison_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE comparisons SET reason = ? WHERE id = ?", (reason, comparison_id)
    )
    conn.commit()


def format_lesson(topic: str, winning_style: str, reason: str | None) -> str:
    """Phrase a plain-English lesson from a recorded preference."""
    win = _STYLE_DESC.get(winning_style, "detailed, analytical")
    other = _OTHER_STYLE_DESC.get(winning_style, "the alternative")
    text = f"For questions about {topic}, {win} answers are preferred over {other}"
    why = _REASON_DESC.get((reason or "").lower())
    if why:
        text += f" — {why}"
    return text + "."


def record_lesson(
    conn: sqlite3.Connection, *, topic: str, winning_style: str, reason: str | None,
    session_id: str | None = None,
) -> dict:
    """Create a lesson from a preference, or reinforce the user's matching one.

    Lessons are scoped per user (``session_id`` = the user's google_id): each user
    trains their own SearchIQ. A lesson for the same (session_id, topic,
    winning_style) is refreshed rather than duplicated. Returns the lesson.
    """
    if session_id is None:
        existing = conn.execute(
            "SELECT id FROM lessons WHERE topic = ? AND winning_style = ? AND session_id IS NULL",
            (topic, winning_style),
        ).fetchone()
    else:
        existing = conn.execute(
            "SELECT id FROM lessons WHERE topic = ? AND winning_style = ? AND session_id = ?",
            (topic, winning_style, session_id),
        ).fetchone()
    text = format_lesson(topic, winning_style, reason)
    if existing:
        lesson_id = int(existing[0])
        conn.execute(
            "UPDATE lessons SET reason = ?, text = ? WHERE id = ?",
            (reason, text, lesson_id),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO lessons (session_id, topic, winning_style, reason, text, applied_count, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (session_id, topic, winning_style, reason, text, _dt.datetime.now().isoformat(timespec="seconds")),
        )
        lesson_id = int(cur.lastrowid)
    conn.commit()
    return {"id": lesson_id, "topic": topic, "text": text, "winning_style": winning_style}


def match_lessons(conn: sqlite3.Connection, query: str, limit: int = 5,
                  session_id: str | None = None) -> list[dict]:
    """Return the user's lessons relevant to a query (topic appears in it).

    Scoped to ``session_id`` so a winning style the user taught becomes the
    preferred pattern for their SIMILAR future queries (``None`` = all lessons,
    used by the CLI). Returns dicts {id, text, topic}.
    """
    q = (query or "").lower()
    if session_id is None:
        rows = conn.execute("SELECT id, topic, text FROM lessons ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT id, topic, text FROM lessons WHERE session_id = ? ORDER BY id DESC",
            (session_id,),
        ).fetchall()
    matched = [
        {"id": int(i), "topic": t, "text": txt}
        for (i, t, txt) in rows
        if t and t in q
    ]
    return matched[:limit]


def bump_applied_count(conn: sqlite3.Connection, lesson_ids: list[int]) -> None:
    """Increment how many times each lesson has been applied to an answer."""
    if not lesson_ids:
        return
    conn.executemany(
        "UPDATE lessons SET applied_count = applied_count + 1 WHERE id = ?",
        [(int(i),) for i in lesson_ids],
    )
    conn.commit()


def all_lessons(conn: sqlite3.Connection, limit: int = 50,
                session_id: str | None = None) -> list[dict]:
    """Return the user's lessons feed for the dashboard, newest first (None = all)."""
    if session_id is None:
        rows = conn.execute(
            "SELECT id, text, topic, applied_count, created_at FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, text, topic, applied_count, created_at FROM lessons "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [
        {"id": int(i), "text": txt, "topic": t, "applied_count": int(ac), "created_at": ca}
        for (i, txt, t, ac, ca) in rows
    ]


def training_progress(conn: sqlite3.Connection, session_id: str | None = None) -> dict:
    """Training Progress totals for the user: lessons learned + times applied."""
    if session_id is None:
        learned = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        applied = conn.execute("SELECT COALESCE(SUM(applied_count), 0) FROM lessons").fetchone()[0]
    else:
        learned = conn.execute(
            "SELECT COUNT(*) FROM lessons WHERE session_id = ?", (session_id,)).fetchone()[0]
        applied = conn.execute(
            "SELECT COALESCE(SUM(applied_count), 0) FROM lessons WHERE session_id = ?",
            (session_id,)).fetchone()[0]
    return {"lessons_learned": int(learned), "lessons_applied": int(applied)}


# --------------------------------------------------------------------------- #
# Reinforcement learning — reward log, reward stats, and the learned policy
# --------------------------------------------------------------------------- #
def routing_facets(routing: dict | None) -> tuple[str | None, str | None]:
    """Pull (model_label, complexity) from a routing dict for reward attribution."""
    if not routing:
        return None, None
    model = routing.get("model_label")
    if not model:
        m = routing.get("model", "") or ""
        model = "Sonnet" if m.startswith("claude-sonnet") else "Haiku"
    return model, routing.get("complexity")


def record_reward(
    conn: sqlite3.Connection,
    *,
    signal: str,
    score: float,
    interaction_id: int | None = None,
    query_group: int | None = None,
    session_id: str | None = None,
    model: str | None = None,
    style: str | None = None,
    query_type: str | None = None,
) -> int:
    """Log one reward signal; return its row id. See REWARD_SCORES for the scale."""
    cur = conn.execute(
        """
        INSERT INTO rewards
            (interaction_id, query_group, session_id, signal, score,
             model, style, query_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            interaction_id, query_group, session_id, signal, float(score),
            model, style, query_type, _dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_routing(conn: sqlite3.Connection, interaction_id: int) -> dict | None:
    """Return the stored routing dict for an interaction (or None)."""
    row = conn.execute(
        "SELECT routing FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def last_interaction(conn: sqlite3.Connection, session_id: str | None) -> dict | None:
    """Most recent interaction for a session — the answer a follow-up rewards."""
    if not session_id:
        return None
    row = conn.execute(
        "SELECT id, query_group, routing FROM interactions WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    rid, group, routing = row
    try:
        rt = json.loads(routing) if routing else None
    except (json.JSONDecodeError, TypeError):
        rt = None
    return {"id": int(rid), "query_group": int(group) if group is not None else None, "routing": rt}


def _reward_velocity(group_totals: list[float]) -> float:
    """Least-squares slope of per-query reward, scaled to 'reward gained per 10 queries'."""
    n = len(group_totals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(group_totals) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, group_totals)) / denom
    return round(slope * 10, 3)


def reward_stats(conn: sqlite3.Connection, session_id: str | None = None) -> dict:
    """Aggregate the reward log for the Reward dashboard panel (None = all users)."""
    if session_id is None:
        rows = conn.execute(
            "SELECT query_group, score, model, style, query_type FROM rewards ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT query_group, score, model, style, query_type FROM rewards "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    total = len(rows)
    pos = [r for r in rows if r[1] > 0]
    neg = [r for r in rows if r[1] < 0]
    avg = (sum(r[1] for r in rows) / total) if total else 0.0

    def _avg_by(idx: int) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in rows:
            key = r[idx]
            if key is None:
                continue
            buckets.setdefault(key, []).append(r[1])
        return {k: round(sum(v) / len(v), 3) for k, v in buckets.items()}

    by_model = _avg_by(2)
    by_style = _avg_by(3)
    by_query_type = _avg_by(4)

    # Per-query-group total reward, in chronological (id) order, with running avg.
    group_order: list[int] = []
    group_totals: dict[int, float] = {}
    for group, score, *_ in rows:
        g = group if group is not None else -1
        if g not in group_totals:
            group_totals[g] = 0.0
            group_order.append(g)
        group_totals[g] += score

    timeline = []
    running_sum = 0.0
    for i, g in enumerate(group_order, start=1):
        running_sum += group_totals[g]
        timeline.append({"group": g, "score": round(group_totals[g], 3), "avg": round(running_sum / i, 3)})

    return {
        "total_events": total,
        "avg_score": round(avg, 3),
        "positive_count": len(pos),
        "negative_count": len(neg),
        "positive_sum": round(sum(r[1] for r in pos), 3),
        "negative_sum": round(sum(r[1] for r in neg), 3),
        "by_model": {"Haiku": by_model.get("Haiku", 0.0), "Sonnet": by_model.get("Sonnet", 0.0)},
        "by_query_type": {k: by_query_type.get(k, 0.0) for k in ("simple", "medium", "complex")},
        "by_style": {k: by_style.get(k, 0.0) for k in ("concise", "detailed")},
        "timeline": timeline,
        "learning_velocity": _reward_velocity([group_totals[g] for g in group_order]),
    }


def compute_best_pattern(conn: sqlite3.Connection, session_id: str | None = None) -> dict | None:
    """Find the user's highest-average-reward model+style+query_type combo (None = all)."""
    if session_id is None:
        rows = conn.execute(
            """
            SELECT model, style, query_type, AVG(score) AS avg_score, COUNT(*) AS n
            FROM rewards
            WHERE model IS NOT NULL AND style IS NOT NULL
            GROUP BY model, style, query_type
            ORDER BY avg_score DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT model, style, query_type, AVG(score) AS avg_score, COUNT(*) AS n
            FROM rewards
            WHERE model IS NOT NULL AND style IS NOT NULL AND session_id = ?
            GROUP BY model, style, query_type
            ORDER BY avg_score DESC
            """,
            (session_id,),
        ).fetchall()
    if not rows:
        return None
    model, style, query_type, best_avg, n = rows[0]
    # Multiplier = best avg vs. the mean of the other patterns (only when both positive).
    multiplier = None
    others = [r[3] for r in rows[1:]]
    if best_avg > 0 and others:
        other_mean = sum(others) / len(others)
        if other_mean > 0:
            multiplier = round(best_avg / other_mean, 1)
    return {
        "model": model,
        "style": style,
        "query_type": query_type or "general",
        "avg_reward": round(best_avg, 3),
        "multiplier": multiplier,
        "samples": int(n),
    }


def _policy_text(pattern: dict) -> str:
    """Plain-English statement of a learned policy, e.g. the lessons-feed line."""
    style_word = "Detailed" if pattern["style"] == "detailed" else "Concise"
    qt = pattern["query_type"]
    scope = "queries" if qt == "general" else f"{qt} queries"
    if pattern.get("multiplier") and pattern["multiplier"] >= 1.5:
        return f"{style_word} {pattern['model']} answers get {pattern['multiplier']:g}x reward for {scope}"
    return (f"{style_word} {pattern['model']} answers earn the highest reward "
            f"(avg {pattern['avg_reward']:+.2f}) for {scope}")


def maybe_update_policy(conn: sqlite3.Connection, session_id: str | None = None) -> dict | None:
    """Re-evaluate the user's policy after every POLICY_EVERY reward-bearing queries.

    Scoped to ``session_id`` (None = the CLI/global bucket). Picks the user's
    highest-reward pattern, records a per-user policy_updates row, and upserts a
    visible line into their lessons feed. Returns the pattern (with its plain-English
    ``text``) when an update happens, else None. Idempotent within a checkpoint.
    """
    if session_id is None:
        n = int(conn.execute(
            "SELECT COUNT(DISTINCT query_group) FROM rewards WHERE query_group IS NOT NULL"
        ).fetchone()[0])
        last = conn.execute(
            "SELECT queries_at FROM policy_updates WHERE session_id IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
    else:
        n = int(conn.execute(
            "SELECT COUNT(DISTINCT query_group) FROM rewards WHERE query_group IS NOT NULL AND session_id = ?",
            (session_id,),
        ).fetchone()[0])
        last = conn.execute(
            "SELECT queries_at FROM policy_updates WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if n == 0 or n % POLICY_EVERY != 0:
        return None
    if last and last[0] is not None and int(last[0]) >= n:
        return None  # already updated at this checkpoint
    pattern = compute_best_pattern(conn, session_id)
    if not pattern:
        return None

    text = _policy_text(pattern)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO policy_updates
            (queries_at, model, style, query_type, avg_reward, multiplier, text, created_at, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (n, pattern["model"], pattern["style"], pattern["query_type"],
         pattern["avg_reward"], pattern.get("multiplier"), text, now, session_id),
    )
    # Log the policy change into the user's lessons feed (upsert by topic). topic
    # 'policy:*' never matches a query (match_lessons does substring matching), so
    # it stays display-only.
    topic = f"policy:{pattern['query_type']}"
    feed_text = f"Policy: {text}"
    winning_style = "B" if pattern["style"] == "detailed" else "A"
    if session_id is None:
        existing = conn.execute(
            "SELECT id FROM lessons WHERE topic = ? AND session_id IS NULL", (topic,)).fetchone()
    else:
        existing = conn.execute(
            "SELECT id FROM lessons WHERE topic = ? AND session_id = ?", (topic, session_id)).fetchone()
    if existing:
        conn.execute(
            "UPDATE lessons SET text = ?, winning_style = ? WHERE id = ?",
            (feed_text, winning_style, int(existing[0])),
        )
    else:
        conn.execute(
            """
            INSERT INTO lessons (session_id, topic, winning_style, reason, text, applied_count, created_at)
            VALUES (?, ?, ?, NULL, ?, 0, ?)
            """,
            (session_id, topic, winning_style, feed_text, now),
        )
    conn.commit()
    pattern["text"] = text
    return pattern


def current_policy(conn: sqlite3.Connection, session_id: str | None = None) -> dict | None:
    """The user's most recent learned policy (None = the CLI/global bucket)."""
    if session_id is None:
        row = conn.execute(
            """
            SELECT model, style, query_type, avg_reward, multiplier, text, created_at
            FROM policy_updates WHERE session_id IS NULL ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT model, style, query_type, avg_reward, multiplier, text, created_at
            FROM policy_updates WHERE session_id = ? ORDER BY id DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if not row:
        return None
    keys = ["model", "style", "query_type", "avg_reward", "multiplier", "text", "created_at"]
    return dict(zip(keys, row))


def policy_directive(conn: sqlite3.Connection, session_id: str | None = None) -> str:
    """A system-prompt note steering answers toward the user's policy winning style.

    Prepended to the lessons list so it flows through build_system_prompt into BOTH
    A/B drafts. Returns "" when the user has no policy yet.
    """
    p = current_policy(conn, session_id)
    if not p:
        return ""
    style_word = ("detailed, analytical and well-structured" if p["style"] == "detailed"
                  else "concise, factual and skimmable")
    qt = p.get("query_type")
    scope = "all queries" if qt in (None, "general") else f"{qt} queries"
    return (f"LEARNED POLICY (from your reward signals): for {scope}, you most reward "
            f"{style_word} answers — lean toward that where it fits the question.")


def apply_policy(conn: sqlite3.Connection, routing: dict, session_id: str | None = None) -> dict:
    """Bias model routing toward the user's policy highest-reward model.

    When a policy exists, applies to this query's complexity (or is general), and
    prefers a different model than the rule-based router chose, swap the model in
    while keeping the complexity tier. Sets routing["policy_biased"]=True so the UI
    can flag it. No-op when there's no policy or it already agrees with the router.
    """
    p = current_policy(conn, session_id)
    if not p or not routing:
        return routing
    qt = p.get("query_type")
    if qt not in (None, "general", routing.get("complexity")):
        return routing
    target_label = p.get("model")
    if target_label not in ("Haiku", "Sonnet") or routing.get("model_label") == target_label:
        return routing
    complexity = routing.get("complexity", "medium")
    tier = TIERS.get(complexity, TIERS["medium"])
    biased = dict(routing)
    biased["model"] = SONNET_MODEL if target_label == "Sonnet" else HAIKU_MODEL
    biased["model_label"] = target_label
    biased["badge"] = f"{tier['icon']} {complexity.capitalize()} · {target_label} · {tier['mode']}"
    biased["policy_biased"] = True
    return biased


# --------------------------------------------------------------------------- #
# Users + personality profiles
# --------------------------------------------------------------------------- #
def upsert_user(conn: sqlite3.Connection, *, google_id: str, name: str | None,
                email: str | None, picture: str | None) -> None:
    """Create or refresh a Google account row; bumps last_login on every sign-in."""
    now = _dt.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO users (google_id, name, email, profile_picture, created_at, last_login)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(google_id) DO UPDATE SET
            name = excluded.name, email = excluded.email,
            profile_picture = excluded.profile_picture, last_login = excluded.last_login
        """,
        (google_id, name, email, picture, now, now),
    )
    conn.commit()


def get_user(conn: sqlite3.Connection, google_id: str) -> dict | None:
    row = conn.execute(
        "SELECT google_id, name, email, profile_picture, created_at, last_login "
        "FROM users WHERE google_id = ?",
        (google_id,),
    ).fetchone()
    if not row:
        return None
    return dict(zip(["google_id", "name", "email", "profile_picture", "created_at", "last_login"], row))


def _profile_summary(expertise: str, style: str, topics: list[str], model: str | None) -> str:
    """The one-line personality summary injected into Claude's system prompt."""
    article = "an" if expertise in ("expert", "intermediate") else "a"
    style_phrase = "detailed, technical answers" if style == "detailed" else "concise, to-the-point answers"
    out = f"This user is {article} {expertise}-level user who prefers {style_phrase}."
    if topics:
        out += f" Top topics: {', '.join(topics)}."
    if model:
        out += f" They get the most value from {model} responses."
    out += " Tailor depth, vocabulary, and length accordingly."
    return out


def build_user_profile(conn: sqlite3.Connection, google_id: str) -> dict:
    """Derive a personality profile from the user's own queries + reward signals.

    Heuristic (no extra LLM call): expertise from query complexity + vocabulary,
    preferred style/model from reward-by-style/model, top topics from query topics.
    """
    rows = conn.execute(
        "SELECT original_query, routing FROM interactions WHERE session_id = ?",
        (google_id,),
    ).fetchall()
    comp = {"simple": 0, "medium": 0, "complex": 0}
    words_total, nq = 0, 0
    topics: dict[str, int] = {}
    for original_query, routing in rows:
        nq += 1
        words_total += len((original_query or "").split())
        complexity = None
        if routing:
            try:
                complexity = (json.loads(routing) or {}).get("complexity")
            except (json.JSONDecodeError, TypeError):
                complexity = None
        if complexity is None:
            complexity = classify_query(original_query or "").get("complexity")
        if complexity in comp:
            comp[complexity] += 1
        topic = extract_topic(original_query or "")
        if topic:
            topics[topic] = topics.get(topic, 0) + 1

    total = max(1, nq)
    complex_frac = comp["complex"] / total
    simple_frac = comp["simple"] / total
    avg_words = (words_total / nq) if nq else 0.0
    if complex_frac >= 0.5 or (complex_frac >= 0.3 and avg_words >= 12):
        expertise = "expert"
    elif simple_frac >= 0.6 and avg_words <= 6:
        expertise = "beginner"
    else:
        expertise = "intermediate"

    stats = reward_stats(conn, google_id)
    by_style, by_model = stats["by_style"], stats["by_model"]
    preferred_style = "detailed" if by_style.get("detailed", 0.0) >= by_style.get("concise", 0.0) else "concise"
    preferred_model = "Sonnet" if by_model.get("Sonnet", 0.0) >= by_model.get("Haiku", 0.0) else "Haiku"
    if stats["total_events"] == 0:
        preferred_model = None  # no reward data yet — don't claim a model preference
    top_topics = [t for t, _ in sorted(topics.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

    return {
        "expertise_level": expertise,
        "preferred_style": preferred_style,
        "top_topics": top_topics,
        "preferred_model": preferred_model,
        "summary": _profile_summary(expertise, preferred_style, top_topics, preferred_model),
        "queries": nq,
    }


def maybe_update_profile(conn: sqlite3.Connection, google_id: str | None) -> dict | None:
    """Rebuild the user's profile after every PROFILE_EVERY queries (idempotent)."""
    if not google_id:
        return None
    n = int(conn.execute(
        "SELECT COUNT(DISTINCT query_group) FROM interactions WHERE session_id = ?",
        (google_id,),
    ).fetchone()[0])
    if n == 0 or n % PROFILE_EVERY != 0:
        return None
    last = conn.execute(
        "SELECT queries_at FROM user_profiles WHERE google_id = ?", (google_id,)
    ).fetchone()
    if last and last[0] is not None and int(last[0]) >= n:
        return None
    prof = build_user_profile(conn, google_id)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO user_profiles
            (google_id, expertise_level, preferred_style, top_topics, preferred_model,
             summary, queries_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(google_id) DO UPDATE SET
            expertise_level = excluded.expertise_level, preferred_style = excluded.preferred_style,
            top_topics = excluded.top_topics, preferred_model = excluded.preferred_model,
            summary = excluded.summary, queries_at = excluded.queries_at, updated_at = excluded.updated_at
        """,
        (google_id, prof["expertise_level"], prof["preferred_style"], json.dumps(prof["top_topics"]),
         prof["preferred_model"], prof["summary"], n, now),
    )
    conn.commit()
    return prof


def get_user_profile(conn: sqlite3.Connection, google_id: str | None) -> dict | None:
    """Return the stored personality profile (top_topics parsed), or None."""
    if not google_id:
        return None
    row = conn.execute(
        """
        SELECT expertise_level, preferred_style, top_topics, preferred_model, summary, updated_at
        FROM user_profiles WHERE google_id = ?
        """,
        (google_id,),
    ).fetchone()
    if not row:
        return None
    d = dict(zip(["expertise_level", "preferred_style", "top_topics", "preferred_model",
                  "summary", "updated_at"], row))
    try:
        d["top_topics"] = json.loads(d["top_topics"]) if d["top_topics"] else []
    except (json.JSONDecodeError, TypeError):
        d["top_topics"] = []
    return d


def profile_directive(profile: dict | None) -> str:
    """System-prompt note that personalizes answers to the user's profile ("" if none)."""
    if not profile or not profile.get("summary"):
        return ""
    return "USER PROFILE — personalize to it: " + profile["summary"]


# --------------------------------------------------------------------------- #
# Claude calls
# --------------------------------------------------------------------------- #
# A/B answer styles — both web-grounded, but two distinct synthesis approaches
# the user compares side by side.
STYLE_A = (
    "STYLE — FACTUAL & CONCISE: Lead with the direct answer in the first sentence. "
    "Keep it tight and skimmable: short paragraphs or a few bullet points, only the "
    "key facts, no filler. Aim for brevity over breadth."
)
STYLE_B = (
    "STYLE — DETAILED & ANALYTICAL: Give a thorough, well-structured answer. Add "
    "context, compare options, note trade-offs and caveats, and explain the 'why' "
    "behind the facts. Use headings or bullet lists where they aid clarity."
)


def build_system_prompt(lessons: list[str], style: str | None = None) -> str:
    base = (
        "You are a precise search-answer assistant. Given a user's search query, "
        "produce a direct, well-structured, factual answer. "
        "If the query is ambiguous, answer the most likely intent and note the assumption. "
        "Respond directly without preamble like 'Here is' or 'Sure'.\n\n"
        "GROUNDING REQUIREMENT: You have a web_search tool and you MUST use it on "
        "every query. Always run at least one web search BEFORE writing your answer — "
        "never answer from training memory alone, even when you are confident you "
        "already know the answer. If a first search is thin, search again with a "
        "better query before answering.\n\n"
        "Read the search results carefully. Your answer MUST be based on what the "
        "sources say, not your prior knowledge. If a source says X happened, report "
        "X. Quote or closely paraphrase the relevant passages from the sources as you "
        "build your answer, use your own knowledge only to add light context around "
        "the sourced facts, and cite the sources you used."
    )
    if style:
        base += "\n\n" + style
    if lessons:
        joined = "\n".join(f"- {lesson}" for lesson in lessons)
        base += (
            "\n\nLessons learned from past user preferences — these reflect what users "
            f"liked about previous answers, so apply them here:\n{joined}"
        )
    return base


def extract_sources(message) -> list[dict]:
    """Pull web sources from a response: citations first, then searched results.

    Returns a deduped list of {"url", "title"} dicts (cited sources prioritized).
    """
    sources: list[dict] = []
    seen: set[str] = set()

    def add(url, title):
        if url and url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": title or url})

    # Pass 1 — sources Claude actually cited in its answer.
    for block in message.content:
        if getattr(block, "type", None) == "text":
            for cit in getattr(block, "citations", None) or []:
                add(getattr(cit, "url", None), getattr(cit, "title", None))

    # Pass 2 — results returned by the search tool (fills in if citations are sparse).
    for block in message.content:
        if getattr(block, "type", None) == "web_search_tool_result":
            content = getattr(block, "content", None) or []
            if isinstance(content, list):
                for r in content:
                    add(getattr(r, "url", None), getattr(r, "title", None))

    return sources[:MAX_SOURCES]


def _history_messages(history: list[dict] | None, query: str) -> list[dict]:
    """Build the messages array: prior conversation turns, then the new question.

    ``history`` is [{role, content}] from conversation_history(). Only user/
    assistant roles with non-empty content are kept, and a leading assistant turn
    is dropped so the array always starts with a user message (API requirement).
    """
    msgs: list[dict] = []
    for h in history or []:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    msgs.append({"role": "user", "content": query})
    return msgs


def search_and_answer(
    client: anthropic.Anthropic,
    query: str,
    lessons: list[str],
    routing: dict,
    style: str | None = None,
    on_text=None,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Answer a query with web-search RAG, aggregated by the routed model.

    Web search ALWAYS runs (same retrieval for every query) — ``routing`` (from
    classify_query) only selects which model aggregates the results into the
    answer: Haiku for simple/medium, Sonnet for complex. ``style`` (STYLE_A /
    STYLE_B) tunes the synthesis approach for A/B comparison. ``history`` (prior
    conversation turns) is prepended so follow-up questions resolve in context.
    Returns (answer_text, sources). If ``on_text`` is given, text chunks are
    streamed to it as they arrive (used by the terminal for live output).
    """
    kwargs = dict(
        model=routing["model"],
        max_tokens=8000,
        system=build_system_prompt(lessons, style=style),
        tools=[WEB_SEARCH_TOOL],
        messages=_history_messages(history, query),
        **_reasoning_kwargs(routing["model"]),
    )

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            if on_text:
                on_text(text)
        final = stream.get_final_message()

    answer = "".join(b.text for b in final.content if b.type == "text").strip()
    return answer, extract_sources(final)


def iter_ab_answers(
    client: anthropic.Anthropic,
    query: str,
    lessons: list[str],
    routing_by_side: dict[str, dict],
    history: list[dict] | None = None,
    timeout: float = ANSWER_TIMEOUT_S,
):
    """Generate Answer A and B in parallel, yielding each as soon as it's ready.

    ``routing_by_side`` maps "A"/"B" → routing dict (see ab_routings; on complex
    queries A is Haiku and B is Sonnet, so the fast one streams out first). Yields
    ``(side, {"answer", "sources"})`` on success or ``(side, {"error": msg})`` when a
    draft fails or exceeds ``timeout`` seconds. Each draft is bounded independently,
    so a slow Sonnet B never blocks delivering a finished Haiku A.

    A is factual & concise (STYLE_A); B is detailed & analytical (STYLE_B) — the
    two distinct styles (plus the model split) keep the answers genuinely different.
    """
    styles = {"A": STYLE_A, "B": STYLE_B}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {
            pool.submit(search_and_answer, client, query, lessons,
                        routing_by_side[side], styles[side], None, history): side
            for side in ("A", "B")
        }
        pending = set(futs)
        try:
            for fut in as_completed(futs, timeout=timeout):
                side = futs[fut]
                pending.discard(fut)
                try:
                    ans, src = fut.result()
                    yield side, {"answer": ans, "sources": src}
                except Exception as exc:  # noqa: BLE001 — keep the other side alive
                    yield side, {"error": str(exc) or exc.__class__.__name__}
        except FuturesTimeoutError:
            pass  # remaining sides blew the per-answer budget; reported below
        for fut in pending:
            fut.cancel()
            yield futs[fut], {"error": f"timed out after {int(timeout)}s"}


def generate_ab_answers(
    client: anthropic.Anthropic,
    query: str,
    lessons: list[str],
    routing: dict,
    history: list[dict] | None = None,
) -> dict:
    """Non-streaming A/B generation: collect both drafts, then return them together.

    Thin wrapper over iter_ab_answers for callers that want a single dict rather
    than a stream. Both sides use the same ``routing``. Resilient to a flaky API:
    if only one draft succeeds it's returned with ``single=True``; raises only if
    BOTH fail. Returns {"a": {...}, "b": {...} | None, "single": bool}.
    """
    results: dict[str, dict] = {}
    for side, res in iter_ab_answers(client, query, lessons,
                                     {"A": routing, "B": routing}, history):
        if "answer" in res:
            results[side] = res
    if "A" in results and "B" in results:
        return {"a": results["A"], "b": results["B"], "single": False}
    if results:  # exactly one survived — serve it as a single answer
        only = results.get("A") or results.get("B")
        return {"a": only, "b": None, "single": True}
    raise RuntimeError("Both A/B answers failed to generate.")


def suggest_followups(
    client: anthropic.Anthropic, query: str, answer: str, n: int = 3
) -> list[str]:
    """Suggest ``n`` short follow-up questions based on an answer (cheap Haiku call).

    Returns a list of question strings, or [] on any failure (non-critical UI).
    """
    prompt = (
        f"A user asked: {query}\n\n"
        f"They received this answer:\n{answer[:1500]}\n\n"
        f"Suggest {n} natural follow-up questions the user might ask next. "
        "Each must be short (under 8 words), specific, and standalone. "
        f"Reply with ONLY the {n} questions, one per line, no numbering or extra text."
    )
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError:
        return []
    text = next((b.text for b in resp.content if b.type == "text"), "")
    lines = [
        ln.strip().lstrip("-*0123456789. ").strip()
        for ln in text.splitlines()
        if ln.strip()
    ]
    return [ln for ln in lines if ln][:n]


def regenerate_answer(
    client: anthropic.Anthropic,
    *,
    original_query: str,
    improved_query: str,
    previous_answer: str,
    sources: list[dict],
    comment: str | None,
    lessons: list[str],
    on_text=None,
) -> str:
    """Re-answer WITHOUT searching again — reuse the already-retrieved results.

    Used for retries after a thumbs-down so they stay fast: no second web search,
    only the answer is regenerated with improved framing, grounded in the same
    findings (the prior answer + the original source list). Returns the new answer
    text; the caller reuses the original ``sources`` for display.
    """
    complaint = comment.strip() if comment and comment.strip() else "(no specific comment given)"
    source_lines = "\n".join(f"- {s.get('title') or s.get('url')} ({s.get('url')})" for s in sources)

    system = (
        "You are a precise search-answer assistant improving a previous answer the "
        "user found unhelpful. You are NOT searching the web this time — reuse the "
        "information already gathered from the earlier search (provided below). Keep "
        "every factual claim strictly grounded in that gathered information; do not "
        "introduce facts from your own prior knowledge and do not invent details. "
        "Improve the framing, clarity, focus, and completeness to address the user's "
        "complaint. Respond directly without preamble."
    )
    if lessons:
        system += "\n\nLessons learned from past user feedback — apply these:\n" + "\n".join(
            f"- {lesson}" for lesson in lessons
        )

    user = (
        f"Original question: {original_query}\n"
        f"Improved framing to address: {improved_query}\n"
        f"User's complaint about the previous answer: {complaint}\n\n"
        "Information already gathered from the earlier web search — base your answer "
        "ONLY on this:\n\n"
        f"Previous answer (grounded in the sources):\n{previous_answer}\n\n"
        f"Sources that were retrieved:\n{source_lines or '(none)'}\n\n"
        "Write an improved answer that fixes the user's complaint while staying "
        "faithful to the gathered information above. Do not search again."
    )

    # Retries improve an answer the user rejected, so use the capable model.
    with client.messages.stream(
        model=SONNET_MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        **_reasoning_kwargs(SONNET_MODEL),
    ) as stream:
        for text in stream.text_stream:
            if on_text:
                on_text(text)
        final = stream.get_final_message()

    return "".join(b.text for b in final.content if b.type == "text").strip()


def print_sources(sources: list[dict]) -> None:
    """Terminal: print a numbered Sources list under an answer."""
    if not sources:
        return
    print("Sources:")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s.get('title') or s.get('url')}")
        print(f"      {s.get('url')}")
    print()


def rephrase_query(
    client: anthropic.Anthropic,
    *,
    original_query: str,
    failed_answer: str,
    comment: str | None,
) -> str:
    """Ask Claude to improve a query that produced an unhelpful answer."""
    complaint = comment.strip() if comment and comment.strip() else "(no specific comment given)"
    prompt = (
        "A search query produced an answer the user judged unhelpful. "
        "Rewrite the query so a fresh attempt is more likely to succeed: make the "
        "intent explicit, add useful constraints, and remove ambiguity. "
        "Reply with ONLY the rewritten query — no quotes, no explanation.\n\n"
        f"Original query: {original_query}\n"
        f"User's complaint: {complaint}\n"
        f"Previous answer (which fell short):\n{failed_answer[:1500]}"
    )
    # A small rewriting task — the cheap model handles it well.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        **_reasoning_kwargs(MODEL),
    )
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    return text or original_query


# --------------------------------------------------------------------------- #
# Feedback collection
# --------------------------------------------------------------------------- #
def collect_feedback() -> tuple[str | None, str | None]:
    """Prompt the user for a thumbs up/down and an optional comment."""
    while True:
        raw = input("Did this answer help? [u]p / [d]own / [s]kip: ").strip().lower()
        if raw in ("u", "up", "👍"):
            feedback: str | None = "up"
            break
        if raw in ("d", "down", "👎"):
            feedback = "down"
            break
        if raw in ("s", "skip", ""):
            return None, None
        print("  Please enter 'u', 'd', or 's'.")
    comment = input("Optional comment (press Enter to skip): ").strip()
    return feedback, (comment or None)


# --------------------------------------------------------------------------- #
# Dashboard / metrics
# --------------------------------------------------------------------------- #
def compute_metrics(conn: sqlite3.Connection, session_id: str | None = None) -> dict:
    """Compute query-group-level dashboard metrics for one user (None = all users).

    The North Star — total queries, completion rate, model mix, cost saved — is
    scoped to ``session_id`` so each signed-in user sees their own stats.
    """
    if session_id is None:
        rows = conn.execute(
            """
            SELECT query_group, original_query, attempt, feedback, routing
            FROM interactions
            ORDER BY query_group, attempt
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT query_group, original_query, attempt, feedback, routing
            FROM interactions
            WHERE session_id = ?
            ORDER BY query_group, attempt
            """,
            (session_id,),
        ).fetchall()

    groups: dict[int, dict] = {}
    # Per-model-call tallies (every interaction is one aggregation call).
    models_used = {"Haiku": 0, "Sonnet": 0}
    cost_actual = 0.0       # estimated $ actually spent on aggregation calls
    cost_all_sonnet = 0.0   # estimated $ had we always used Sonnet
    for group, original, attempt, feedback, routing in rows:
        g = groups.setdefault(
            group,
            {"original": original, "first_feedback": None, "best_feedback": None},
        )
        if attempt == 1:
            g["first_feedback"] = feedback
        if feedback == "up":
            g["best_feedback"] = "up"
        elif feedback == "down" and g["best_feedback"] is None:
            g["best_feedback"] = "down"

        # Model-mix + cost accounting across every aggregation call.
        if routing:
            try:
                rt = json.loads(routing)
            except (json.JSONDecodeError, TypeError):
                rt = None
            if rt:
                model = rt.get("model") or SONNET_MODEL
                label = "Sonnet" if model.startswith("claude-sonnet") else "Haiku"
                models_used[label] = models_used.get(label, 0) + 1
                cost_actual += estimate_query_cost(model)
                cost_all_sonnet += estimate_query_cost(SONNET_MODEL)

    total_queries = len(groups)
    # Estimated $ saved by routing to Haiku vs. always using Sonnet.
    cost_saved_usd = round(cost_all_sonnet - cost_actual, 4)
    # Only count queries that received any feedback toward the completion rate.
    rated = [g for g in groups.values() if g["best_feedback"] is not None]
    completed = [g for g in rated if g["best_feedback"] == "up"]
    completion_rate = (len(completed) / len(rated) * 100) if rated else 0.0

    # "Most improved": started with a thumbs-down on attempt 1, recovered to up.
    most_improved = [
        g["original"]
        for g in groups.values()
        if g["first_feedback"] == "down" and g["best_feedback"] == "up"
    ]

    return {
        "total_queries": total_queries,
        "rated_queries": len(rated),
        "completion_rate": completion_rate,
        "most_improved": most_improved,
        "models_used": models_used,
        "cost_saved_usd": cost_saved_usd,
    }


def show_dashboard(conn: sqlite3.Connection) -> None:
    m = compute_metrics(conn)
    bar_width = 30
    filled = int(round(m["completion_rate"] / 100 * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)

    print("\n" + "=" * 56)
    print("  📊  SEARCH AGENT DASHBOARD")
    print("=" * 56)
    print(f"  Total queries handled : {m['total_queries']}")
    print(f"  Queries with feedback : {m['rated_queries']}")
    mu = m["models_used"]
    print(f"  🤖 Models used        : Haiku {mu.get('Haiku', 0)} · Sonnet {mu.get('Sonnet', 0)}")
    print(f"  💰 Est. cost saved    : ${m['cost_saved_usd']:.4f}  (vs always Sonnet)")
    print("\n  ★ NORTH STAR — Task Completion Rate")
    print(f"    [{bar}] {m['completion_rate']:.1f}%")
    print("    (% of rated queries that earned a thumbs up)")
    print("\n  Most improved queries (👎 → 👍 after retry):")
    if m["most_improved"]:
        for q in m["most_improved"][-5:]:
            print(f"    • {textwrap.shorten(q, width=46)}")
    else:
        print("    (none yet)")
    print("=" * 56 + "\n")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def handle_query(
    client: anthropic.Anthropic, conn: sqlite3.Connection, original_query: str
) -> None:
    """Run one query through search → feedback → (reuse-results retry) cycles."""
    group = next_query_group(conn)
    _stream = lambda t: print(t, end="", flush=True)  # noqa: E731

    # Pre-process: size the query and pick the model + tools before calling Claude.
    routing = classify_query(original_query)
    print(f"\n🧭 Routing: {routing['badge']}")
    print("\n🔎 Searching the web…\n\nAnswer:\n")
    answer, sources = search_and_answer(
        client, original_query, recent_lessons(conn), routing, on_text=_stream
    )
    print("\n")
    print_sources(sources)
    effective_query = original_query
    # Retries improve a rejected answer with the capable model, reusing sources.
    retry_routing = build_routing("complex")

    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
        feedback, comment = collect_feedback()
        record_interaction(
            conn,
            query_group=group,
            original_query=original_query,
            effective_query=effective_query,
            answer=answer,
            feedback=feedback,
            comment=comment,
            attempt=attempt,
            sources=sources,
            routing=routing if attempt == 1 else retry_routing,
        )

        if feedback != "down":
            # Positive or skipped — nothing to improve on.
            return

        if attempt > MAX_RETRIES:
            print("  Reached the retry limit for this query. Logged for learning.\n")
            return

        # Retry is fast: reuse the original search results, only re-answer.
        print("  Got negative feedback — improving the answer (reusing earlier sources, no new search)…\n")
        effective_query = rephrase_query(
            client,
            original_query=original_query,
            failed_answer=answer,
            comment=comment,
        )
        print(f'  Improved framing → "{effective_query}"')
        print(f"  🧭 Routing: {retry_routing['badge']}\n\nAnswer:\n")
        answer = regenerate_answer(
            client,
            original_query=original_query,
            improved_query=effective_query,
            previous_answer=answer,
            sources=sources,
            comment=comment,
            lessons=recent_lessons(conn),
            on_text=_stream,
        )
        print("\n")
        print_sources(sources)  # reuse the original retrieval


def main() -> int:
    api_key = sanitize_env_secret("ANTHROPIC_API_KEY")
    sanitize_env_secret("ANTHROPIC_AUTH_TOKEN")
    if not (api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "Error: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) before running.",
            file=sys.stderr,
        )
        return 1

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print("Search Evaluation Agent — type a query, or 'quit' to exit.")
    print(f"(Storing to {DB_PATH})\n")

    queries_this_run = 0
    try:
        while True:
            try:
                query = input("Search query> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() in ("dashboard", "stats"):
                show_dashboard(conn)
                continue

            handle_query(client, conn, query)
            queries_this_run += 1

            if queries_this_run % DASHBOARD_EVERY == 0:
                show_dashboard(conn)
    except KeyboardInterrupt:
        print()
    finally:
        if queries_this_run:
            print("Final stats for this run:")
            show_dashboard(conn)
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
