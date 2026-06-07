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
from concurrent.futures import ThreadPoolExecutor

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

# Anthropic's built-in server-side web search tool. Runs the search on
# Anthropic's infrastructure, feeds results into Claude's context, and returns
# citations — the RAG retrieval layer. GA, no beta header needed.
# allowed_callers=["direct"] disables the tool's programmatic-tool-calling
# (dynamic-filtering) path, which Haiku 4.5 does not support — without it, a
# Haiku request with this tool 400s. Direct calling works on every routed model.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 5,
    "allowed_callers": ["direct"],
}
MAX_SOURCES = 8          # cap how many sources we surface per answer


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
# Category → clarifying chips. First matching category wins; else a generic set.
_CLARIFY_CATEGORIES = [
    (("tool", "tools", "software", "app", "apps", "platform", "framework",
      "library", "stack", "service"),
     ["For what purpose?", "Free or paid?", "For a team or individual?"]),
    (("learn", "study", "course", "tutorial", "book", "books"),
     ["What's your current level?", "For work or personal?", "How much time do you have?"]),
    (("buy", "cheap", "budget", "price", "phone", "laptop", "car", "gift"),
     ["What's your budget?", "What will you use it for?", "Any must-have features?"]),
    (("help", "advice", "tips", "ideas"),
     ["What are you trying to do?", "What have you tried so far?", "What's the context?"]),
]
_CLARIFY_DEFAULT = ["What's the context?", "What are you trying to achieve?", "Any specific requirements?"]


def clarifying_questions(query: str) -> list[str]:
    """Return 2-3 clarifying chips if the query is ambiguous, else an empty list.

    Ambiguous = very short (≤3 words) or a vague generic term in a short query.
    Specific multi-word queries (even with 'best') are left alone. Chips are
    chosen contextually from the query's category.
    """
    q = (query or "").lower().strip()
    words = [w.strip(".,!?\"'()") for w in q.split()]
    word_count = len(words)
    has_vague = any(w in _VAGUE_TERMS for w in words)

    ambiguous = word_count <= 3 or (has_vague and word_count <= 4)
    if not ambiguous:
        return []

    for keywords, chips in _CLARIFY_CATEGORIES:
        if any(w in keywords for w in words):
            return chips
    return _CLARIFY_DEFAULT


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
    _ensure("lessons", "session_id", "TEXT")
    conn.commit()


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
) -> None:
    conn.execute(
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
) -> int:
    """Persist a freshly generated A/B pair (no preference yet); return its id."""
    cur = conn.execute(
        """
        INSERT INTO comparisons
            (query_group, session_id, conversation_id, original_query, effective_query,
             answer_a, answer_b, sources_a, sources_b, routing, preferred, reason,
             lessons_applied, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
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
               answer_a, answer_b, sources_a, sources_b, routing, preferred, reason
        FROM comparisons WHERE id = ?
        """,
        (comparison_id,),
    ).fetchone()
    if not row:
        return None
    keys = ["id", "query_group", "session_id", "conversation_id", "original_query",
            "effective_query", "answer_a", "answer_b", "sources_a", "sources_b",
            "routing", "preferred", "reason"]
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
    """Create a lesson from a preference, or reinforce an existing matching one.

    Lessons are GLOBAL — every session teaches one shared SearchIQ (the dashboard
    aggregates them across all users). If a lesson already exists for the same
    topic + winning style, its reason is refreshed rather than duplicated. The
    originating ``session_id`` is still stored for provenance. Returns the lesson.
    """
    existing = conn.execute(
        "SELECT id FROM lessons WHERE topic = ? AND winning_style = ?",
        (topic, winning_style),
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


def match_lessons(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """Return lessons relevant to a query (topic appears in it), across all sessions.

    Lessons are shared globally, so a winning style learned by anyone becomes the
    preferred pattern for SIMILAR future queries. Returns dicts {id, text, topic}.
    """
    q = (query or "").lower()
    rows = conn.execute(
        "SELECT id, topic, text FROM lessons ORDER BY id DESC"
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


def all_lessons(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Return the global lessons feed for the dashboard, newest first."""
    rows = conn.execute(
        """
        SELECT id, text, topic, applied_count, created_at
        FROM lessons ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {"id": int(i), "text": txt, "topic": t, "applied_count": int(ac), "created_at": ca}
        for (i, txt, t, ac, ca) in rows
    ]


def training_progress(conn: sqlite3.Connection) -> dict:
    """Global totals for the Training Progress stat: lessons learned + times applied."""
    learned = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    applied = conn.execute("SELECT COALESCE(SUM(applied_count), 0) FROM lessons").fetchone()[0]
    return {"lessons_learned": int(learned), "lessons_applied": int(applied)}


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


def generate_ab_answers(
    client: anthropic.Anthropic,
    query: str,
    lessons: list[str],
    routing: dict,
    history: list[dict] | None = None,
) -> dict:
    """Generate two web-grounded answers in parallel for side-by-side comparison.

    Answer A is factual & concise; Answer B is detailed & analytical. Both run
    web search with the routed model and share the same conversation ``history``;
    running them on threads keeps total latency close to a single answer.

    Resilient to a flaky API: if one of the two calls fails, we fall back to
    whichever answer succeeded and flag ``single`` so the caller serves one
    answer instead of failing the whole request. Returns
    {"a": {...}, "b": {...} | None, "single": bool}. Raises only if BOTH fail.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "a": pool.submit(search_and_answer, client, query, lessons, routing, STYLE_A, None, history),
            "b": pool.submit(search_and_answer, client, query, lessons, routing, STYLE_B, None, history),
        }
        results: dict[str, dict] = {}
        errors: list[Exception] = []
        for key, fut in futures.items():  # .result() waits for both either way
            try:
                ans, src = fut.result()
                results[key] = {"answer": ans, "sources": src}
            except Exception as exc:  # noqa: BLE001 — keep the other answer alive
                errors.append(exc)

    if "a" in results and "b" in results:
        return {"a": results["a"], "b": results["b"], "single": False}
    if results:  # exactly one survived — serve it as a single answer
        only = results.get("a") or results.get("b")
        return {"a": only, "b": None, "single": True}
    raise errors[0]  # both failed — surface the error to the caller


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
def compute_metrics(conn: sqlite3.Connection) -> dict:
    """Compute query-group-level metrics for the dashboard, GLOBALLY.

    The North Star — total queries, completion rate, model mix, cost saved — is
    an all-time, all-session view of the whole system (only the conversation
    history sidebar is per-session).
    """
    rows = conn.execute(
        """
        SELECT query_group, original_query, attempt, feedback, routing
        FROM interactions
        ORDER BY query_group, attempt
        """
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
