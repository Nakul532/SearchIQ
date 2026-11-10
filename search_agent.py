"""Search Evaluation Agent (SearchIQ backend).

A search agent that:
  1. Takes a user search query.
  2. Searches the web with Tavily, then answers with Groq-hosted models.
  3. Collects A/B preferences and thumbs up/down feedback.
  4. Persists every query, answer, and piece of feedback (Turso / local SQLite).
  5. Learns over time: rejected answers are rephrased and retried, and
     preferences become "lessons", rewards and a per-user policy.
  6. Tracks a North Star Metric: Task Completion Rate (% of queries that
     eventually earned positive feedback).

Run the CLI:
    export GROQ_API_KEY=... TAVILY_API_KEY=...
    python search_agent.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

import db
import providers

# Load a local .env (if present) so GROQ_API_KEY / TAVILY_API_KEY etc. can live in
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
# Pipeline: EVERY query runs ONE Tavily web search first (shared by answers A
# and B). A rule-based pre-processor (classify_query) then sizes the query and
# picks which Groq model AGGREGATES the search results into the answer:
#   simple  → fast model (factual / definitional / single facts)
#   medium  → fast model (comparisons / explanations / how-does-X-work)
#   complex → deep model (analysis / research / multi-step reasoning)
# Model ids live in providers.py (env-overridable).
# --------------------------------------------------------------------------- #
FAST_MODEL = providers.FAST_MODEL
DEEP_MODEL = providers.DEEP_MODEL
FAST_LABEL = providers.FAST_LABEL
DEEP_LABEL = providers.DEEP_LABEL

# Kept for the rephrase step (a small rewriting call).
MODEL = FAST_MODEL

# Per-tier routing config — model, badge icon, and the one-word "mode" label.
TIERS = {
    "simple":  {"model": FAST_MODEL, "icon": "⚡", "mode": "Fast"},
    "medium":  {"model": FAST_MODEL, "icon": "🔍", "mode": "Balanced"},
    "complex": {"model": DEEP_MODEL, "icon": "🧠", "mode": "Deep"},
}

# Rough per-call token profile, used ONLY for the dashboard's labelled cost
# estimate at Groq list prices (we don't meter per-interaction tokens).
AVG_INPUT_TOKENS = 1800
AVG_OUTPUT_TOKENS = 900


def label_for(model: str | None) -> str:
    return providers.model_label(model or FAST_MODEL)


def estimate_query_cost(model: str) -> float:
    """Estimated USD cost of one answer call on ``model`` at Groq list prices."""
    pin, pout = providers.LIST_PRICES.get(model, providers.LIST_PRICES.get(DEEP_MODEL, (0.15, 0.6)))
    return AVG_INPUT_TOKENS / 1e6 * pin + AVG_OUTPUT_TOKENS / 1e6 * pout


DB_PATH = db.LOCAL_DB_PATH
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

# Tavily (basic search, 1 credit) runs ONCE per live query; both A/B drafts are
# grounded in the same results.
MAX_SOURCES = 3          # cap how many sources we surface per answer (top 3)
ANSWER_TIMEOUT_S = 75    # hard per-answer generation timeout (each A/B draft)
HISTORY_TURNS = 6        # prior turns replayed for context (kept small: free-tier TPM)
HISTORY_CHARS = 700      # per-turn truncation for replayed history


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
    "⚡ Simple · GPT-OSS 20B · Fast" / "🧠 Complex · GPT-OSS 120B · Deep".
    """
    tier = TIERS.get(complexity, TIERS["medium"])
    model = tier["model"]
    model_label = label_for(model)
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
    Answer A uses the fast model (concise) and Answer B the deep model
    (thorough, detailed). For simpler tiers both sides keep the (policy-biased) base routing.
    The complex split intentionally overrides the policy's model for the A/B pair —
    that latency strategy is the whole point of the split.
    """
    if base_routing.get("complexity") == "complex":
        tier = TIERS["complex"]
        a = dict(base_routing)
        a.update({
            "model": FAST_MODEL,
            "model_label": FAST_LABEL,
            "badge": f"{tier['icon']} Complex · {FAST_LABEL} · {tier['mode']}",
        })
        a.pop("policy_biased", None)  # this side is fixed to the fast model by design
        b = build_routing("complex")  # deep model
        return a, b
    return dict(base_routing), dict(base_routing)


def _effort(model: str) -> str:
    """Reasoning effort for gpt-oss models: a little more thinking on the deep
    model, minimal on the fast one (reasoning tokens count toward free-tier TPD)."""
    return "medium" if model == DEEP_MODEL else "low"


# --------------------------------------------------------------------------- #
# Clarifying questions — disambiguate vague queries before answering
# --------------------------------------------------------------------------- #
# Generic "vague" terms that signal an under-specified query.
_VAGUE_TERMS = {"best", "good", "top", "help", "recommend", "recommendation",
                "recommendations", "tips", "advice", "ideas", "options"}
# Category → a natural, conversational fallback question (used when the LLM call
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


def clarifying_question(query: str) -> str | None:
    """If the query is ambiguous, return ONE natural clarifying question to ask.

    The agent asks this in the chat like a person would; the user answers in their
    own words (no chips/buttons), and that reply — together with this question, both
    kept in the conversation history — gives the model the context to answer the
    original query. The question is phrased by the fast model for a human touch and tailored
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
        text = providers.chat(FAST_MODEL, [{"role": "user", "content": prompt}],
                              max_tokens=400).strip().strip('"').strip()
        if text:
            return text
    except (providers.ProviderLimit, providers.ProviderError):
        pass
    return fallback


def sanitize_env_secret(name: str) -> str | None:
    """Strip non-ASCII chars (e.g. a pasted U+2028) and whitespace from a secret.

    HTTP headers must be ASCII, so a stray Unicode character in the API key
    raises "'ascii' codec can't encode character ...". API keys are pure
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
_SCHEMA_READY: set[str] = set()


def init_db(conn) -> None:
    """Create every table (once per process per database; cheap no-op after)."""
    key = db.describe()
    if key in _SCHEMA_READY:
        return
    stmts: list[str] = []
    stmts.append(
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
    stmts.append(
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
    stmts.append(
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
    stmts.append(
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
    stmts.append(
        """
        CREATE TABLE IF NOT EXISTS rewards (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER,            -- the interaction this reward is for (may be NULL)
            query_group    INTEGER,            -- groups rewards by query (for per-query totals)
            session_id     TEXT,
            signal         TEXT    NOT NULL,   -- 'ab_winner' | 'ab_loser' | 'thumbs_up' | … | 'reason_bonus'
            score          REAL    NOT NULL,   -- the numeric reward
            model          TEXT,               -- model label (which model produced the answer)
            style          TEXT,               -- 'concise' | 'detailed' | NULL (single/followup)
            query_type     TEXT,               -- 'simple' | 'medium' | 'complex'
            created_at     TEXT    NOT NULL
        )
        """
    )
    # Learned policy updates — each row is one re-evaluation that picked the
    # highest-reward model+style+query_type pattern and adjusted behavior to it.
    stmts.append(
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
    stmts.append(
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
    stmts.append(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            google_id       TEXT PRIMARY KEY,
            expertise_level TEXT,              -- beginner | intermediate | expert
            preferred_style TEXT,              -- concise | detailed
            top_topics      TEXT,              -- JSON list of subjects
            preferred_model TEXT,              -- model label (highest-reward)
            summary         TEXT,              -- the system-prompt sentence
            queries_at      INTEGER,           -- # of the user's queries at build time
            updated_at      TEXT    NOT NULL
        )
        """
    )

    import limits as _limits  # app guard-rail tables (counters + answer cache)
    stmts.extend(_limits.SCHEMA)
    if hasattr(conn, "execute_batch"):
        conn.execute_batch([(sql, ()) for sql in stmts])
    else:
        for sql in stmts:
            conn.execute(sql)

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
    # Per-side routing — A and B can use different models (fast A / deep B on
    # complex queries), so the reward system can attribute the winner to its model.
    _ensure("comparisons", "routing_a", "TEXT")
    _ensure("comparisons", "routing_b", "TEXT")
    _ensure("lessons", "session_id", "TEXT")
    # Policy is now learned per user (scoped by the originating session_id/google_id).
    _ensure("policy_updates", "session_id", "TEXT")
    conn.commit()
    _SCHEMA_READY.add(key)


# The app's per-user data tables (used by /api/reset). `users` is excluded so
# accounts survive a reset; user_profiles is derived data and regenerates.
DATA_TABLES = ("interactions", "comparisons", "lessons", "messages",
               "rewards", "policy_updates", "user_profiles")


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
    session_id: str | None = None,
) -> int:
    """Insert an answer (with its web sources) with no feedback yet; return row id.

    Used by the web UI, where feedback arrives in a later request.
    """
    cur = conn.execute(
        """
        INSERT INTO interactions
            (query_group, original_query, effective_query, answer,
             feedback, comment, attempt, created_at, sources, routing, session_id)
        VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
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
            session_id,
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


# Topic categories for the personality profile — related words/phrases grouped
# under a clean display name. Checked in order (specific → general), so a query
# about LLMs lands in "Artificial Intelligence" before the broad "Technology".
# Single short tokens (ai, ml, vc) match as whole words; longer keywords match a
# token prefix (so "invest" catches "investing"); phrases match as substrings.
_TOPIC_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("Artificial Intelligence", ("ai", "artificial intelligence", "llm", "llms", "gpt",
        "chatgpt", "claude", "gemini", "openai", "anthropic", "neural", "deep learning",
        "transformer", "generative", "rag", "chatbot", "prompt", "agentic", "agent", "agents")),
    ("Machine Learning", ("ml", "machine learning", "dataset", "classifier", "regression",
        "clustering", "scikit", "pytorch", "tensorflow", "fine-tuning", "fine-tune",
        "embedding", "embeddings", "model training")),
    ("Startups & Founders", ("startup", "startups", "founder", "venture capital", "vc",
        "seed round", "series a", "fundraising", "fundraise", "saas", "mvp", "pitch deck",
        "y combinator", "bootstrapping")),
    ("Programming", ("python", "javascript", "typescript", "rust", "golang", "code",
        "coding", "programming", "developer", "api", "framework", "react", "nodejs",
        "sql", "database", "git", "debugging", "algorithm", "compiler")),
    ("Finance & Investing", ("stock", "stocks", "investing", "investment", "crypto",
        "bitcoin", "ethereum", "portfolio", "trading", "etf", "interest rate", "valuation",
        "dividend")),
    ("Business & Marketing", ("business", "marketing", "sales", "strategy", "customer",
        "b2b", "ecommerce", "advertising", "branding", "revenue", "growth")),
    ("Science", ("physics", "chemistry", "biology", "quantum", "genetics", "astronomy",
        "neuroscience", "climate", "experiment", "scientific")),
    ("Health & Fitness", ("health", "fitness", "diet", "nutrition", "workout", "exercise",
        "medical", "disease", "symptom", "wellness")),
    ("News & Current Events", ("news", "latest", "breaking", "election", "politics",
        "political", "government", "policy", "regulation")),
    ("Technology", ("technology", "tech", "software", "hardware", "gadget", "computer",
        "cloud", "cybersecurity", "smartphone", "device")),
    ("Travel", ("travel", "flight", "flights", "hotel", "vacation", "destination",
        "tourism", "itinerary")),
    ("Education & Learning", ("learn", "learning", "course", "courses", "tutorial",
        "study", "textbook", "university", "degree", "certification")),
    ("Food & Cooking", ("recipe", "recipes", "cooking", "food", "restaurant", "cuisine",
        "baking", "meal")),
    ("Entertainment", ("movie", "movies", "film", "music", "gaming", "videogame", "novel",
        "celebrity", "netflix")),
    ("Sports", ("sports", "football", "soccer", "basketball", "nba", "nfl", "tennis",
        "cricket", "olympics", "baseball")),
]
# Generic verbs / filler that should never become a fallback topic on their own.
_TOPIC_FALLBACK_STOP = {
    "make", "makes", "made", "analyze", "compare", "explain", "find", "tell", "give",
    "list", "get", "help", "need", "want", "know", "use", "using", "build", "create",
    "show", "write", "today", "latest", "recent", "current", "thing", "things", "stuff",
}


def categorize_query(query: str) -> str | None:
    """Map a query to a clean topic category name (or None if nothing matches)."""
    q = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9+#]+", q))
    for name, keywords in _TOPIC_CATEGORIES:
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in q:                       # multi-word phrase → substring
                    return name
            elif len(kw) <= 3:
                if kw in tokens:                  # short token (ai, ml, vc) → whole word
                    return name
            elif any(t == kw or t.startswith(kw) for t in tokens):  # prefix (invest→investing)
                return name
    return None


def profile_topics(queries: list[str], n: int = 3) -> list[str]:
    """Return the user's top ``n`` topics as clean phrases (e.g. "Artificial
    Intelligence", "Startups & Founders") by grouping query words into categories.

    Counts how often each category appears across the user's queries. If fewer
    than ``n`` categories are found, fills the rest with Title-cased salient words
    (skipping generic verbs) so the profile still shows meaningful labels.
    """
    cat_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    for q in queries:
        category = categorize_query(q)
        if category:
            cat_counts[category] = cat_counts.get(category, 0) + 1
            continue
        word = extract_topic(q)
        if word and word not in _TOPIC_FALLBACK_STOP and len(word) > 2:
            label = word.title()
            fallback_counts[label] = fallback_counts.get(label, 0) + 1
    topics = [c for c, _ in sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(topics) < n:
        extra = [w for w, _ in sorted(fallback_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                 if w not in topics]
        topics += extra
    return topics[:n]


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
    conn: sqlite3.Connection, conversation_id: str, limit: int = 10,
    session_id: str | None = None,
) -> list[dict]:
    """Return the last ``limit`` turns as [{role, content}] for the model's context.

    This is what makes follow-ups work ("tell me more about point 2"): the prior
    user/assistant turns are replayed to the model before the new question.
    Scoped to ``session_id`` when given, so one user can't pull another's turns.
    """
    if session_id is None:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? AND session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, session_id, limit),
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
        model = label_for(routing.get("model"))
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
        "by_model": {FAST_LABEL: by_model.get(FAST_LABEL, 0.0), DEEP_LABEL: by_model.get(DEEP_LABEL, 0.0)},
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
    if target_label not in (FAST_LABEL, DEEP_LABEL) or routing.get("model_label") == target_label:
        return routing
    complexity = routing.get("complexity", "medium")
    tier = TIERS.get(complexity, TIERS["medium"])
    biased = dict(routing)
    biased["model"] = DEEP_MODEL if target_label == DEEP_LABEL else FAST_MODEL
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
    query_texts: list[str] = []
    for original_query, routing in rows:
        nq += 1
        words_total += len((original_query or "").split())
        query_texts.append(original_query or "")
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
    preferred_model = DEEP_LABEL if by_model.get(DEEP_LABEL, 0.0) >= by_model.get(FAST_LABEL, 0.0) else FAST_LABEL
    if stats["total_events"] == 0:
        preferred_model = None  # no reward data yet — don't claim a model preference
    top_topics = profile_topics(query_texts)

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
# LLM calls (Groq) grounded in Tavily search results
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
        "You are a precise search-answer assistant. Given a user's search query and "
        "fresh web search results, produce a direct, well-structured, factual answer. "
        "If the query is ambiguous, answer the most likely intent and note the assumption. "
        "Respond directly without preamble like 'Here is' or 'Sure'.\n\n"
        "GROUNDING REQUIREMENT: Base your answer on the numbered search results you are "
        "given, not on prior knowledge. If a source says X happened, report X. Closely "
        "paraphrase the relevant passages, use your own knowledge only to add light "
        "context around the sourced facts, and cite sources inline as [1], [2], [3]. "
        "If the results don't answer the question, say so plainly."
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


def sources_from_results(results: list[dict]) -> list[dict]:
    """The {url, title} list shown under an answer (top MAX_SOURCES results)."""
    out, seen = [], set()
    for r in results or []:
        url = r.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "title": r.get("title") or url})
    return out[:MAX_SOURCES]


def _format_results(results: list[dict]) -> str:
    if not results:
        return "(no search results were returned)"
    blocks = []
    for i, r in enumerate(results[:MAX_SOURCES], 1):
        date = f" — published {r['published_date']}" if r.get("published_date") else ""
        blocks.append(f"[{i}] {r.get('title') or r.get('url')}{date}\n{r.get('url')}\n{r.get('content') or ''}")
    return "\n\n".join(blocks)


def _history_messages(history: list[dict] | None, query: str, results: list[dict]) -> list[dict]:
    """Build the messages array: recent conversation turns, then the new question
    with its search results.

    ``history`` is [{role, content}] from conversation_history(). Only the last
    HISTORY_TURNS user/assistant turns are kept (each truncated) so prompts stay
    inside Groq's free-tier tokens-per-minute limits.
    """
    msgs: list[dict] = []
    for h in (history or [])[-HISTORY_TURNS:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content[:HISTORY_CHARS]})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    today = _dt.date.today().isoformat()
    msgs.append({
        "role": "user",
        "content": f"Today's date: {today}\n\nWeb search results:\n\n{_format_results(results)}"
                   f"\n\nQuestion: {query}",
    })
    return msgs


# Output budgets per style (gpt-oss reasoning tokens count toward these too).
_MAX_TOKENS = {"A": 1400, "B": 2600}


def search_and_answer(
    query: str,
    lessons: list[str],
    routing: dict,
    style: str | None = None,
    on_text=None,
    history: list[dict] | None = None,
    results: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Answer a query from Tavily results, aggregated by the routed Groq model.

    ``results`` are the pre-fetched search results (one search is shared by A
    and B); when omitted, a search is run here (CLI path). ``routing`` (from
    classify_query) selects the model; ``style`` (STYLE_A / STYLE_B) tunes the
    synthesis for the A/B comparison. Returns (answer_text, sources).
    """
    if results is None:
        results = providers.search(query, max_results=MAX_SOURCES)
    model = routing["model"]
    budget = _MAX_TOKENS["B"] if style == STYLE_B else _MAX_TOKENS["A"]
    answer = providers.chat(
        model,
        _history_messages(history, query, results),
        max_tokens=budget,
        system=build_system_prompt(lessons, style=style),
        effort=_effort(model),
    )
    if on_text:
        on_text(answer)
    return answer, sources_from_results(results)


def iter_ab_answers(
    query: str,
    lessons: list[str],
    routing_by_side: dict[str, dict],
    history: list[dict] | None = None,
    results: list[dict] | None = None,
    timeout: float = ANSWER_TIMEOUT_S,
):
    """Generate Answer A and B in parallel from the SAME search results, yielding
    each as soon as it's ready.

    Yields ``(side, {"answer", "sources"})`` on success or ``(side, {"error": msg,
    "limit": bool})`` when a draft fails (``limit`` = provider rate limit) or exceeds
    ``timeout`` seconds. A is factual & concise (STYLE_A); B is detailed &
    analytical (STYLE_B).
    """
    if results is None:
        results = providers.search(query, max_results=MAX_SOURCES)
    styles = {"A": STYLE_A, "B": STYLE_B}
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {
        pool.submit(search_and_answer, query, lessons, routing_by_side[side],
                    styles[side], None, history, results): side
        for side in ("A", "B")
    }
    pending = set(futs)
    try:
        for fut in as_completed(futs, timeout=timeout):
            side = futs[fut]
            pending.discard(fut)
            try:
                ans, src = fut.result()
                if not ans:
                    raise providers.ProviderError("empty answer")
                yield side, {"answer": ans, "sources": src}
            except providers.ProviderLimit as exc:
                yield side, {"error": str(exc), "limit": True}
            except Exception as exc:  # noqa: BLE001 — keep the other side alive
                yield side, {"error": str(exc) or exc.__class__.__name__, "limit": False}
    except FuturesTimeoutError:
        pass  # remaining sides blew the per-answer budget; reported below
    finally:
        pool.shutdown(wait=False)
    for fut in pending:
        fut.cancel()
        yield futs[fut], {"error": f"timed out after {int(timeout)}s", "limit": False}


def suggest_followups(query: str, answer: str, n: int = 3) -> list[str]:
    """Suggest ``n`` short follow-up questions based on an answer (fast model).

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
        text = providers.chat(FAST_MODEL, [{"role": "user", "content": prompt}], max_tokens=500)
    except (providers.ProviderLimit, providers.ProviderError):
        return []
    lines = [
        ln.strip().lstrip("-*0123456789. ").strip()
        for ln in text.splitlines()
        if ln.strip()
    ]
    return [ln for ln in lines if ln][:n]


def regenerate_answer(
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

    Used for retries after a thumbs-down: no second Tavily search (no credit
    spent), only the answer is regenerated with the deep model, grounded in the
    same findings (the prior answer + the original source list).
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
        f"Previous answer (grounded in the sources):\n{previous_answer[:4000]}\n\n"
        f"Sources that were retrieved:\n{source_lines or '(none)'}\n\n"
        "Write an improved answer that fixes the user's complaint while staying "
        "faithful to the gathered information above. Do not search again."
    )
    text = providers.chat(DEEP_MODEL, [{"role": "user", "content": user}], max_tokens=2600,
                          system=system, effort=_effort(DEEP_MODEL))
    if on_text:
        on_text(text)
    return text


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
    *,
    original_query: str,
    failed_answer: str,
    comment: str | None,
) -> str:
    """Ask the fast model to improve a query that produced an unhelpful answer."""
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
    text = providers.chat(MODEL, [{"role": "user", "content": prompt}], max_tokens=400).strip()
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

    The North Star — total queries, completion rate, model mix, cost estimate — is
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
    # Per-model-call tallies (every interaction is one answer call).
    models_used = {FAST_LABEL: 0, DEEP_LABEL: 0}
    est_cost = 0.0          # estimated $ at Groq list prices (free tier → $0 actual)
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

        if routing:
            try:
                rt = json.loads(routing)
            except (json.JSONDecodeError, TypeError):
                rt = None
            if rt:
                model = rt.get("model") or FAST_MODEL
                label = rt.get("model_label") or label_for(model)
                models_used[label] = models_used.get(label, 0) + 1
                est_cost += estimate_query_cost(model)

    total_queries = len(groups)
    calls = sum(models_used.values())
    fast_share_pct = round(models_used.get(FAST_LABEL, 0) / calls * 100, 1) if calls else 0.0
    est_cost_per_query_usd = (est_cost / total_queries) if total_queries else 0.0
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
        "model_labels": {"fast": FAST_LABEL, "deep": DEEP_LABEL},
        "fast_share_pct": fast_share_pct,
        "est_cost_per_query_usd": round(est_cost_per_query_usd, 6),
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
    print(f"  🤖 Models used        : {FAST_LABEL} {mu.get(FAST_LABEL, 0)} · {DEEP_LABEL} {mu.get(DEEP_LABEL, 0)}")
    print(f"  ⚡ Fast-model share   : {m['fast_share_pct']:.0f}%")
    print(f"  💰 Est. cost / query  : ${m['est_cost_per_query_usd']:.5f}  (estimate at Groq list prices)")
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
def handle_query(conn, original_query: str) -> None:
    """Run one query through search → feedback → (reuse-results retry) cycles."""
    group = next_query_group(conn)
    _stream = lambda t: print(t, end="", flush=True)  # noqa: E731

    # Pre-process: size the query and pick the model before calling Groq.
    routing = classify_query(original_query)
    print(f"\n🧭 Routing: {routing['badge']}")
    print("\n🔎 Searching the web…\n\nAnswer:\n")
    answer, sources = search_and_answer(
        original_query, recent_lessons(conn), routing, on_text=_stream
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
            original_query=original_query,
            failed_answer=answer,
            comment=comment,
        )
        print(f'  Improved framing → "{effective_query}"')
        print(f"  🧭 Routing: {retry_routing['badge']}\n\nAnswer:\n")
        answer = regenerate_answer(
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
    sanitize_env_secret("GROQ_API_KEY")
    sanitize_env_secret("TAVILY_API_KEY")
    if not providers.configured():
        print("Error: set GROQ_API_KEY and TAVILY_API_KEY before running.", file=sys.stderr)
        return 1

    conn = db.connect()
    init_db(conn)

    print("Search Evaluation Agent — type a query, or 'quit' to exit.")
    print(f"(Storing to {db.describe()})\n")

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

            handle_query(conn, query)
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
