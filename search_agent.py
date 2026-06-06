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

import anthropic

MODEL = "claude-opus-4-8"


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
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
MAX_SOURCES = 8          # cap how many sources we surface per answer


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
    # Migration: add the sources column to pre-existing databases.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(interactions)")]
    if "sources" not in cols:
        conn.execute("ALTER TABLE interactions ADD COLUMN sources TEXT")  # JSON list of {url,title}
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
) -> None:
    conn.execute(
        """
        INSERT INTO interactions
            (query_group, original_query, effective_query, answer,
             feedback, comment, attempt, created_at, sources)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
) -> int:
    """Insert an answer (with its web sources) with no feedback yet; return row id.

    Used by the web UI, where feedback arrives in a later request.
    """
    cur = conn.execute(
        """
        INSERT INTO interactions
            (query_group, original_query, effective_query, answer,
             feedback, comment, attempt, created_at, sources)
        VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?)
        """,
        (
            query_group,
            original_query,
            effective_query,
            answer,
            attempt,
            _dt.datetime.now().isoformat(timespec="seconds"),
            json.dumps(sources or []),
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
# Claude calls
# --------------------------------------------------------------------------- #
def build_system_prompt(lessons: list[str]) -> str:
    base = (
        "You are a precise search-answer assistant. Given a user's search query, "
        "produce a direct, well-structured, factual answer. Be concise but complete. "
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
    if lessons:
        joined = "\n".join(f"- {lesson}" for lesson in lessons)
        base += (
            "\n\nLessons learned from past user feedback — apply these to avoid "
            f"repeating mistakes:\n{joined}"
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


def search_and_answer(
    client: anthropic.Anthropic,
    query: str,
    lessons: list[str],
    on_text=None,
) -> tuple[str, list[dict]]:
    """Answer a query with web-search RAG.

    Claude searches the web (server-side) when the query needs live information,
    blends results with its own knowledge, and cites sources. Returns
    (answer_text, sources). If ``on_text`` is given, text chunks are streamed to
    it as they arrive (used by the terminal for live output).
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=build_system_prompt(lessons),
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": query}],
    ) as stream:
        for text in stream.text_stream:
            if on_text:
                on_text(text)
        final = stream.get_final_message()

    answer = "".join(b.text for b in final.content if b.type == "text").strip()
    return answer, extract_sources(final)


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

    with client.messages.stream(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": user}],
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
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
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
    """Compute query-group-level metrics for the dashboard."""
    rows = conn.execute(
        """
        SELECT query_group, original_query, attempt, feedback
        FROM interactions
        ORDER BY query_group, attempt
        """
    ).fetchall()

    groups: dict[int, dict] = {}
    for group, original, attempt, feedback in rows:
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

    total_queries = len(groups)
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

    # First attempt: a real web search. Retries reuse THESE results (no re-search).
    print("\n🔎 Searching the web…\n\nAnswer:\n")
    answer, sources = search_and_answer(
        client, original_query, recent_lessons(conn), on_text=_stream
    )
    print("\n")
    print_sources(sources)
    effective_query = original_query

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
        print(f'  Improved framing → "{effective_query}"\n\nAnswer:\n')
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
