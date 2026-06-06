"""Flask web UI for the Search Evaluation Agent.

Reuses the search_agent.py backend (Claude calls, SQLite storage, rephrase /
retry, and metrics) and exposes it over a small JSON API consumed by a
single-page dark-theme dashboard.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py
    # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import sqlite3

import anthropic
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

import search_agent as sa

# Strip stray non-ASCII chars (e.g. a pasted U+2028) from the credentials before
# building the client — HTTP headers must be ASCII or the SDK raises an
# 'ascii' codec encode error. See search_agent.sanitize_env_secret.
_API_KEY = sa.sanitize_env_secret("ANTHROPIC_API_KEY")
sa.sanitize_env_secret("ANTHROPIC_AUTH_TOKEN")

app = Flask(__name__)
client = anthropic.Anthropic(api_key=_API_KEY) if _API_KEY else anthropic.Anthropic()


# --------------------------------------------------------------------------- #
# JSON error handling — every route (and every failure) returns JSON.
# --------------------------------------------------------------------------- #
def json_error(message: str, status: int = 500, kind: str = "error"):
    return jsonify({"error": message, "type": kind}), status


def claude_error(exc: Exception):
    """Map an Anthropic SDK exception to a (message, status) JSON error."""
    if isinstance(exc, anthropic.AuthenticationError):
        return json_error(
            "Authentication failed — check that ANTHROPIC_API_KEY is set to a valid key.",
            401,
            "authentication_error",
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return json_error("API key lacks permission for this model.", 403, "permission_error")
    if isinstance(exc, anthropic.RateLimitError):
        return json_error("Rate limited by the Anthropic API — please retry shortly.", 429, "rate_limit_error")
    if isinstance(exc, anthropic.APIConnectionError):
        return json_error("Could not reach the Anthropic API — check your connection.", 502, "connection_error")
    if isinstance(exc, anthropic.APIStatusError):
        return json_error(f"Anthropic API error ({exc.status_code}).", 502, "api_error")
    return None  # not an Anthropic error we recognize


@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    return jsonify({"error": e.description, "type": "http_error", "status": e.code}), e.code


@app.errorhandler(Exception)
def handle_unexpected(e: Exception):
    mapped = claude_error(e)
    if mapped is not None:
        return mapped
    app.logger.exception("Unhandled error")
    return json_error(str(e) or "Internal server error", 500, "server_error")


# --------------------------------------------------------------------------- #
# DB helpers — one connection per request (SQLite + threads don't mix).
# --------------------------------------------------------------------------- #
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(sa.DB_PATH)
    sa.init_db(conn)
    return conn


def generate_answer_text(query: str, lessons: list[str]) -> tuple[str, list[dict]]:
    """Web-search RAG answer for the web flow. Returns (answer, sources)."""
    return sa.search_and_answer(client, query, lessons)


def metrics_payload(conn: sqlite3.Connection) -> dict:
    m = sa.compute_metrics(conn)
    rows = conn.execute(
        """
        SELECT feedback FROM interactions
        WHERE feedback IS NOT NULL
        ORDER BY id DESC LIMIT 20
        """
    ).fetchall()
    recent = [r[0] for r in rows][::-1]  # oldest → newest
    m["recent_feedback"] = recent
    m["thumbs_up"] = recent.count("up")
    m["thumbs_down"] = recent.count("down")
    m["improved_count"] = len(m["most_improved"])
    return m


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    conn = db()
    try:
        return jsonify(metrics_payload(conn))
    finally:
        conn.close()


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return json_error("Query is empty.", 400, "invalid_request")

    conn = db()
    try:
        group = sa.next_query_group(conn)
        lessons = sa.recent_lessons(conn)
        try:
            answer, sources = generate_answer_text(query, lessons)
        except anthropic.APIError as exc:
            return claude_error(exc) or json_error(str(exc), 502, "api_error")

        interaction_id = sa.insert_answer(
            conn,
            query_group=group,
            original_query=query,
            effective_query=query,
            answer=answer,
            attempt=1,
            sources=sources,
        )
        return jsonify(
            {
                "interaction_id": interaction_id,
                "group_id": group,
                "original_query": query,
                "effective_query": query,
                "answer": answer,
                "sources": sources,
                "attempt": 1,
                "lessons_applied": len(lessons),
            }
        )
    finally:
        conn.close()


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    interaction_id = data.get("interaction_id")
    group_id = data.get("group_id")
    original_query = str(data.get("original_query", ""))
    feedback = data.get("feedback")  # 'up' | 'down'
    comment = (data.get("comment") or "").strip() or None

    if interaction_id is None or group_id is None or feedback not in ("up", "down"):
        return json_error("Missing interaction_id/group_id or invalid feedback.", 400, "invalid_request")

    conn = db()
    try:
        sa.set_feedback(conn, int(interaction_id), feedback, comment)

        retry = None
        learning = False
        if feedback == "down":
            attempts = sa.attempts_in_group(conn, int(group_id))
            if attempts <= sa.MAX_RETRIES:  # 1 initial + MAX_RETRIES retries
                learning = True
                failed_answer = sa.get_answer(conn, int(interaction_id))
                # Reuse the ORIGINAL search results — no second web search (keeps
                # the retry fast and well under the request timeout).
                reused_sources = sa.get_sources(conn, int(interaction_id))
                try:
                    new_query = sa.rephrase_query(
                        client,
                        original_query=original_query,
                        failed_answer=failed_answer,
                        comment=comment,
                    )
                    lessons = sa.recent_lessons(conn)
                    new_answer = sa.regenerate_answer(
                        client,
                        original_query=original_query,
                        improved_query=new_query,
                        previous_answer=failed_answer,
                        sources=reused_sources,
                        comment=comment,
                        lessons=lessons,
                    )
                except anthropic.APIError as exc:
                    # Feedback is already saved; report the retry failure as JSON.
                    return claude_error(exc) or json_error(str(exc), 502, "api_error")
                new_id = sa.insert_answer(
                    conn,
                    query_group=int(group_id),
                    original_query=original_query,
                    effective_query=new_query,
                    answer=new_answer,
                    attempt=attempts + 1,
                    sources=reused_sources,
                )
                retry = {
                    "interaction_id": new_id,
                    "group_id": int(group_id),
                    "original_query": original_query,
                    "effective_query": new_query,
                    "answer": new_answer,
                    "sources": reused_sources,
                    "attempt": attempts + 1,
                    "lessons_applied": len(lessons),
                }

        return jsonify(
            {
                "status": "ok",
                "learning": learning,
                "retry": retry,
                "metrics": metrics_payload(conn),
            }
        )
    finally:
        conn.close()


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear all interactions — handy for starting a clean demo."""
    conn = db()
    try:
        conn.execute("DELETE FROM interactions")
        conn.commit()
        return jsonify({"status": "ok", "metrics": metrics_payload(conn)})
    finally:
        conn.close()


if __name__ == "__main__":
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Warning: ANTHROPIC_API_KEY is not set — queries will fail until you set it.")
    # Debug off by default: the Werkzeug interactive debugger returns an HTML
    # error page that overrides our JSON error handlers. Opt in with FLASK_DEBUG=1.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    # Bind to 0.0.0.0 and the platform-provided $PORT so Railway (and other
    # PaaS hosts) can route to the app; fall back to 5000 for local dev.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
