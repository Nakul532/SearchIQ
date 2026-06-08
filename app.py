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

import json
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


def session_id() -> str | None:
    """The per-browser session id sent in the X-Session-Id header (or None)."""
    return (request.headers.get("X-Session-Id") or "").strip() or None


def metrics_payload(conn: sqlite3.Connection) -> dict:
    """Build the GLOBAL dashboard payload (all sessions, all time)."""
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
    # Visible learning loop: lessons feed + training-progress totals (global).
    m["lessons_feed"] = sa.all_lessons(conn)
    m["training_progress"] = sa.training_progress(conn)
    # Reinforcement-learning reward signals + the current learned policy.
    m["reward"] = sa.reward_stats(conn)
    m["policy"] = sa.current_policy(conn)
    return m


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    sid = session_id()
    conn = db()
    try:
        return jsonify(metrics_payload(conn))
    finally:
        conn.close()


@app.route("/api/conversations")
def api_conversations():
    """List this session's past conversations for the history sidebar."""
    sid = session_id()
    if not sid:
        return jsonify({"conversations": []})
    conn = db()
    try:
        return jsonify({"conversations": sa.list_conversations(conn, sid)})
    finally:
        conn.close()


@app.route("/api/conversation/<conversation_id>")
def api_conversation(conversation_id: str):
    """Load the full transcript of one past conversation (scoped to the session)."""
    sid = session_id()
    if not sid:
        return json_error("Missing session id.", 400, "invalid_request")
    conn = db()
    try:
        return jsonify({
            "conversation_id": conversation_id,
            "messages": sa.load_conversation(conn, sid, conversation_id),
        })
    finally:
        conn.close()


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    force = bool(data.get("force"))  # bypass the clarifying-question check
    followup = bool(data.get("followup"))  # query came from a follow-up chip
    conversation_id = str(data.get("conversation_id", "")).strip()
    sid = session_id()
    if not query:
        return json_error("Query is empty.", 400, "invalid_request")
    if not sid or not conversation_id:
        return json_error("Missing session or conversation id.", 400, "invalid_request")

    # Pre-processor step 1: if the query is ambiguous, ask before answering.
    # The user can tap a chip (which refines the query) or resend with force=true.
    if not force:
        chips = sa.clarifying_questions(query)
        if chips:
            return jsonify({"clarify": chips, "original_query": query})

    conn = db()
    try:
        # Reward the PRIOR answer when this query is a follow-up — the user was
        # engaged enough to keep going (+0.2). Attributed before we record the new
        # turn, so it lands on the answer that prompted the continuation.
        if followup:
            prev = sa.last_interaction(conn, sid)
            if prev:
                p_model, p_qtype = sa.routing_facets(prev["routing"])
                sa.record_reward(
                    conn, signal="followup", score=sa.REWARD_SCORES["followup"],
                    interaction_id=prev["id"], query_group=prev["query_group"],
                    session_id=sid, model=p_model, query_type=p_qtype,
                )

        # Load prior turns for context (this question is saved only after the
        # answers come back, so a transient API failure leaves no dangling turn).
        history = sa.conversation_history(conn, conversation_id, limit=10)

        group = sa.next_query_group(conn)
        # Apply lessons the whole system has learned from past preferences on similar queries.
        applied = sa.match_lessons(conn, query)
        applied_ids = [l["id"] for l in applied]
        # Pre-processor step 2: classify complexity → choose the aggregation model,
        # then let the learned policy bias the model toward the highest-reward one.
        routing = sa.apply_policy(conn, sa.classify_query(query))
        # Steer BOTH drafts toward the policy's winning style via the system prompt
        # (prepended to the matched lessons, which build_system_prompt renders).
        directive = sa.policy_directive(conn)
        lesson_texts = ([directive] if directive else []) + [l["text"] for l in applied]
        try:
            answers = sa.generate_ab_answers(client, query, lesson_texts, routing, history=history)
        except anthropic.APIError as exc:
            return claude_error(exc) or json_error(str(exc), 502, "api_error")

        # Record the user's turn now that we have answers to pair it with.
        sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                        role="user", content=query)
        if applied_ids:
            sa.bump_applied_count(conn, applied_ids)

        # Fallback: only one of the two answers came back. Serve it directly —
        # there's nothing to compare, so commit it as the assistant turn now and
        # skip the prefer/reason step.
        if answers.get("single"):
            single = answers["a"]
            sa.record_interaction(
                conn, query_group=group, original_query=query, effective_query=query,
                answer=single["answer"], feedback=None, comment=None, attempt=1,
                sources=single["sources"], routing=routing, session_id=sid,
            )
            sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                            role="assistant", content=single["answer"], sources=single["sources"])
            followups = sa.suggest_followups(client, query, single["answer"])
            return jsonify(
                {
                    "single": True,
                    "group_id": group,
                    "conversation_id": conversation_id,
                    "original_query": query,
                    "effective_query": query,
                    "routing": routing,
                    "answer": single["answer"],
                    "sources": single["sources"],
                    "applied_lessons": applied,
                    "applied_count": len(applied),
                    "followups": followups,
                }
            )

        comparison_id = sa.record_comparison(
            conn,
            query_group=group,
            original_query=query,
            effective_query=query,
            answers=answers,
            routing=routing,
            lessons_applied=applied_ids,
            session_id=sid,
            conversation_id=conversation_id,
        )
        return jsonify(
            {
                "comparison_id": comparison_id,
                "group_id": group,
                "conversation_id": conversation_id,
                "original_query": query,
                "effective_query": query,
                "routing": routing,
                "answer_a": answers["a"]["answer"],
                "sources_a": answers["a"]["sources"],
                "answer_b": answers["b"]["answer"],
                "sources_b": answers["b"]["sources"],
                "applied_lessons": applied,
                "applied_count": len(applied),
            }
        )
    finally:
        conn.close()


@app.route("/api/prefer", methods=["POST"])
def api_prefer():
    """Record which A/B answer the user preferred.

    Winner → a positive interaction; loser → a negative one (so the North Star
    completion-rate and feedback history stay intact). The lesson is created
    later, once the user gives a reason (see /api/prefer_reason).
    """
    data = request.get_json(silent=True) or {}
    comparison_id = data.get("comparison_id")
    preferred = data.get("preferred")  # 'A' | 'B'
    if comparison_id is None or preferred not in ("A", "B"):
        return json_error("Missing comparison_id or invalid preference.", 400, "invalid_request")

    conn = db()
    try:
        comp = sa.get_comparison(conn, int(comparison_id))
        if not comp:
            return json_error("Comparison not found.", 404, "not_found_error")

        routing = json.loads(comp["routing"]) if comp["routing"] else None
        win_answer = comp["answer_a"] if preferred == "A" else comp["answer_b"]
        lose_answer = comp["answer_b"] if preferred == "A" else comp["answer_a"]
        win_sources = comp["sources_a"] if preferred == "A" else comp["sources_b"]
        lose_sources = comp["sources_b"] if preferred == "A" else comp["sources_a"]

        def _src(s):
            try:
                return json.loads(s) if s else []
            except (json.JSONDecodeError, TypeError):
                return []

        sa.set_preference(conn, int(comparison_id), preferred)
        # Winner = thumbs up (attempt 1), loser = thumbs down (attempt 2).
        win_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=win_answer, feedback="up",
            comment=None, attempt=1, sources=_src(win_sources), routing=routing,
            session_id=comp["session_id"],
        )
        lose_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=lose_answer, feedback="down",
            comment=None, attempt=2, sources=_src(lose_sources), routing=routing,
            session_id=comp["session_id"],
        )

        # Reward signals: winner +1.0, loser -0.5 (A = concise, B = detailed).
        r_model, r_qtype = sa.routing_facets(routing)
        loser_side = "B" if preferred == "A" else "A"
        sa.record_reward(
            conn, signal="ab_winner", score=sa.REWARD_SCORES["ab_winner"],
            interaction_id=win_id, query_group=comp["query_group"], session_id=comp["session_id"],
            model=r_model, style=sa.STYLE_OF_SIDE[preferred], query_type=r_qtype,
        )
        sa.record_reward(
            conn, signal="ab_loser", score=sa.REWARD_SCORES["ab_loser"],
            interaction_id=lose_id, query_group=comp["query_group"], session_id=comp["session_id"],
            model=r_model, style=sa.STYLE_OF_SIDE[loser_side], query_type=r_qtype,
        )
        # Re-evaluate the policy after every POLICY_EVERY reward-bearing queries.
        policy_update = sa.maybe_update_policy(conn)

        # The winning answer becomes the assistant's turn in the conversation
        # transcript — so it's part of the memory passed to future questions.
        win_src = _src(win_sources)
        if comp["session_id"] and comp["conversation_id"]:
            sa.save_message(
                conn, session_id=comp["session_id"], conversation_id=comp["conversation_id"],
                role="assistant", content=win_answer, sources=win_src,
            )

        # Suggest 3 follow-up questions based on the winning answer (cheap Haiku).
        followups = sa.suggest_followups(client, comp["original_query"], win_answer)

        return jsonify({
            "status": "ok",
            "preferred": preferred,
            "followups": followups,
            "policy_update": policy_update,
            "metrics": metrics_payload(conn),
        })
    finally:
        conn.close()


@app.route("/api/prefer_reason", methods=["POST"])
def api_prefer_reason():
    """Record why the user preferred an answer, and learn a lesson from it.

    Returns the plain-English "Learning Card" text plus refreshed metrics so the
    Lessons feed and Training Progress update immediately.
    """
    data = request.get_json(silent=True) or {}
    comparison_id = data.get("comparison_id")
    reason = (data.get("reason") or "").strip()
    if comparison_id is None or not reason:
        return json_error("Missing comparison_id or reason.", 400, "invalid_request")

    conn = db()
    try:
        comp = sa.get_comparison(conn, int(comparison_id))
        if not comp:
            return json_error("Comparison not found.", 404, "not_found_error")
        if comp["preferred"] not in ("A", "B"):
            return json_error("Record a preference before a reason.", 400, "invalid_request")

        sa.set_reason(conn, int(comparison_id), reason)

        # Reward bonus for explaining the preference (more accurate +0.3, clearer
        # +0.2, better sources +0.1) — credited to the winning answer's style.
        bonus = sa.REASON_BONUS.get(reason.lower())
        if bonus:
            routing = json.loads(comp["routing"]) if comp["routing"] else None
            b_model, b_qtype = sa.routing_facets(routing)
            sa.record_reward(
                conn, signal="reason_bonus", score=bonus,
                query_group=comp["query_group"], session_id=comp["session_id"],
                model=b_model, style=sa.STYLE_OF_SIDE.get(comp["preferred"]), query_type=b_qtype,
            )

        topic = sa.extract_topic(comp["original_query"])
        lesson = sa.record_lesson(
            conn, topic=topic, winning_style=comp["preferred"], reason=reason,
            session_id=comp["session_id"],
        )
        return jsonify(
            {
                "status": "ok",
                "learning_card": lesson["text"],
                "lesson": lesson,
                "metrics": metrics_payload(conn),
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

        # Reward signal for a thumbs vote on a single (non-A/B) answer: 👍 +1.0,
        # 👎 -1.0. Style is unknown for a single answer, so it's left NULL; model
        # and query type come from the interaction's stored routing.
        fb_routing = sa.get_routing(conn, int(interaction_id))
        fb_model, fb_qtype = sa.routing_facets(fb_routing)
        sa.record_reward(
            conn, signal="thumbs_up" if feedback == "up" else "thumbs_down",
            score=sa.REWARD_SCORES["thumbs_up" if feedback == "up" else "thumbs_down"],
            interaction_id=int(interaction_id), query_group=int(group_id),
            session_id=session_id(), model=fb_model, query_type=fb_qtype,
        )
        policy_update = sa.maybe_update_policy(conn)

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
                # Retries reuse prior sources and improve the answer with the
                # capable model (Sonnet) — see search_agent.regenerate_answer.
                retry_routing = sa.build_routing("complex")
                new_id = sa.insert_answer(
                    conn,
                    query_group=int(group_id),
                    original_query=original_query,
                    effective_query=new_query,
                    answer=new_answer,
                    attempt=attempts + 1,
                    sources=reused_sources,
                    routing=retry_routing,
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
                    "routing": retry_routing,
                }

        return jsonify(
            {
                "status": "ok",
                "learning": learning,
                "retry": retry,
                "policy_update": policy_update,
                "metrics": metrics_payload(conn),
            }
        )
    finally:
        conn.close()


@app.route("/api/migrate", methods=["GET", "POST"])
def api_migrate():
    """One-time DB restore: import backup.sql into the live database.

    Intended to be hit once after deploy to seed the Railway volume from the
    committed backup.sql. Guarded by a token so it can't be triggered casually:
    set MIGRATE_TOKEN in the environment and pass it as ?token= (or JSON
    {"token": ...}). If MIGRATE_TOKEN is unset the endpoint is disabled.
    """
    expected = os.environ.get("MIGRATE_TOKEN")
    if not expected:
        return json_error(
            "Migration endpoint is disabled — set MIGRATE_TOKEN to enable it.",
            403, "forbidden",
        )
    data = request.get_json(silent=True) or {}
    supplied = request.args.get("token") or data.get("token")
    if supplied != expected:
        return json_error("Invalid or missing migration token.", 401, "authentication_error")

    backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup.sql")
    if not os.path.exists(backup_path):
        return json_error(f"backup.sql not found at {backup_path}.", 404, "not_found_error")

    with open(backup_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    conn = db()
    try:
        counts = sa.restore_from_sql(conn, sql_text)
        return jsonify({
            "status": "ok",
            "db_path": sa.DB_PATH,
            "imported": counts,
            "metrics": metrics_payload(conn),
        })
    finally:
        conn.close()


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear ALL data (global) — interactions, comparisons, lessons, messages.

    The dashboard is a global, all-session view, so the demo Reset wipes
    everything (including every session's conversation history).
    """
    conn = db()
    try:
        # Clears every data table (incl. rewards + policy_updates) for a clean demo.
        for table in sa.DATA_TABLES:
            conn.execute(f"DELETE FROM {table}")
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
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=debug)
