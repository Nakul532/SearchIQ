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
import secrets
import sqlite3
from datetime import timedelta
from functools import wraps

import anthropic
from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   session, stream_with_context, url_for)
from werkzeug.exceptions import HTTPException

import search_agent as sa

# Strip stray non-ASCII chars (e.g. a pasted U+2028) from the credentials before
# building the client — HTTP headers must be ASCII or the SDK raises an
# 'ascii' codec encode error. See search_agent.sanitize_env_secret.
_API_KEY = sa.sanitize_env_secret("ANTHROPIC_API_KEY")
sa.sanitize_env_secret("ANTHROPIC_AUTH_TOKEN")

app = Flask(__name__)
client = anthropic.Anthropic(api_key=_API_KEY) if _API_KEY else anthropic.Anthropic()

# Honor the platform's proxy headers (Railway/most PaaS terminate TLS and forward
# over http with X-Forwarded-Proto: https). Without this, url_for(_external=True)
# would build an http:// redirect_uri even on an https site — Google rejects http
# redirect URIs for non-localhost, so OAuth would fail. With it, the redirect_uri
# is correctly https on Railway, which is exactly why no insecure-transport flag
# (OAUTHLIB_INSECURE_TRANSPORT) is needed there.
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# --------------------------------------------------------------------------- #
# Auth: Google sign-in (Authlib) + a 2-week signed-cookie session
# --------------------------------------------------------------------------- #
# The session cookie carries the logged-in user; it must be signed. Set
# FLASK_SECRET_KEY in production — a per-boot random key is used otherwise (which
# silently logs everyone out on restart), with a warning.
_SECRET = os.environ.get("FLASK_SECRET_KEY")
if not _SECRET:
    _SECRET = secrets.token_hex(32)
    print("Warning: FLASK_SECRET_KEY is not set — using a random key (sessions reset on "
          "restart). Set FLASK_SECRET_KEY for persistent 2-week logins.")
app.secret_key = _SECRET
app.permanent_session_lifetime = timedelta(days=14)  # 2-week persistent login

_GOOGLE_CLIENT_ID = (os.environ.get("GOOGLE_CLIENT_ID") or "").strip()
_GOOGLE_CLIENT_SECRET = (os.environ.get("GOOGLE_CLIENT_SECRET") or "").strip()
OAUTH_CONFIGURED = bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET)

oauth = None
google = None
if OAUTH_CONFIGURED:
    from authlib.integrations.flask_client import OAuth
    oauth = OAuth(app)
    google = oauth.register(
        name="google",
        client_id=_GOOGLE_CLIENT_ID,
        client_secret=_GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
else:
    print("Warning: GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set — Google sign-in is "
          "disabled. Set both to enable login.")


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


def current_user() -> dict | None:
    """The signed-in user dict ({google_id,name,email,picture}) from the session."""
    return session.get("user")


def session_id() -> str | None:
    """Identity for all per-user data: the authenticated user's google_id (or None)."""
    user = current_user()
    return user.get("google_id") if user else None


def require_login(fn):
    """API guard: return 401 JSON when the request isn't authenticated."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session_id():
            return json_error("Login required.", 401, "auth_required")
        return fn(*args, **kwargs)
    return wrapper


def metrics_payload(conn: sqlite3.Connection, sid: str | None) -> dict:
    """Build the dashboard payload scoped to one user (all their data, all time)."""
    m = sa.compute_metrics(conn, sid)
    if sid is None:
        rows = conn.execute(
            "SELECT feedback FROM interactions WHERE feedback IS NOT NULL ORDER BY id DESC LIMIT 20"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT feedback FROM interactions WHERE feedback IS NOT NULL AND session_id = ? "
            "ORDER BY id DESC LIMIT 20",
            (sid,),
        ).fetchall()
    recent = [r[0] for r in rows][::-1]  # oldest → newest
    m["recent_feedback"] = recent
    m["thumbs_up"] = recent.count("up")
    m["thumbs_down"] = recent.count("down")
    m["improved_count"] = len(m["most_improved"])
    # Visible learning loop: the user's lessons feed + training-progress totals.
    m["lessons_feed"] = sa.all_lessons(conn, session_id=sid)
    m["training_progress"] = sa.training_progress(conn, sid)
    # Reinforcement-learning reward signals + the user's current learned policy.
    m["reward"] = sa.reward_stats(conn, sid)
    m["policy"] = sa.current_policy(conn, sid)
    return m


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.route("/login")
def login():
    if session_id():
        return redirect(url_for("index"))
    return render_template("login.html", oauth_configured=OAUTH_CONFIGURED)


@app.route("/auth/login")
def auth_login():
    if not OAUTH_CONFIGURED:
        return redirect(url_for("login"))
    redirect_uri = url_for("auth_callback", _external=True)
    # Log the EXACT redirect URI sent to Google — it must match an "Authorized
    # redirect URI" in your Google OAuth client byte-for-byte. (print → stdout so
    # it always shows in Railway/gunicorn logs regardless of log level.)
    print(f"[oauth] redirect_uri sent to Google: {redirect_uri}", flush=True)
    app.logger.info("OAuth redirect_uri: %s", redirect_uri)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def auth_callback():
    if not OAUTH_CONFIGURED:
        return redirect(url_for("login"))
    try:
        token = google.authorize_access_token()
    except Exception:  # noqa: BLE001 — denied consent / bad state / network
        app.logger.exception("OAuth callback failed")
        return redirect(url_for("login"))
    info = token.get("userinfo") or {}
    google_id = info.get("sub")
    if not google_id:
        return redirect(url_for("login"))
    name = info.get("name") or info.get("given_name") or info.get("email") or "User"
    email = info.get("email")
    picture = info.get("picture")
    conn = db()
    try:
        sa.upsert_user(conn, google_id=google_id, name=name, email=email, picture=picture)
    finally:
        conn.close()
    session["user"] = {"google_id": google_id, "name": name, "email": email, "picture": picture}
    session.permanent = True  # honor the 2-week lifetime
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/me")
@require_login
def api_me():
    """The signed-in user + their personality profile (for the header + sidebar card)."""
    user = current_user()
    conn = db()
    try:
        profile = sa.get_user_profile(conn, user["google_id"])
    finally:
        conn.close()
    return jsonify({
        "user": {"name": user.get("name"), "email": user.get("email"),
                 "picture": user.get("picture")},
        "profile": profile,
    })


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    if not session_id():
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/api/metrics")
@require_login
def api_metrics():
    sid = session_id()
    conn = db()
    try:
        return jsonify(metrics_payload(conn, sid))
    finally:
        conn.close()


@app.route("/api/conversations")
@require_login
def api_conversations():
    """List this user's past conversations for the history sidebar."""
    sid = session_id()
    conn = db()
    try:
        return jsonify({"conversations": sa.list_conversations(conn, sid)})
    finally:
        conn.close()


@app.route("/api/conversation/<conversation_id>")
@require_login
def api_conversation(conversation_id: str):
    """Load the full transcript of one past conversation (scoped to the user)."""
    sid = session_id()
    conn = db()
    try:
        return jsonify({
            "conversation_id": conversation_id,
            "messages": sa.load_conversation(conn, sid, conversation_id),
        })
    finally:
        conn.close()


@app.route("/api/query", methods=["POST"])
@require_login
def api_query():
    """Stream an A/B answer pair as newline-delimited JSON (NDJSON).

    Each draft is delivered the instant it finishes, so the fast one (Haiku on
    complex queries) reaches the UI in ~10s while the thorough one (Sonnet) is
    still generating. Event types: clarify | meta | answer | comparison | single |
    error (see static/app.js for the consumer).
    """
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

    def emit(obj) -> str:
        return json.dumps(obj) + "\n"

    def generate():
        conn = db()
        try:
            # Pre-processor step 1: if the query is ambiguous, ask ONE natural
            # question and wait for the user's reply. The original query and the
            # question are saved to the transcript, so when the user answers in
            # their own words the next turn has the full context (no chips/buttons).
            if not force:
                question = sa.clarifying_question(client, query)
                if question:
                    sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                    role="user", content=query)
                    sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                    role="assistant", content=question)
                    yield emit({"type": "clarify", "question": question,
                                "conversation_id": conversation_id})
                    return

            # Reward the PRIOR answer when this is a follow-up — the user was engaged
            # enough to keep going (+0.2). Recorded before the new turn, so it lands
            # on the answer that prompted the continuation.
            if followup:
                prev = sa.last_interaction(conn, sid)
                if prev:
                    p_model, p_qtype = sa.routing_facets(prev["routing"])
                    sa.record_reward(
                        conn, signal="followup", score=sa.REWARD_SCORES["followup"],
                        interaction_id=prev["id"], query_group=prev["query_group"],
                        session_id=sid, model=p_model, query_type=p_qtype,
                    )

            # Prior turns for context (the user turn is saved only after ≥1 answer
            # comes back, so a total API failure leaves no dangling turn).
            history = sa.conversation_history(conn, conversation_id, limit=10)
            group = sa.next_query_group(conn)
            applied = sa.match_lessons(conn, query, session_id=sid)
            applied_ids = [l["id"] for l in applied]
            # Classify → policy-bias the base model; split per side (Haiku A / Sonnet
            # B on complex). Steer both drafts toward the user's policy + personality
            # profile via the system prompt (prepended to the matched lessons).
            routing = sa.apply_policy(conn, sa.classify_query(query), session_id=sid)
            routing_a, routing_b = sa.ab_routings(routing)
            directive = sa.policy_directive(conn, sid)
            profile = sa.get_user_profile(conn, sid)
            profile_note = sa.profile_directive(profile)
            personalized = bool(profile_note)
            lesson_texts = (
                ([profile_note] if profile_note else [])
                + ([directive] if directive else [])
                + [l["text"] for l in applied]
            )

            yield emit({
                "type": "meta",
                "group_id": group,
                "conversation_id": conversation_id,
                "original_query": query,
                "effective_query": query,
                "routing": routing,
                "routing_a": routing_a,
                "routing_b": routing_b,
                "applied_lessons": applied,
                "applied_count": len(applied),
                "personalized": personalized,
            })

            # Generate A and B in parallel; stream each as soon as it's ready.
            results: dict[str, dict] = {}
            for side, res in sa.iter_ab_answers(
                client, query, lesson_texts, {"A": routing_a, "B": routing_b}, history=history
            ):
                if "answer" in res:
                    results[side] = res
                    yield emit({"type": "answer", "side": side,
                                "answer": res["answer"], "sources": res["sources"]})

            if not results:
                yield emit({"type": "error",
                            "error": "Both answers failed to generate — please try again.",
                            "status": 502})
                return

            # ≥1 answer succeeded — record the user's turn now.
            sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                            role="user", content=query)
            if applied_ids:
                sa.bump_applied_count(conn, applied_ids)

            if "A" in results and "B" in results:
                comparison_id = sa.record_comparison(
                    conn, query_group=group, original_query=query, effective_query=query,
                    answers={"a": results["A"], "b": results["B"]},
                    routing=routing, lessons_applied=applied_ids,
                    session_id=sid, conversation_id=conversation_id,
                    routing_a=routing_a, routing_b=routing_b,
                )
                yield emit({"type": "comparison", "comparison_id": comparison_id})
            else:
                # Exactly one side survived — commit it as a single answer (no pick).
                side = "A" if "A" in results else "B"
                single = results[side]
                single_routing = routing_a if side == "A" else routing_b
                sa.record_interaction(
                    conn, query_group=group, original_query=query, effective_query=query,
                    answer=single["answer"], feedback=None, comment=None, attempt=1,
                    sources=single["sources"], routing=single_routing, session_id=sid,
                )
                sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                role="assistant", content=single["answer"], sources=single["sources"])
                sa.maybe_update_profile(conn, sid)  # rebuild the profile every 5 queries
                followups = sa.suggest_followups(client, query, single["answer"])
                yield emit({
                    "type": "single",
                    "side": side,
                    "group_id": group,
                    "conversation_id": conversation_id,
                    "routing": single_routing,
                    "answer": single["answer"],
                    "sources": single["sources"],
                    "followups": followups,
                })
        except Exception as exc:  # noqa: BLE001 — stream the failure, headers are sent
            app.logger.exception("Streaming query failed")
            yield emit({"type": "error", "error": str(exc) or "Internal server error", "status": 500})
        finally:
            conn.close()

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson", headers=headers)


@app.route("/api/prefer", methods=["POST"])
@require_login
def api_prefer():
    """Record which A/B answer the user preferred.

    Winner → a positive interaction; loser → a negative one (so the North Star
    completion-rate and feedback history stay intact). The lesson is created
    later, once the user gives a reason (see /api/prefer_reason).
    """
    sid = session_id()
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

        def _json(s):
            try:
                return json.loads(s) if s else None
            except (json.JSONDecodeError, TypeError):
                return None

        loser_side = "B" if preferred == "A" else "A"
        win_answer = comp["answer_a"] if preferred == "A" else comp["answer_b"]
        lose_answer = comp["answer_b"] if preferred == "A" else comp["answer_a"]
        win_sources = comp["sources_a"] if preferred == "A" else comp["sources_b"]
        lose_sources = comp["sources_b"] if preferred == "A" else comp["sources_a"]
        # Per-side routing (A and B can be different models on complex queries) so
        # rewards attribute the winner to the model that actually produced it.
        routing_a = _json(comp.get("routing_a")) or _json(comp["routing"])
        routing_b = _json(comp.get("routing_b")) or _json(comp["routing"])
        win_routing = routing_a if preferred == "A" else routing_b
        lose_routing = routing_b if preferred == "A" else routing_a

        def _src(s):
            return _json(s) or []

        sa.set_preference(conn, int(comparison_id), preferred)
        # Winner = thumbs up (attempt 1), loser = thumbs down (attempt 2).
        win_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=win_answer, feedback="up",
            comment=None, attempt=1, sources=_src(win_sources), routing=win_routing,
            session_id=comp["session_id"],
        )
        lose_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=lose_answer, feedback="down",
            comment=None, attempt=2, sources=_src(lose_sources), routing=lose_routing,
            session_id=comp["session_id"],
        )

        # Reward signals: winner +1.0, loser -0.5 (A = concise, B = detailed).
        win_model, win_qtype = sa.routing_facets(win_routing)
        lose_model, _ = sa.routing_facets(lose_routing)
        sa.record_reward(
            conn, signal="ab_winner", score=sa.REWARD_SCORES["ab_winner"],
            interaction_id=win_id, query_group=comp["query_group"], session_id=comp["session_id"],
            model=win_model, style=sa.STYLE_OF_SIDE[preferred], query_type=win_qtype,
        )
        sa.record_reward(
            conn, signal="ab_loser", score=sa.REWARD_SCORES["ab_loser"],
            interaction_id=lose_id, query_group=comp["query_group"], session_id=comp["session_id"],
            model=lose_model, style=sa.STYLE_OF_SIDE[loser_side], query_type=win_qtype,
        )
        # Re-evaluate this user's policy + personality profile on their cadence.
        policy_update = sa.maybe_update_policy(conn, comp["session_id"])
        sa.maybe_update_profile(conn, comp["session_id"])

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
            "metrics": metrics_payload(conn, sid),
        })
    finally:
        conn.close()


@app.route("/api/prefer_reason", methods=["POST"])
@require_login
def api_prefer_reason():
    """Record why the user preferred an answer, and learn a lesson from it.

    Returns the plain-English "Learning Card" text plus refreshed metrics so the
    Lessons feed and Training Progress update immediately.
    """
    sid = session_id()
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
            # Attribute the bonus to the winning side's model (Haiku A / Sonnet B
            # can differ on complex queries).
            win_routing_key = "routing_a" if comp["preferred"] == "A" else "routing_b"
            win_routing_raw = comp.get(win_routing_key) or comp["routing"]
            try:
                routing = json.loads(win_routing_raw) if win_routing_raw else None
            except (json.JSONDecodeError, TypeError):
                routing = None
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
                "metrics": metrics_payload(conn, sid),
            }
        )
    finally:
        conn.close()


@app.route("/api/feedback", methods=["POST"])
@require_login
def api_feedback():
    sid = session_id()
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
            session_id=sid, model=fb_model, query_type=fb_qtype,
        )
        policy_update = sa.maybe_update_policy(conn, sid)
        sa.maybe_update_profile(conn, sid)

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
                "metrics": metrics_payload(conn, sid),
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
            "metrics": metrics_payload(conn, None),
        })
    finally:
        conn.close()


# Tables the restore endpoint may write to (every app table, incl. accounts).
_RESTORE_TABLES = set(sa.DATA_TABLES) | {"users"}


def _coerce_value(v):
    """Bind-safe value: serialize nested JSON (dict/list) to a string for TEXT cols."""
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


@app.route("/api/restore", methods=["POST"])
def api_restore():
    """Restore data from JSON instead of SQL: {"table_name": [ {col: val, ...}, ... ]}.

    Inserts the given rows directly into their tables (INSERT OR REPLACE, so it's
    idempotent and re-runnable by primary key). A JSON alternative to /api/migrate.
    Guarded by the same MIGRATE_TOKEN — pass it as ?token=, an X-Migrate-Token
    header, or a "token" field in the JSON body. Only known app tables are allowed,
    and only columns that actually exist on each table are written (so table/column
    names can't be used for injection).
    """
    expected = os.environ.get("MIGRATE_TOKEN")
    if not expected:
        return json_error("Restore endpoint is disabled — set MIGRATE_TOKEN to enable it.",
                          403, "forbidden")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error('Body must be a JSON object: {"table_name": [rows]}.',
                          400, "invalid_request")
    # Token from query / header / body; pop it so it isn't treated as a table.
    supplied = (request.args.get("token") or request.headers.get("X-Migrate-Token")
                or data.pop("token", None))
    if supplied != expected:
        return json_error("Invalid or missing restore token.", 401, "authentication_error")

    conn = db()
    try:
        restored: dict[str, int] = {}
        for table, rows in data.items():
            if table not in _RESTORE_TABLES:
                return json_error(f"Unknown or disallowed table: {table!r}.", 400, "invalid_request")
            if not isinstance(rows, list):
                return json_error(f"Rows for {table!r} must be a list.", 400, "invalid_request")
            valid_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            inserted = 0
            for row in rows:
                if not isinstance(row, dict):
                    return json_error(f"Each row in {table!r} must be a JSON object.", 400, "invalid_request")
                pairs = [(k, v) for k, v in row.items() if k in valid_cols]
                if not pairs:
                    continue  # nothing usable in this row — skip
                col_sql = ", ".join(k for k, _ in pairs)        # validated against PRAGMA
                placeholders = ", ".join("?" for _ in pairs)
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})",
                    [_coerce_value(v) for _, v in pairs],
                )
                inserted += 1
            restored[table] = inserted
        conn.commit()
        return jsonify({"status": "ok", "db_path": sa.DB_PATH, "restored": restored})
    finally:
        conn.close()


@app.route("/api/reset", methods=["POST"])
@require_login
def api_reset():
    """Clear the SIGNED-IN USER's data — their conversations, feedback, lessons,
    rewards, policy, and profile. Other users' data and accounts are untouched.
    """
    sid = session_id()
    conn = db()
    try:
        # Every data table carries the owner's id (user_profiles via google_id).
        for table in sa.DATA_TABLES:
            owner_col = "google_id" if table == "user_profiles" else "session_id"
            conn.execute(f"DELETE FROM {table} WHERE {owner_col} = ?", (sid,))
        conn.commit()
        return jsonify({"status": "ok", "metrics": metrics_payload(conn, sid)})
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
