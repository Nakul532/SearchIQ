"""SearchIQ web app (Flask) — runs as a single Vercel serverless function.

Reuses the search_agent.py backend (Groq answers grounded in Tavily search,
A/B comparison, rephrase / retry, rewards, metrics) and exposes it over a small
JSON API consumed by the single-page UI.

Anyone can search as a guest (identified by a signed cookie); Google sign-in is
optional and adds personalization. Free-tier guard rails (daily caps, token
budgets, the answer cache) live in limits.py.

Local run:
    cp .env.example .env   # fill in keys, or set SEARCHIQ_FAKE_PROVIDERS=1
    python app.py          # http://127.0.0.1:5001
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from datetime import timedelta

from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   session, stream_with_context, url_for)
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

import db as dbmod
import limits
import providers
import agent
import search_agent as sa

sa.sanitize_env_secret("GROQ_API_KEY")
sa.sanitize_env_secret("TAVILY_API_KEY")

ON_VERCEL = bool(os.environ.get("VERCEL"))
_HERE = os.path.dirname(os.path.abspath(__file__))

# Static assets live in public/ so Vercel serves them from its CDN; Flask serves
# the same folder locally.
app = Flask(__name__, static_folder=os.path.join(_HERE, "public", "static"),
            static_url_path="/static", template_folder=os.path.join(_HERE, "templates"),
            instance_path=os.path.join(_HERE, "instance"))

# Vercel terminates TLS and forwards with X-Forwarded-Proto/Host — honor them so
# url_for(_external=True) builds an https:// OAuth redirect URI.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# --------------------------------------------------------------------------- #
# Sessions: signed cookie carrying either a Google user or a guest id
# --------------------------------------------------------------------------- #
_SECRET = (os.environ.get("FLASK_SECRET_KEY") or "").strip()
if not _SECRET:
    if ON_VERCEL:
        # Without a fixed key every cold start would mint new guest ids (resetting
        # guest caps) and log everyone out — refuse to run misconfigured.
        raise RuntimeError("FLASK_SECRET_KEY must be set in the Vercel environment.")
    _SECRET = secrets.token_hex(32)
    print("Warning: FLASK_SECRET_KEY is not set — using a random key (sessions reset on restart).")
app.secret_key = _SECRET
app.permanent_session_lifetime = timedelta(days=14)
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                  SESSION_COOKIE_SECURE=ON_VERCEL)

_GOOGLE_CLIENT_ID = (os.environ.get("GOOGLE_CLIENT_ID") or "").strip()
_GOOGLE_CLIENT_SECRET = (os.environ.get("GOOGLE_CLIENT_SECRET") or "").strip()
OAUTH_CONFIGURED = bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET)

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


# --------------------------------------------------------------------------- #
# DB + guard-rail wiring
# --------------------------------------------------------------------------- #
def db():
    conn = dbmod.connect()
    sa.init_db(conn)
    return conn


# Every Groq call counts its tokens against today's per-model budget.
limits.install_token_budget(db)


def current_user() -> dict | None:
    return session.get("user")


def signed_in() -> bool:
    return bool(current_user())


def session_id() -> str:
    """Identity for all per-user data: the Google id, or a cookie-held guest id."""
    user = current_user()
    if user:
        return user["google_id"]
    gid = session.get("guest_id")
    if not gid:
        gid = session["guest_id"] = "guest-" + uuid.uuid4().hex
        session.permanent = True
    return gid


def client_ip_hash() -> str:
    # On Vercel, x-real-ip is set by the platform (client-supplied values are
    # overwritten). Locally, fall back to the socket address.
    ip = request.headers.get("x-real-ip") if ON_VERCEL else None
    return limits.hash_ip(ip or request.remote_addr)


def profile_sid() -> str | None:
    """Personalization is for signed-in users only."""
    return session_id() if signed_in() else None


def quota(conn) -> dict:
    return limits.remaining(conn, session_id(), signed_in(), client_ip_hash())


# --------------------------------------------------------------------------- #
# JSON error handling — every failure returns JSON (never an HTML error page)
# --------------------------------------------------------------------------- #
def json_error(message: str, status: int = 500, kind: str = "error"):
    return jsonify({"error": message, "type": kind}), status


@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    return jsonify({"error": e.description, "type": "http_error", "status": e.code}), e.code


@app.errorhandler(Exception)
def handle_unexpected(e: Exception):
    if isinstance(e, providers.ProviderLimit):
        conn = db()
        try:
            return jsonify({"limit": limits.limit_info("site", limits.cached_examples(conn))}), 429
        finally:
            conn.close()
    if isinstance(e, providers.ProviderError):
        return json_error("The AI provider is unavailable right now — please try again.", 502,
                          "provider_error")
    app.logger.exception("Unhandled error")
    return json_error("Internal server error", 500, "server_error")


def metrics_payload(conn, sid: str) -> dict:
    """Dashboard payload scoped to one user/guest."""
    m = sa.compute_metrics(conn, sid)
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
    m["lessons_feed"] = sa.all_lessons(conn, session_id=sid)
    m["training_progress"] = sa.training_progress(conn, sid)
    m["reward"] = sa.reward_stats(conn, sid)
    m["policy"] = sa.current_policy(conn, sid)
    return m


# --------------------------------------------------------------------------- #
# Auth routes (optional Google sign-in)
# --------------------------------------------------------------------------- #
@app.route("/login")
def login():
    if signed_in():
        return redirect(url_for("index"))
    return render_template("login.html", oauth_configured=OAUTH_CONFIGURED)


@app.route("/auth/login")
def auth_login():
    if not OAUTH_CONFIGURED:
        return redirect(url_for("login"))
    redirect_uri = url_for("auth_callback", _external=True)
    print(f"[oauth] redirect_uri sent to Google: {redirect_uri}", flush=True)
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
    session.permanent = True
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("user", None)  # back to guest mode (the guest id + its caps remain)
    return redirect(url_for("index"))


@app.route("/api/me")
def api_me():
    """Who's here (user or guest), their profile, and today's search quota."""
    sid = session_id()
    user = current_user()
    conn = db()
    try:
        profile = sa.get_user_profile(conn, sid) if user else None
        q = quota(conn)
    finally:
        conn.close()
    return jsonify({
        "user": ({"name": user.get("name"), "email": user.get("email"),
                  "picture": user.get("picture")} if user else None),
        "guest": not user,
        "oauth_configured": OAUTH_CONFIGURED,
        "profile": profile,
        "quota": q,
        "demo_url": limits.DEMO_VIDEO_URL or None,
        "models": {"fast": sa.FAST_LABEL, "deep": sa.DEEP_LABEL},
    })


# --------------------------------------------------------------------------- #
# App routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    session_id()  # mint the guest id up front so the cookie is set
    return render_template("index.html")


@app.route("/track-record")
def track_record():
    """Public eval results page (data: public/static/track_record.json)."""
    return render_template("track_record.html")


@app.route("/api/metrics")
def api_metrics():
    conn = db()
    try:
        return jsonify(metrics_payload(conn, session_id()))
    finally:
        conn.close()


@app.route("/api/conversations")
def api_conversations():
    conn = db()
    try:
        return jsonify({"conversations": sa.list_conversations(conn, session_id())})
    finally:
        conn.close()


@app.route("/api/conversation/<conversation_id>")
def api_conversation(conversation_id: str):
    conn = db()
    try:
        return jsonify({
            "conversation_id": conversation_id,
            "messages": sa.load_conversation(conn, session_id(), conversation_id),
        })
    finally:
        conn.close()


def _lesson_context(conn, sid: str, query: str):
    """Lessons + policy + (signed-in only) personality profile → system-prompt notes."""
    applied = sa.match_lessons(conn, query, session_id=sid)
    directive = sa.policy_directive(conn, sid)
    profile = sa.get_user_profile(conn, profile_sid()) if signed_in() else None
    profile_note = sa.profile_directive(profile)
    texts = (([profile_note] if profile_note else []) + ([directive] if directive else [])
             + [l["text"] for l in applied])
    return applied, texts, bool(profile_note)


@app.route("/api/query", methods=["POST"])
def api_query():
    """Stream an A/B answer pair as newline-delimited JSON (NDJSON).

    Event types: clarify | meta | step | answer | check | comparison | single |
    limit | quota | error (see public/static/app.js for the consumer). Order:
      1. first turn → serve from cache if we have it (free, no cap)
      2. ambiguous → ask one clarifying question (no search)
      3. over a cap → limit card (with demo link + cached examples)
      4. live: one Tavily search → Groq drafts A and B in parallel → the v2
         agent checks both drafts against the sources (agent.py)
    `step` events drive the steps panel: {step: search|draft|check|cache,
    status: active|done|skipped, detail}.
    """
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()[:1000]
    force = bool(data.get("force"))
    followup = bool(data.get("followup"))
    conversation_id = str(data.get("conversation_id", "")).strip()[:64]
    sid = session_id()
    is_signed_in = signed_in()
    ip_hash = client_ip_hash()
    if not query:
        return json_error("Query is empty.", 400, "invalid_request")
    if not conversation_id:
        return json_error("Missing conversation id.", 400, "invalid_request")

    def emit(obj) -> str:
        return json.dumps(obj) + "\n"

    def emit_quota(conn) -> str:
        return emit({"type": "quota", **limits.remaining(conn, sid, is_signed_in, ip_hash)})

    def generate():
        conn = db()
        try:
            history = sa.conversation_history(conn, conversation_id, limit=10, session_id=sid)
            first_turn = not history

            # Reward the PRIOR answer when this is a follow-up (+0.2).
            if followup:
                prev = sa.last_interaction(conn, sid)
                if prev:
                    p_model, p_qtype = sa.routing_facets(prev["routing"])
                    sa.record_reward(
                        conn, signal="followup", score=sa.REWARD_SCORES["followup"],
                        interaction_id=prev["id"], query_group=prev["query_group"],
                        session_id=sid, model=p_model, query_type=p_qtype,
                    )

            # 1. Cache — first turn of a conversation only (later turns depend on
            # context), except the example questions, which are always served cached.
            use_cache = first_turn or limits.is_example(query)
            cached = limits.cache_get(conn, query) if use_cache and not force else None
            # Example answers cached before the v2 agent have no check: regenerate
            # them once so the showcase questions show the claim check.
            if cached and cached.get("kind") == "ab" and limits.is_example(query) and "check" not in cached:
                cached = None
            if cached and cached.get("kind") == "clarify" and not force:
                sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                role="user", content=query)
                sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                role="assistant", content=cached["question"])
                yield emit({"type": "clarify", "question": cached["question"],
                            "conversation_id": conversation_id, "cached": True,
                            "cached_at": cached["cached_at"]})
                return
            if cached and cached.get("kind") == "ab":
                yield from _serve_cached(conn, cached, query, sid, conversation_id, emit)
                yield emit_quota(conn)
                return

            # 2. Clarifying question for ambiguous queries (no search spent).
            if not force:
                question = sa.clarifying_question(query)
                if question:
                    sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                    role="user", content=query)
                    sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                    role="assistant", content=question)
                    if first_turn and limits.is_example(query):
                        limits.cache_put(conn, query, {"kind": "clarify", "question": question})
                    yield emit({"type": "clarify", "question": question,
                                "conversation_id": conversation_id})
                    return

            # 3. Caps.
            blocked = limits.check_live_search(conn, sid, is_signed_in, ip_hash)
            if blocked:
                blocked["examples"] = limits.cached_examples(conn)
                yield emit(blocked)
                return

            # 4. Live search + answers.
            group = sa.next_query_group(conn)
            applied, lesson_texts, personalized = _lesson_context(conn, sid, query)
            applied_ids = [l["id"] for l in applied]
            routing = sa.apply_policy(conn, sa.classify_query(query), session_id=sid)
            routing_a, routing_b = sa.ab_routings(routing)

            # Effective search query: include the previous user turn for short follow-ups.
            search_q = query
            if history:
                prev_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
                if prev_user and len(query.split()) < 12:
                    search_q = f"{prev_user} — {query}"
            yield emit({
                "type": "meta", "group_id": group, "conversation_id": conversation_id,
                "original_query": query, "effective_query": query,
                "routing": routing, "routing_a": routing_a, "routing_b": routing_b,
                "applied_lessons": applied, "applied_count": len(applied),
                "personalized": personalized, "cached": False,
            })
            yield emit({"type": "step", "step": "search", "status": "active",
                        "detail": "Searching the web"})
            try:
                results = providers.search(search_q, max_results=sa.MAX_SOURCES)
            except providers.ProviderLimit:
                yield emit(limits.limit_info("site", limits.cached_examples(conn)))
                return
            limits.record_search_spend(conn)
            hosts = list(dict.fromkeys(h for h in (_host(r.get("url")) for r in results) if h))
            yield emit({"type": "step", "step": "search", "status": "done",
                        "detail": f"{len(results)} source{'s' if len(results) != 1 else ''}"
                                  + (f": {', '.join(hosts)}" if hosts else "")})
            yield emit({"type": "step", "step": "draft", "status": "active",
                        "detail": "Writing answers A and B from the same sources"})

            results_by_side: dict[str, dict] = {}
            limited = False
            for side, res in sa.iter_ab_answers(query, lesson_texts,
                                                {"A": routing_a, "B": routing_b},
                                                history=history, results=results):
                if "answer" in res:
                    results_by_side[side] = res
                    yield emit({"type": "answer", "side": side,
                                "answer": res["answer"], "sources": res["sources"]})
                elif res.get("limit"):
                    limited = True

            yield emit({"type": "step", "step": "draft", "status": "done" if results_by_side else "skipped",
                        "detail": f"{len(results_by_side)} of 2 drafts"})
            check = None
            if results_by_side:
                yield emit({"type": "step", "step": "check", "status": "active",
                            "detail": "Checking each claim against the sources"})
                check = agent.check_answers(query, results,
                                            {s: r["answer"] for s, r in results_by_side.items()},
                                            n_sources=sa.MAX_SOURCES)
                if check:
                    yield emit({"type": "check", **check})
                yield emit({"type": "step", "step": "check", "status": "done" if check else "skipped",
                            "detail": _check_summary(check) if check else "Check unavailable this time"})
                # The check acts: rewrite drafts with a false premise or a
                # contradicted claim, then re-check the rewrites.
                if check and any(agent.needs_revision(check, s) for s in results_by_side):
                    yield emit({"type": "step", "step": "revise", "status": "active",
                                "detail": "Rewriting to fix what the check found"})
                    answers_now = {s: r["answer"] for s, r in results_by_side.items()}
                    check, revisions = agent.revise_flagged(query, results, answers_now, check,
                                                            n_sources=sa.MAX_SOURCES)
                    for side, rv in revisions.items():
                        results_by_side[side] = {**results_by_side[side], "answer": rv["answer"],
                                                 "original": rv["original"], "revision_reasons": rv["reasons"]}
                        yield emit({"type": "revise", "side": side, "answer": rv["answer"],
                                    "original": rv["original"], "reasons": rv["reasons"]})
                    if revisions:
                        yield emit({"type": "check", **check})
                    yield emit({"type": "step", "step": "revise", "status": "done" if revisions else "skipped",
                                "detail": (f"Rewrote {' and '.join(sorted(revisions))}" if revisions
                                           else "Rewrite unavailable: showing the warning only")})

            if not results_by_side:
                if limited:
                    yield emit(limits.limit_info("site", limits.cached_examples(conn)))
                else:
                    yield emit({"type": "error",
                                "error": "Both answers failed to generate — please try again.",
                                "status": 502})
                return

            limits.record_caller_search(conn, sid, is_signed_in, ip_hash)
            sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                            role="user", content=query)
            if applied_ids:
                sa.bump_applied_count(conn, applied_ids)

            if "A" in results_by_side and "B" in results_by_side:
                comparison_id = sa.record_comparison(
                    conn, query_group=group, original_query=query, effective_query=query,
                    answers={"a": results_by_side["A"], "b": results_by_side["B"]},
                    routing=routing, lessons_applied=applied_ids,
                    session_id=sid, conversation_id=conversation_id,
                    routing_a=routing_a, routing_b=routing_b,
                )
                if first_turn:
                    limits.cache_put(conn, query, {
                        "kind": "ab", "routing": routing, "routing_a": routing_a,
                        "routing_b": routing_b,
                        "A": results_by_side["A"], "B": results_by_side["B"],
                        "check": check,
                    })
                yield emit({"type": "comparison", "comparison_id": comparison_id})
            else:
                side = "A" if "A" in results_by_side else "B"
                single = results_by_side[side]
                single_routing = routing_a if side == "A" else routing_b
                sa.record_interaction(
                    conn, query_group=group, original_query=query, effective_query=query,
                    answer=single["answer"], feedback=None, comment=None, attempt=1,
                    sources=single["sources"], routing=single_routing, session_id=sid,
                )
                sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                                role="assistant", content=single["answer"],
                                sources=single["sources"])
                sa.maybe_update_profile(conn, profile_sid())
                yield emit({
                    "type": "single", "side": side, "group_id": group,
                    "conversation_id": conversation_id, "routing": single_routing,
                    "answer": single["answer"], "sources": single["sources"],
                    "followups": sa.suggest_followups(query, single["answer"]),
                })
            yield emit_quota(conn)
        except Exception:  # noqa: BLE001 — stream the failure, headers are sent
            app.logger.exception("Streaming query failed")
            yield emit({"type": "error", "error": "Something went wrong — please try again.",
                        "status": 500})
        finally:
            conn.close()

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson", headers=headers)


def _serve_cached(conn, cached: dict, query: str, sid: str, conversation_id: str, emit):
    """Replay a cached A/B pair as a fresh comparison for this user (so picking a
    winner, reasons, rewards and lessons all still work). Costs nothing."""
    group = sa.next_query_group(conn)
    routing, routing_a, routing_b = cached["routing"], cached["routing_a"], cached["routing_b"]
    yield emit({
        "type": "meta", "group_id": group, "conversation_id": conversation_id,
        "original_query": query, "effective_query": query,
        "routing": routing, "routing_a": routing_a, "routing_b": routing_b,
        "applied_lessons": [], "applied_count": 0, "personalized": False,
        "cached": True, "cached_at": cached["cached_at"],
    })
    yield emit({"type": "step", "step": "cache", "status": "done",
                "detail": "Served from the answer cache: no new search"})
    for side in ("A", "B"):
        yield emit({"type": "answer", "side": side, "answer": cached[side]["answer"],
                    "sources": cached[side]["sources"]})
        if cached[side].get("original"):
            yield emit({"type": "revise", "side": side, "answer": cached[side]["answer"],
                        "original": cached[side]["original"],
                        "reasons": cached[side].get("revision_reasons", [])})
    if cached.get("check"):
        yield emit({"type": "check", **cached["check"]})
        yield emit({"type": "step", "step": "check", "status": "done",
                    "detail": _check_summary(cached["check"]) + " (checked when cached)"})
    sa.save_message(conn, session_id=sid, conversation_id=conversation_id,
                    role="user", content=query)
    comparison_id = sa.record_comparison(
        conn, query_group=group, original_query=query, effective_query=query,
        answers={"a": cached["A"], "b": cached["B"]}, routing=routing, lessons_applied=[],
        session_id=sid, conversation_id=conversation_id,
        routing_a=routing_a, routing_b=routing_b,
    )
    yield emit({"type": "comparison", "comparison_id": comparison_id})


def _host(url) -> str:
    from urllib.parse import urlparse
    try:
        h = urlparse(url or "").hostname or ""
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def _check_summary(check: dict) -> str:
    parts = []
    for side, s in sorted(check["sides"].items()):
        c = s["confidence"]
        parts.append(f"{side}: {c['backed']}/{c['total']} claims backed" if c["total"] else f"{side}: no checkable claims")
    if not check["premise"]["ok"]:
        parts.append("premise flagged")
    if check.get("disagreements"):
        n = len(check["disagreements"])
        parts.append(f"{n} disagreement{'s' if n != 1 else ''}")
    return " · ".join(parts)


def _owned_comparison(conn, comparison_id) -> dict | None:
    comp = sa.get_comparison(conn, int(comparison_id))
    if not comp or comp.get("session_id") != session_id():
        return None
    return comp


@app.route("/api/prefer", methods=["POST"])
def api_prefer():
    """Record which A/B answer the user preferred (winner 👍, loser 👎 + rewards)."""
    sid = session_id()
    data = request.get_json(silent=True) or {}
    comparison_id = data.get("comparison_id")
    preferred = data.get("preferred")  # 'A' | 'B'
    if comparison_id is None or preferred not in ("A", "B"):
        return json_error("Missing comparison_id or invalid preference.", 400, "invalid_request")

    conn = db()
    try:
        comp = _owned_comparison(conn, comparison_id)
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
        routing_a = _json(comp.get("routing_a")) or _json(comp["routing"])
        routing_b = _json(comp.get("routing_b")) or _json(comp["routing"])
        win_routing = routing_a if preferred == "A" else routing_b
        lose_routing = routing_b if preferred == "A" else routing_a

        def _src(s):
            return _json(s) or []

        sa.set_preference(conn, int(comparison_id), preferred)
        win_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=win_answer, feedback="up",
            comment=None, attempt=1, sources=_src(win_sources), routing=win_routing,
            session_id=sid,
        )
        lose_id = sa.record_interaction(
            conn, query_group=comp["query_group"], original_query=comp["original_query"],
            effective_query=comp["effective_query"], answer=lose_answer, feedback="down",
            comment=None, attempt=2, sources=_src(lose_sources), routing=lose_routing,
            session_id=sid,
        )

        win_model, win_qtype = sa.routing_facets(win_routing)
        lose_model, _ = sa.routing_facets(lose_routing)
        sa.record_reward(
            conn, signal="ab_winner", score=sa.REWARD_SCORES["ab_winner"],
            interaction_id=win_id, query_group=comp["query_group"], session_id=sid,
            model=win_model, style=sa.STYLE_OF_SIDE[preferred], query_type=win_qtype,
        )
        sa.record_reward(
            conn, signal="ab_loser", score=sa.REWARD_SCORES["ab_loser"],
            interaction_id=lose_id, query_group=comp["query_group"], session_id=sid,
            model=lose_model, style=sa.STYLE_OF_SIDE[loser_side], query_type=win_qtype,
        )
        policy_update = sa.maybe_update_policy(conn, sid)
        sa.maybe_update_profile(conn, profile_sid())

        win_src = _src(win_sources)
        if comp["conversation_id"]:
            sa.save_message(conn, session_id=sid, conversation_id=comp["conversation_id"],
                            role="assistant", content=win_answer, sources=win_src)

        followups = sa.suggest_followups(comp["original_query"], win_answer)
        return jsonify({
            "status": "ok", "preferred": preferred, "followups": followups,
            "policy_update": policy_update, "metrics": metrics_payload(conn, sid),
        })
    finally:
        conn.close()


@app.route("/api/prefer_reason", methods=["POST"])
def api_prefer_reason():
    """Record why the user preferred an answer, and learn a lesson from it."""
    sid = session_id()
    data = request.get_json(silent=True) or {}
    comparison_id = data.get("comparison_id")
    reason = (data.get("reason") or "").strip()[:200]
    if comparison_id is None or not reason:
        return json_error("Missing comparison_id or reason.", 400, "invalid_request")

    conn = db()
    try:
        comp = _owned_comparison(conn, comparison_id)
        if not comp:
            return json_error("Comparison not found.", 404, "not_found_error")
        if comp["preferred"] not in ("A", "B"):
            return json_error("Record a preference before a reason.", 400, "invalid_request")

        sa.set_reason(conn, int(comparison_id), reason)
        bonus = sa.REASON_BONUS.get(reason.lower())
        if bonus:
            win_routing_key = "routing_a" if comp["preferred"] == "A" else "routing_b"
            win_routing_raw = comp.get(win_routing_key) or comp["routing"]
            try:
                routing = json.loads(win_routing_raw) if win_routing_raw else None
            except (json.JSONDecodeError, TypeError):
                routing = None
            b_model, b_qtype = sa.routing_facets(routing)
            sa.record_reward(
                conn, signal="reason_bonus", score=bonus, query_group=comp["query_group"],
                session_id=sid, model=b_model, style=sa.STYLE_OF_SIDE.get(comp["preferred"]),
                query_type=b_qtype,
            )

        topic = sa.extract_topic(comp["original_query"])
        lesson = sa.record_lesson(conn, topic=topic, winning_style=comp["preferred"],
                                  reason=reason, session_id=sid)
        return jsonify({"status": "ok", "learning_card": lesson["text"], "lesson": lesson,
                        "metrics": metrics_payload(conn, sid)})
    finally:
        conn.close()


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Thumbs up/down on a single answer; a 👎 triggers a rephrase-and-retry that
    reuses the original sources (no new Tavily search)."""
    sid = session_id()
    data = request.get_json(silent=True) or {}
    interaction_id = data.get("interaction_id")
    group_id = data.get("group_id")
    original_query = str(data.get("original_query", ""))[:1000]
    feedback = data.get("feedback")
    comment = (data.get("comment") or "").strip()[:500] or None
    if interaction_id is None or group_id is None or feedback not in ("up", "down"):
        return json_error("Missing interaction_id/group_id or invalid feedback.", 400, "invalid_request")

    conn = db()
    try:
        owner = conn.execute("SELECT session_id FROM interactions WHERE id = ?",
                             (int(interaction_id),)).fetchone()
        if not owner or owner[0] != sid:
            return json_error("Interaction not found.", 404, "not_found_error")
        sa.set_feedback(conn, int(interaction_id), feedback, comment)

        fb_routing = sa.get_routing(conn, int(interaction_id))
        fb_model, fb_qtype = sa.routing_facets(fb_routing)
        sa.record_reward(
            conn, signal="thumbs_up" if feedback == "up" else "thumbs_down",
            score=sa.REWARD_SCORES["thumbs_up" if feedback == "up" else "thumbs_down"],
            interaction_id=int(interaction_id), query_group=int(group_id),
            session_id=sid, model=fb_model, query_type=fb_qtype,
        )
        policy_update = sa.maybe_update_policy(conn, sid)
        sa.maybe_update_profile(conn, profile_sid())

        retry, learning, limit = None, False, None
        if feedback == "down":
            attempts = sa.attempts_in_group(conn, int(group_id))
            if attempts <= sa.MAX_RETRIES:
                learning = True
                failed_answer = sa.get_answer(conn, int(interaction_id))
                reused_sources = sa.get_sources(conn, int(interaction_id))
                try:
                    new_query = sa.rephrase_query(original_query=original_query,
                                                  failed_answer=failed_answer, comment=comment)
                    lessons = sa.recent_lessons(conn)
                    new_answer = sa.regenerate_answer(
                        original_query=original_query, improved_query=new_query,
                        previous_answer=failed_answer, sources=reused_sources,
                        comment=comment, lessons=lessons,
                    )
                except providers.ProviderLimit:
                    learning = False
                    limit = limits.limit_info("site", limits.cached_examples(conn))
                else:
                    retry_routing = sa.build_routing("complex")
                    new_id = sa.insert_answer(
                        conn, query_group=int(group_id), original_query=original_query,
                        effective_query=new_query, answer=new_answer, attempt=attempts + 1,
                        sources=reused_sources, routing=retry_routing, session_id=sid,
                    )
                    retry = {
                        "interaction_id": new_id, "group_id": int(group_id),
                        "original_query": original_query, "effective_query": new_query,
                        "answer": new_answer, "sources": reused_sources,
                        "attempt": attempts + 1, "lessons_applied": len(lessons),
                        "routing": retry_routing,
                    }

        return jsonify({"status": "ok", "learning": learning, "retry": retry, "limit": limit,
                        "policy_update": policy_update, "metrics": metrics_payload(conn, sid)})
    finally:
        conn.close()


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear THIS user's/guest's data. Usage counters and the cache are untouched
    (resetting must not refund searches)."""
    sid = session_id()
    conn = db()
    try:
        for table in sa.DATA_TABLES:
            owner_col = "google_id" if table == "user_profiles" else "session_id"
            conn.execute(f"DELETE FROM {table} WHERE {owner_col} = ?", (sid,))
        conn.commit()
        return jsonify({"status": "ok", "metrics": metrics_payload(conn, sid)})
    finally:
        conn.close()


if __name__ == "__main__":
    if not providers.configured():
        print("Warning: GROQ_API_KEY / TAVILY_API_KEY not set — live searches will fail "
              "(set SEARCHIQ_FAKE_PROVIDERS=1 for canned local answers).")
    print(f"[db] {dbmod.describe()}")
    port = int(os.environ.get("PORT", 5001))
    # .env is already loaded by search_agent (load_dotenv=False avoids Flask's
    # cwd-based lookup, which fails in sandboxed folders).
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1",
            load_dotenv=False)
