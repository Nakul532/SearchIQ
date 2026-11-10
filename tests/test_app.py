"""End-to-end tests for the Vercel/Groq/Tavily rebuild (no real API keys needed).

Run:  .venv/bin/python -m pytest -q tests
      SEARCHIQ_TEST_TURSO=1 .venv/bin/python -m pytest -q tests   # via the Turso HTTP adapter

Providers run in fake mode (SEARCHIQ_FAKE_PROVIDERS=1). With SEARCHIQ_TEST_TURSO=1
every DB call goes through db.TursoConnection against an in-process fake of
Turso's /v2/pipeline endpoint, backed by SQLite.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_TMP = tempfile.mkdtemp()
os.environ.update({
    "SEARCHIQ_FAKE_PROVIDERS": "1",
    "FLASK_SECRET_KEY": "test-secret",
    "DB_PATH": os.path.join(_TMP, "test.db"),
    "GUEST_DAILY_CAP": "3",
    "GUEST_IP_DAILY_CAP": "10",
    "USER_DAILY_CAP": "10",
    "SITE_DAILY_CAP": "25",
    "DEMO_VIDEO_URL": "https://example.com/demo",
    "GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "",
})
os.environ.pop("VERCEL", None)
os.environ.pop("SEARCHIQ_SIMULATE_GROQ_429", None)
USE_TURSO = os.environ.get("SEARCHIQ_TEST_TURSO") == "1"
if USE_TURSO:
    os.environ["TURSO_DATABASE_URL"] = "libsql://fake-db.turso.io"
    os.environ["TURSO_AUTH_TOKEN"] = "fake-token"
else:
    os.environ.pop("TURSO_DATABASE_URL", None)

import db as dbmod  # noqa: E402


# --------------------------------------------------------------------------- #
# Fake Turso /v2/pipeline server (in-process), backed by SQLite
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._p


class _FakeTursoSession:
    def __init__(self, path):
        self.path = path

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        assert url == "https://fake-db.turso.io/v2/pipeline", url
        assert headers["Authorization"] == "Bearer fake-token"
        conn = sqlite3.connect(self.path)
        results = []
        try:
            for req in json["requests"]:
                if req["type"] == "close":
                    results.append({"type": "ok", "response": {"type": "close"}})
                    continue
                stmt = req["stmt"]
                args = [self._arg(a) for a in stmt.get("args", [])]
                try:
                    cur = conn.execute(stmt["sql"], args)
                    rows = cur.fetchall()
                    conn.commit()
                except sqlite3.Error as exc:
                    results.append({"type": "error", "error": {"message": str(exc)}})
                    continue
                cols = [{"name": d[0], "decltype": None} for d in (cur.description or [])]
                results.append({"type": "ok", "response": {"type": "execute", "result": {
                    "cols": cols,
                    "rows": [[self._cell(v) for v in r] for r in rows],
                    "affected_row_count": max(cur.rowcount, 0),
                    "last_insert_rowid": str(cur.lastrowid) if cur.lastrowid else None,
                }}})
        finally:
            conn.close()
        return _FakeResp({"baton": None, "base_url": None, "results": results})

    @staticmethod
    def _arg(a):
        t = a["type"]
        if t == "null":
            return None
        if t == "integer":
            assert isinstance(a["value"], str)  # Turso wants integers as strings
            return int(a["value"])
        if t == "float":
            return float(a["value"])
        return a["value"]

    @staticmethod
    def _cell(v):
        if v is None:
            return {"type": "null"}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            return {"type": "float", "value": v}
        return {"type": "text", "value": v}


if USE_TURSO:
    _fake = _FakeTursoSession(os.path.join(_TMP, "turso_backing.db"))
    dbmod._session = lambda: _fake  # route every pipeline call to the fake

import app as appmod  # noqa: E402
import limits  # noqa: E402
import search_agent as sa  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    conn = appmod.db()
    for t in sa.DATA_TABLES + ("usage_counters", "answer_cache", "users"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()
    os.environ.pop("SEARCHIQ_SIMULATE_GROQ_429", None)
    yield


def client(ip="10.0.0.1"):
    c = appmod.app.test_client()
    c.environ_base["REMOTE_ADDR"] = ip
    return c


def ask(c, query, conv="conv-1", force=False):
    r = c.post("/api/query", json={"query": query, "conversation_id": conv, "force": force})
    assert r.status_code == 200, r.data
    return [json.loads(l) for l in r.data.decode().splitlines() if l.strip()]


def types(events):
    return [e["type"] for e in events]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_db_backend():
    assert dbmod.describe().startswith("Turso" if USE_TURSO else "SQLite")


def test_guest_can_search_without_signing_in():
    c = client()
    assert c.get("/").status_code == 200          # no redirect to /login
    me = c.get("/api/me").get_json()
    assert me["guest"] is True and me["user"] is None
    assert me["quota"]["remaining"] == 3
    ev = ask(c, "How does photosynthesis convert light into energy in plants?")
    assert [t for t in types(ev) if t != "step"] == ["meta", "answer", "answer", "check", "comparison", "quota"]
    assert ev[0]["cached"] is False
    assert {e["side"] for e in ev if e["type"] == "answer"} == {"A", "B"}
    assert ev[-1]["remaining"] == 2
    steps = [(e["step"], e["status"]) for e in ev if e["type"] == "step"]
    assert steps == [("search", "active"), ("search", "done"), ("draft", "active"), ("draft", "done"),
                     ("check", "active"), ("check", "done")]
    check = next(e for e in ev if e["type"] == "check")
    assert check["premise"]["ok"] is True
    assert check["sides"]["A"]["confidence"] == {"level": "high", "backed": 1, "total": 1, "contradicted": 0}
    assert check["sides"]["B"]["confidence"]["level"] == "medium"   # 1 of 2 backed


def test_complex_query_routes_ab_to_fast_and_deep_models():
    ev = ask(client(), "Analyze the trade-offs of microservices versus a monolith architecture")
    meta = ev[0]
    assert meta["routing"]["complexity"] == "complex"
    assert meta["routing_a"]["model"] == sa.FAST_MODEL
    assert meta["routing_b"]["model"] == sa.DEEP_MODEL
    simple = ask(client("10.0.0.9"), "What is the boiling point of water at sea level?")[0]
    assert simple["routing_a"]["model"] == simple["routing_b"]["model"] == sa.FAST_MODEL


def test_ab_pick_reason_feedback_loop_and_metrics():
    c = client()
    ev = ask(c, "How do vaccines train the immune system to fight viruses?")
    cid = next(e["comparison_id"] for e in ev if e["type"] == "comparison")
    r = c.post("/api/prefer", json={"comparison_id": cid, "preferred": "B"}).get_json()
    assert r["status"] == "ok" and len(r["followups"]) == 3
    m = r["metrics"]
    assert m["total_queries"] == 1 and m["completion_rate"] == 100.0
    assert "fast_share_pct" in m and "est_cost_per_query_usd" in m and "cost_saved_usd" not in m
    assert m["fast_share_pct"] == 100.0 and m["est_cost_per_query_usd"] > 0
    assert set(m["models_used"]) == {sa.FAST_LABEL, sa.DEEP_LABEL}
    r2 = c.post("/api/prefer_reason", json={"comparison_id": cid, "reason": "more accurate"}).get_json()
    assert r2["learning_card"] and r2["metrics"]["training_progress"]["lessons_learned"] == 1
    assert r2["metrics"]["reward"]["total_events"] == 3
    # Another user can't touch this comparison.
    assert client("10.9.9.9").post("/api/prefer", json={"comparison_id": cid,
                                                        "preferred": "A"}).status_code == 404


def test_thumbs_down_retry_reuses_sources():
    c = client()
    c.get("/")
    with c.session_transaction() as s:
        sid = s["guest_id"]
    conn = appmod.db()
    iid = sa.insert_answer(conn, query_group=1, original_query="q", effective_query="q",
                           answer="meh", attempt=1, sources=[{"url": "https://x", "title": "x"}],
                           routing=sa.build_routing("simple"), session_id=sid)
    conn.close()
    r = c.post("/api/feedback", json={"interaction_id": iid, "group_id": 1, "original_query": "q",
                                      "feedback": "down", "comment": "too short"}).get_json()
    assert r["learning"] is True and r["retry"]["sources"] == [{"url": "https://x", "title": "x"}]
    assert r["retry"]["routing"]["model"] == sa.DEEP_MODEL


def test_guest_cap_is_three_and_shows_limit_card():
    c = client()
    for i in range(3):
        assert "comparison" in types(ask(c, f"Explain how topic number {i} of geology works", conv=f"c{i}"))
    ev = ask(c, "Explain how plate tectonics shape mountains", conv="c4")
    assert types(ev) == ["limit"]
    card = ev[0]
    assert card["reason"] == "guest" and card["demo_url"] == "https://example.com/demo"
    assert "3 free searches" in card["title"] and card["resets_at"]


def test_guest_cap_survives_clearing_cookies():
    # 3 per cookie; clearing cookies (a fresh cookie jar) only helps until the
    # per-IP cap of 10 is reached.
    live = 0
    for jar in range(5):
        c = client(ip="203.0.113.7")  # new cookie jar = cookies cleared
        for i in range(3):
            ev = ask(c, f"Explain how glaciers carve valley {jar}-{i}", conv=f"c{jar}{i}")
            live += "comparison" in types(ev)
    assert live == 10
    fresh = client(ip="203.0.113.7")
    assert fresh.get("/api/me").get_json()["quota"]["remaining"] == 0
    assert types(ask(fresh, "Explain how rivers form deltas over time")) == ["limit"]
    # A different network is unaffected.
    other = client(ip="198.51.100.4")
    assert "comparison" in types(ask(other, "Explain how rivers form deltas over time"))


def test_signed_in_user_gets_user_cap_not_ip_cap():
    c = client(ip="203.0.113.50")
    for i in range(3):
        ask(c, f"Explain how coral reefs grow, part {i}", conv=f"g{i}")
    assert types(ask(c, "Explain how coral polyps feed")) == ["limit"]  # guest cap hit
    with c.session_transaction() as s:
        s["user"] = {"google_id": "g-123", "name": "T", "email": None, "picture": None}
    assert c.get("/api/me").get_json()["quota"]["remaining"] == 10
    assert "comparison" in types(ask(c, "Explain how coral bleaching happens"))


def test_site_wide_daily_cap(monkeypatch):
    monkeypatch.setattr(limits, "SITE_DAILY_CAP", 2)
    ask(client("1.1.1.1"), "Explain how bridges carry heavy loads")
    ask(client("1.1.1.2"), "Explain how tunnels are dug under rivers")
    ev = ask(client("1.1.1.3"), "Explain how dams generate electricity")
    assert types(ev) == ["limit"] and ev[0]["reason"] == "site"


def test_token_budget_exhaustion_shows_limit_card(monkeypatch):
    monkeypatch.setitem(limits.TOKEN_BUDGETS, sa.FAST_MODEL, 1)
    ask(client("2.2.2.1"), "Explain how compilers optimize loops")  # spends > 1 token
    ev = ask(client("2.2.2.2"), "Explain how garbage collectors work")
    assert types(ev) == ["limit"]


def test_simulated_groq_429_shows_limit_card_not_error():
    os.environ["SEARCHIQ_SIMULATE_GROQ_429"] = "1"
    c = client()
    ev = ask(c, "Explain how solar panels convert sunlight to power")
    assert "limit" in types(ev) and "error" not in types(ev)
    card = next(e for e in ev if e["type"] == "limit")
    assert card["demo_url"] == "https://example.com/demo" and card["title"]
    # A 429 must not count against the guest's own quota.
    assert c.get("/api/me").get_json()["quota"]["remaining"] == 3
    # The non-streaming endpoints degrade the same way.
    os.environ.pop("SEARCHIQ_SIMULATE_GROQ_429")
    c2 = client("5.5.5.5")
    ev2 = ask(c2, "How do vaccines train the immune system?")
    cid = next(e["comparison_id"] for e in ev2 if e["type"] == "comparison")
    os.environ["SEARCHIQ_SIMULATE_GROQ_429"] = "1"
    r = c2.post("/api/prefer", json={"comparison_id": cid, "preferred": "A"})
    assert r.status_code == 200 and r.get_json()["followups"] == []


def test_cached_answer_shows_date_and_is_free():
    c = client()
    q = "How do noise-cancelling headphones work?"
    first = ask(c, q, conv="a")
    assert first[0]["cached"] is False
    before = c.get("/api/me").get_json()["quota"]["remaining"]
    again = ask(client("7.7.7.7"), q.upper() + "  ", conv="b")  # normalized match, new user
    assert again[0]["cached"] is True and again[0]["cached_at"]
    assert types(again) == ["meta", "step", "answer", "answer", "check", "step", "comparison", "quota"]
    assert next(e for e in again if e["type"] == "check")["sides"]["B"]["claims"]  # check is cached too
    assert c.get("/api/me").get_json()["quota"]["remaining"] == before
    # The cached pair is a real comparison for the new user (pick still works).


def test_time_sensitive_queries_skip_the_cache():
    for q in ("What is the latest iPhone model today?", "Bitcoin price", "current weather in Boston",
              "news about NASA"):
        assert limits.is_time_sensitive(q), q
    c = client()
    q = "What are the latest developments in fusion energy?"
    ask(c, q, conv="a")
    assert types(ask(c, q, conv="b"))[0] == "meta"
    assert ask(c, q, conv="c")[0]["cached"] is False


def test_examples_served_from_cache_even_when_capped():
    c = client("9.9.9.9")
    nvidia, apple, msft = limits.EXAMPLE_QUERIES
    for i, q in enumerate(limits.EXAMPLE_QUERIES):   # first asks are live: 3 searches
        assert not limits.is_time_sensitive(q), q
        assert ask(c, q, conv=f"e{i}")[0]["cached"] is False
    assert types(ask(c, "Explain how magnets work", conv="x")) == ["limit"]  # capped now
    card = ask(c, "Explain how magnets work", conv="y")[0]
    assert {e["query"] for e in card["examples"]} == set(limits.EXAMPLE_QUERIES)
    for i, q in enumerate((nvidia, apple.upper(), msft + "  ")):  # served cached while capped
        ev = ask(c, q, conv=f"again{i}")
        assert ev[0]["type"] == "meta" and ev[0]["cached"] is True, q
        assert "comparison" in types(ev)


def test_history_and_reset_are_per_guest():
    c = client()
    ask(c, "How do tides work on Earth?", conv="h1")
    convs = c.get("/api/conversations").get_json()["conversations"]
    assert len(convs) == 1
    assert client("4.4.4.4").get("/api/conversations").get_json()["conversations"] == []
    assert c.get("/api/conversation/h1").get_json()["messages"]
    c.post("/api/reset")
    assert c.get("/api/conversations").get_json()["conversations"] == []
    assert c.get("/api/me").get_json()["quota"]["remaining"] == 2  # reset doesn't refund


def test_removed_admin_endpoints():
    c = client()
    assert c.post("/api/migrate").status_code == 404
    assert c.post("/api/restore", json={}).status_code == 404


def test_vercel_requires_secret_key():
    import subprocess
    env = dict(os.environ, VERCEL="1", FLASK_SECRET_KEY="")
    r = subprocess.run([sys.executable, "-c", "import app"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "FLASK_SECRET_KEY must be set" in r.stderr


def test_starts_cleanly_on_vercel_without_google_vars():
    """No GOOGLE_* vars: the app imports, /api/me reports sign-in disabled (so the
    UI hides the button), and the auth routes fall back instead of erroring."""
    import subprocess
    code = (
        "import app\n"
        "c = app.app.test_client()\n"
        "me = c.get('/api/me').get_json()\n"
        "assert me['oauth_configured'] is False and me['guest'] is True, me\n"
        "assert c.get('/').status_code == 200\n"
        "r = c.get('/auth/login'); assert r.status_code == 302 and r.location.endswith('/login'), r\n"
        "assert b'Continue as a guest' in c.get('/login').data\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GOOGLE_", "TURSO_"))}
    env.update(VERCEL="1", FLASK_SECRET_KEY="x" * 64)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_agent_check_normalizes_and_labels_deterministically():
    import agent
    raw = {"premise": {"ok": False, "note": "The event has not happened."},
           "claims": [{"side": "B", "text": "X won", "status": "backed", "sources": [1]},
                      {"side": "B", "text": "No source given", "status": "backed", "sources": []},
                      {"side": "B", "text": "Bad source", "status": "contradicted", "sources": [9, 2]},
                      {"side": "Z", "text": "ignored", "status": "backed", "sources": [1]},
                      {"side": "A", "text": "ignored status", "status": "maybe", "sources": [1]}],
           "disagreements": [{"topic": "Winner", "A": "not yet", "B": "X", "supported": "a", "sources": [1]}]}
    c = agent.normalize(raw, ["A", "B"], 3)
    b = c["sides"]["B"]
    assert [x["status"] for x in b["claims"]] == ["backed", "unsupported", "contradicted"]
    assert b["claims"][2]["sources"] == [2]                       # out-of-range source dropped
    assert b["confidence"]["level"] == "low"                       # false premise → low
    assert c["sides"]["A"]["confidence"]["level"] == "low"
    assert c["disagreements"][0]["supported"] == "A"
    note = agent.render_note(c, "B")
    assert "premise" in note and "✗ Contradicted [2]" in note and "Double-check" in note
    ok = agent.normalize({"claims": [{"side": "A", "text": "t", "status": "backed", "sources": [1]}]}, ["A"], 3)
    assert ok["sides"]["A"]["confidence"]["level"] == "high" and ok["disagreements"] == []
    assert agent.check_answers("q", [], {"A": "a"}, llm=lambda *a: "not json") is None

