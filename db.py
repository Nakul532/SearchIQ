"""Database connection: Turso (libSQL over HTTP) in production, SQLite locally.

Vercel functions have no persistent disk, so production data lives in a free
Turso database. Turso speaks SQLite's SQL dialect, so every query in
search_agent.py runs unchanged; this module only swaps the *connection*.

- TURSO_DATABASE_URL set  → TursoConnection (stateless HTTP, /v2/pipeline API)
- otherwise               → sqlite3 file at LOCAL_DB_PATH (local dev / CLI)

TursoConnection implements the small slice of the sqlite3.Connection API the app
uses: execute() → cursor (fetchone / fetchall / iteration / lastrowid /
rowcount), executemany(), commit() (no-op: each statement autocommits), close().
"""

from __future__ import annotations

import base64
import os
import sqlite3
import threading

import requests

# Local fallback DB. A NEW file name on purpose: the old search_agent.db (with
# the pre-migration Haiku/Sonnet rows) is left untouched as an archive, and the
# dashboard starts fresh from the migration.
LOCAL_DB_PATH = os.environ.get("DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "searchiq_local.db"
)


class DatabaseError(sqlite3.OperationalError):
    """A Turso statement failed (subclasses sqlite3's error so callers that catch
    sqlite3.OperationalError keep working)."""


def _encode_arg(v):
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode()}
    return {"type": "text", "value": str(v)}


def _decode_cell(c):
    t = c.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(c["value"])
    if t == "float":
        return float(c["value"])
    if t == "blob":
        return base64.b64decode(c.get("base64", ""))
    return c.get("value")


class _Cursor:
    def __init__(self, rows: list[tuple], lastrowid, rowcount: int):
        self._rows = rows
        self._i = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchone(self):
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self):
        rest = self._rows[self._i:]
        self._i = len(self._rows)
        return rest

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row


# One HTTP session per process keeps the TLS connection warm across requests.
_http = threading.local()


def _session() -> requests.Session:
    s = getattr(_http, "s", None)
    if s is None:
        s = _http.s = requests.Session()
    return s


class TursoConnection:
    """Minimal sqlite3-compatible connection over Turso's HTTP pipeline API."""

    def __init__(self, url: str, token: str, timeout: float = 15.0):
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        self._endpoint = url.rstrip("/") + "/v2/pipeline"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout

    def _pipeline(self, stmts: list[tuple[str, tuple]]) -> list[dict]:
        reqs = [
            {"type": "execute", "stmt": {"sql": sql, "args": [_encode_arg(a) for a in params]}}
            for sql, params in stmts
        ]
        reqs.append({"type": "close"})
        resp = _session().post(self._endpoint, json={"requests": reqs},
                               headers=self._headers, timeout=self._timeout)
        if resp.status_code >= 400:
            raise DatabaseError(f"Turso HTTP {resp.status_code}: {resp.text[:300]}")
        results = resp.json().get("results", [])
        out = []
        for r in results[: len(stmts)]:
            if r.get("type") != "ok":
                err = (r.get("error") or {}).get("message") or str(r)
                raise DatabaseError(err)
            out.append(r["response"]["result"])
        return out

    def execute(self, sql: str, params=()):
        res = self._pipeline([(sql, tuple(params or ()))])[0]
        rows = [tuple(_decode_cell(c) for c in row) for row in res.get("rows", [])]
        lid = res.get("last_insert_rowid")
        return _Cursor(rows, int(lid) if lid is not None else None,
                       int(res.get("affected_row_count") or 0))

    def executemany(self, sql: str, seq):
        stmts = [(sql, tuple(p)) for p in seq]
        if stmts:
            self._pipeline(stmts)
        return _Cursor([], None, len(stmts))

    def execute_batch(self, stmts: list[tuple[str, tuple]]) -> None:
        """Run several statements in ONE round trip (used for schema setup)."""
        if stmts:
            self._pipeline(stmts)

    def commit(self) -> None:  # every pipeline statement autocommits
        pass

    def close(self) -> None:
        pass


def using_turso() -> bool:
    return bool(os.environ.get("TURSO_DATABASE_URL"))


def connect():
    """Open a connection to the configured database (Turso if set, else SQLite)."""
    url = (os.environ.get("TURSO_DATABASE_URL") or "").strip()
    if url:
        token = (os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
        return TursoConnection(url, token)
    os.makedirs(os.path.dirname(os.path.abspath(LOCAL_DB_PATH)), exist_ok=True)
    return sqlite3.connect(LOCAL_DB_PATH)


def describe() -> str:
    if using_turso():
        host = os.environ["TURSO_DATABASE_URL"].split("://", 1)[-1].split("/", 1)[0]
        return f"Turso ({host})"
    return f"SQLite ({LOCAL_DB_PATH})"
