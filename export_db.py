#!/usr/bin/env python3
"""Export the local SQLite tables to the JSON shape that /api/restore accepts.

This is the companion to the POST /api/restore endpoint. Because `railway run`
executes locally (it injects Railway's env vars but does NOT mount the /data
volume), it can't write to the deployed database. Instead: export your local
rows here, then POST them to the running app — /api/restore executes *inside* the
container, so it writes to the real persistent volume.

Usage:
    # 1) write db_export.json from the local DB (DB_PATH / search_agent.db)
    python3 export_db.py
    python3 export_db.py mydata.json                 # custom output path

    # 2) upload it to the deployed app in one step (uses the X-Migrate-Token
    #    header; token from --token or the MIGRATE_TOKEN env var)
    MIGRATE_TOKEN=... python3 export_db.py --url https://your-app.up.railway.app

Output JSON: {"table_name": [ {column: value, ...}, ... ], ...}
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import search_agent as sa

# Every app table, including accounts. Mirrors what /api/restore will accept.
EXPORT_TABLES = list(sa.DATA_TABLES) + ["users"]


def export_tables(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Read every app table into {table: [row-dicts]}.

    Values are kept exactly as stored — TEXT columns that hold JSON (sources,
    routing, top_topics, …) come back as strings, which /api/restore inserts
    verbatim, so the round-trip is faithful.
    """
    sa.init_db(conn)  # ensure all known tables exist before SELECTing
    out: dict[str, list[dict]] = {}
    for table in EXPORT_TABLES:
        try:
            cur = conn.execute(f"SELECT * FROM {table}")
        except sqlite3.OperationalError:
            continue
        cols = [d[0] for d in cur.description]
        out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export local SQLite tables to JSON for the /api/restore endpoint."
    )
    parser.add_argument("output", nargs="?", default="db_export.json",
                        help="output JSON path (default: db_export.json)")
    parser.add_argument("--url",
                        help="base URL of the deployed app to POST the data to "
                             "(e.g. https://your-app.up.railway.app)")
    parser.add_argument("--token",
                        help="MIGRATE_TOKEN for /api/restore (defaults to the MIGRATE_TOKEN env var)")
    args = parser.parse_args()

    conn = sqlite3.connect(sa.DB_PATH)
    try:
        data = export_tables(conn)
    finally:
        conn.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    total = sum(len(rows) for rows in data.values())
    print(f"Exported {total} rows from {sa.DB_PATH} -> {args.output}")
    for table, rows in data.items():
        print(f"  {table:<15} {len(rows)}")

    if not args.url:
        print(f"\nTo upload to the deployed app, run:\n"
              f"  curl -X POST '<your-app-url>/api/restore?token=$MIGRATE_TOKEN' \\\n"
              f"    -H 'Content-Type: application/json' --data @{args.output}")
        return 0

    token = args.token or os.environ.get("MIGRATE_TOKEN")
    if not token:
        print("Error: --url given but no token — set MIGRATE_TOKEN or pass --token.", file=sys.stderr)
        return 1
    import requests  # only needed for the upload path

    endpoint = args.url.rstrip("/") + "/api/restore"
    print(f"\nPOSTing {total} rows to {endpoint} ...")
    resp = requests.post(endpoint, json=data, headers={"X-Migrate-Token": token}, timeout=120)
    print(f"Response: {resp.status_code} {resp.text[:500]}")
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
