#!/usr/bin/env python3
"""Restore the SearchIQ database from backup.sql.

Reads backup.sql (a sqlite `.dump`) and replays it into the database at the
resolved DB_PATH — on Railway that's the persistent volume, so the imported
data survives redeploys.

Usage:
    # locally
    python3 import_db.py

    # against the Railway environment (so DB_PATH / RAILWAY_VOLUME_MOUNT_PATH
    # point at the deployed volume)
    railway run python3 import_db.py
"""

from __future__ import annotations

import os
import sqlite3
import sys

import search_agent as sa

BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup.sql")


def main() -> int:
    if not os.path.exists(BACKUP_PATH):
        print(f"Error: backup.sql not found at {BACKUP_PATH}", file=sys.stderr)
        return 1

    with open(BACKUP_PATH, "r", encoding="utf-8") as f:
        sql_text = f.read()

    print(f"Restoring into {sa.DB_PATH} from {BACKUP_PATH} ...")
    conn = sqlite3.connect(sa.DB_PATH)
    try:
        counts = sa.restore_from_sql(conn, sql_text)
    finally:
        conn.close()

    print("Import complete. Row counts:")
    for table, n in counts.items():
        print(f"  {table:<13} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
