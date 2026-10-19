"""cache/sqlite.py: `SQLiteCache`, the local persistent `Cache`.

`db/runa.db`'s `cache_entries` table, in the same connect-and-create-if-missing file every other
local adapter writes to (`db/sqlite.py`), so a cache survives restarts with no setup and no
second service. `runa.db.cache()` returns this whenever the deployment's state is local.
"""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.cache import ENTRIES
from runa.db import sqlite_database
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import read as _read_db


class SQLiteCache:
    """Persistent `Cache`: the `cache_entries` table inside `db/runa.db`, surviving restarts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store where this cache's table lives, defaulting to `db.sqlite_database()`.

        The table is created on first use.
        """
        self.db_path = Path(db_path) if db_path is not None else sqlite_database()

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, ENTRIES)

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired.

        A lookup reads (`db/sqlite.py`'s `read`), so a miss on a project with no `db/runa.db`
        yet is a miss, not a new file: the first `set` is what creates one.
        """
        with _read_db(self.db_path, ENTRIES) as conn:
            if conn is None:
                return None
            row = conn.execute(
                f"SELECT value, expires_at FROM {ENTRIES.name} WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and expires_at <= time.time():
                conn.execute(f"DELETE FROM {ENTRIES.name} WHERE key = ?", (key,))
                conn.commit()
                return None
        return json.loads(value)

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        expires_at = time.time() + ttl if ttl is not None else None
        with closing(self._connect()) as conn:
            conn.execute(
                f"INSERT INTO {ENTRIES.name} (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expires_at = excluded.expires_at",
                (key, json.dumps(value), expires_at),
            )
            conn.commit()

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {ENTRIES.name} WHERE key = ?", (key,))
            conn.commit()

    async def clear(self) -> None:
        """Remove every entry."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {ENTRIES.name}")
            conn.commit()


__all__ = ["SQLiteCache"]
