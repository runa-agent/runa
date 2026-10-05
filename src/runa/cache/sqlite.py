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
from runa.db import DEFAULT_DB_PATH
from runa.db.schema import SQLITE, ddl
from runa.db.sqlite import connect as _connect_db

_DDL = ddl(SQLITE, ENTRIES)


class SQLiteCache:
    """Persistent `Cache`: the `cache_entries` table inside `db/runa.db`, surviving restarts."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Store where this cache's table lives; the table is created on first use."""
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, _DDL)

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        with closing(self._connect()) as conn:
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
