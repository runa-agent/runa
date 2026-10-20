"""cache/sqlite.py: `SQLiteCache`, the local persistent `Cache`.

`db/runa.db`'s `cache_entries` table, in the same connect-and-create-if-missing file every other
local adapter writes to (`db/sqlite.py`), so a cache survives restarts with no setup and no
second service. `runa.db.cache()` returns this whenever the deployment's state is local.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from runa.cache import ENTRIES
from runa.db import sqlite_database
from runa.db.sqlite import run, run_read


class SQLiteCache:
    """Persistent `Cache`: the `cache_entries` table inside `db/runa.db`, surviving restarts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store where this cache's table lives, defaulting to `db.sqlite_database()`.

        The table is created on first use.
        """
        self.db_path = Path(db_path) if db_path is not None else sqlite_database()

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired.

        A lookup reads (`db/sqlite.py`'s `read`), so a miss on a project with no `db/runa.db`
        yet is a miss, not a new file: the first `set` is what creates one.
        """

        def body(conn: sqlite3.Connection | None) -> str | None:
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
            return value

        value = await run_read(self.db_path, (ENTRIES,), body)
        return json.loads(value) if value is not None else None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        expires_at = time.time() + ttl if ttl is not None else None
        encoded = json.dumps(value)

        def body(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"INSERT INTO {ENTRIES.name} (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expires_at = excluded.expires_at",
                (key, encoded, expires_at),
            )
            conn.commit()

        await run(self.db_path, (ENTRIES,), body)

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""

        def body(conn: sqlite3.Connection) -> None:
            conn.execute(f"DELETE FROM {ENTRIES.name} WHERE key = ?", (key,))
            conn.commit()

        await run(self.db_path, (ENTRIES,), body)

    async def clear(self) -> None:
        """Remove every entry."""

        def body(conn: sqlite3.Connection) -> None:
            conn.execute(f"DELETE FROM {ENTRIES.name}")
            conn.commit()

        await run(self.db_path, (ENTRIES,), body)


__all__ = ["SQLiteCache"]
