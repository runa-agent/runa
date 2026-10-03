"""cache.py: `Cache`, a minimal get/set/delete/clear protocol, and its two backends.

Deliberately standalone -- nothing here imports `agent.py`, `memory.py`, or `session.py`, and
nothing in those wires a `Cache` in automatically. It's plain application-level caching, not a
Runa primitive with a `"auto"`/`"llm"` opt-in story like [Memory](./memory.py).

Two backends behind the same `Cache` protocol: `MemoryCache` (an in-process dict, gone when the
process exits) and `SQLiteCache` (the `cache_entries` table inside `runa.db`, same
connect-and-create-if-missing pattern as `session.py`/`memory.py`, so a local app accumulates one
database with no setup regardless of which module writes to it first). Values round-trip through
`json.dumps`/`json.loads`, not `pickle` -- safe to load from a file another process may have
written, at the cost of only JSON-serializable values being cacheable.
"""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

from runa.db.sqlite import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db

_TABLE = "cache_entries"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL
);
"""


class Cache(Protocol):
    """What application code needs from a cache: get/set/delete/clear, nothing lower.

    No inheritance required -- `MemoryCache` and `SQLiteCache` satisfy this by matching shape,
    the same escape hatch as `MemoryStore`/`SessionABC`'s custom-backend story. Both expire
    entries lazily: an expired key is only evicted (and reported as a miss) the next time `get`
    looks it up, not by a background sweep.
    """

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        ...

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        ...

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        ...

    async def clear(self) -> None:
        """Remove every entry."""
        ...


class MemoryCache:
    """In-process `Cache`: a plain dict, gone when the process exits."""

    def __init__(self) -> None:
        """Start empty."""
        self._entries: dict[str, tuple[Any, float | None]] = {}

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at <= time.monotonic():
            del self._entries[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        expires_at = time.monotonic() + ttl if ttl is not None else None
        self._entries[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        self._entries.pop(key, None)

    async def clear(self) -> None:
        """Remove every entry."""
        self._entries.clear()


class SQLiteCache:
    """Persistent `Cache`: the `cache_entries` table inside `runa.db`, surviving restarts."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Store where this cache's table lives; the table is created on first use."""
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, _DDL)

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT value, expires_at FROM {_TABLE} WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and expires_at <= time.time():
                conn.execute(f"DELETE FROM {_TABLE} WHERE key = ?", (key,))
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
                f"INSERT INTO {_TABLE} (key, value, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expires_at = excluded.expires_at",
                (key, json.dumps(value), expires_at),
            )
            conn.commit()

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {_TABLE} WHERE key = ?", (key,))
            conn.commit()

    async def clear(self) -> None:
        """Remove every entry."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {_TABLE}")
            conn.commit()


__all__ = ["Cache", "MemoryCache", "SQLiteCache"]
