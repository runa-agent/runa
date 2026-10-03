"""cache/postgres.py: `PostgresCache`, the shared persistent `Cache`.

The `runa-ai[postgres]` extra, not a core dependency. The same `cache_entries` table as
`cache/sqlite.py`, in the database the deployment already has, so going multi-replica adds no
second service: this is what `runa.db.cache()` returns once `RUNA_DATABASE_URL` is shared, the
way Rails 8 put its cache in the primary database rather than assuming Redis.

Expiry is lazy, exactly as in `cache/sqlite.py`: an expired key is evicted the next time `get`
looks it up. `RedisCache` is the backend to reach for when that tradeoff (or the query-path cost
of a cache read) stops being the right one.
"""

import json
import time
from typing import Any

import asyncpg

from runa.db.pool import connect as _connect

_TABLE = "cache_entries"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at DOUBLE PRECISION
);
"""


class PostgresCache:
    """Persistent `Cache`: the `cache_entries` table in this deployment's shared Postgres."""

    def __init__(self, url: str) -> None:
        """Store which Postgres database this cache's entries live in; connected lazily."""
        self.url = url

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.url, _DDL)

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        pool = await self._pool()
        row = await pool.fetchrow(f"SELECT value, expires_at FROM {_TABLE} WHERE key = $1", key)
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] <= time.time():
            await pool.execute(f"DELETE FROM {_TABLE} WHERE key = $1", key)
            return None
        return json.loads(row["value"])

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        pool = await self._pool()
        await pool.execute(
            f"""
            INSERT INTO {_TABLE} (key, value, expires_at) VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE
            SET value = excluded.value, expires_at = excluded.expires_at
            """,
            key,
            json.dumps(value),
            time.time() + ttl if ttl is not None else None,
        )

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        pool = await self._pool()
        await pool.execute(f"DELETE FROM {_TABLE} WHERE key = $1", key)

    async def clear(self) -> None:
        """Remove every entry."""
        pool = await self._pool()
        await pool.execute(f"DELETE FROM {_TABLE}")


__all__ = ["PostgresCache"]
