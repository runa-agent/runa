"""cache/database.py: `DatabaseCache`, the persistent `Cache` over a `Database`.

The `cache_entries` table in whichever database the deployment already has: `db/runa.db` locally,
the shared Postgres once `RUNA_DATABASE_URL` names one -- the way Rails 8 put its cache in the
primary database rather than assuming Redis. A cache survives restarts with no setup and no second
service. `runa.db.cache()` returns this.

Expiry is lazy: an expired key is evicted, and reported as a miss, the next time `get` looks it
up. `RedisCache` is the backend to reach for when that tradeoff (or the query-path cost of a cache
read) stops being the right one.
"""

import json
import time
from typing import Any

from runa.cache import ENTRIES
from runa.db.handle import Database


class DatabaseCache:
    """Persistent `Cache`: the `cache_entries` table of a `Database`, surviving restarts."""

    def __init__(self, database: Database) -> None:
        """Store the database this cache's entries live in."""
        self.database = database

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired.

        A lookup only reads, so a miss on a project with no `db/runa.db` yet is a miss, not a new
        file: the first `set` is what creates one.
        """
        row = await self.database.fetchrow(
            f"SELECT value, expires_at FROM {ENTRIES.name} WHERE key = $1", key
        )
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] <= time.time():
            await self.database.execute(f"DELETE FROM {ENTRIES.name} WHERE key = $1", key)
            return None
        return json.loads(row["value"])

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        await self.database.execute(
            f"INSERT INTO {ENTRIES.name} (key, value, expires_at) VALUES ($1, $2, $3) "
            "ON CONFLICT (key) DO UPDATE "
            "SET value = excluded.value, expires_at = excluded.expires_at",
            key,
            json.dumps(value),
            time.time() + ttl if ttl is not None else None,
        )

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        await self.database.execute(f"DELETE FROM {ENTRIES.name} WHERE key = $1", key)

    async def clear(self) -> None:
        """Remove every entry."""
        await self.database.execute(f"DELETE FROM {ENTRIES.name}")


__all__ = ["DatabaseCache"]
