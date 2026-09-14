"""db/redis.py: `RedisCache`, a `Cache` backed by Redis, shared across processes.

Optional -- the `runa[redis]` extra, not a core dependency; nothing outside this module imports
it. Same shape as `cache.py`'s `MemoryCache`/`SQLiteCache` -- no inheritance required, just the
matching `get`/`set`/`delete`/`clear` methods -- for a deployment where multiple processes need
to share one cache instead of each having their own in-process dict or `db/runa.db`.

Values round-trip through `json.dumps`/`json.loads`, same as `SQLiteCache`, so only
JSON-serializable values are cacheable. Unlike the other two backends, expiry is Redis's own
(`PX`), not lazy-on-read: an expired key is simply gone, not evicted by the next `get`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import redis.asyncio as redis

DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class RedisCache:
    """`Cache` backed by Redis: `MemoryCache`/`SQLiteCache`'s shape, shared across processes."""

    def __init__(self, url: str = DEFAULT_REDIS_URL) -> None:
        """Store which Redis instance this cache's entries live in; connected lazily."""
        self.url = url
        self._client: tuple[asyncio.AbstractEventLoop, redis.Redis] | None = None

    def _connect(self) -> redis.Redis:
        """Return this cache's client on the *current* event loop, (re)creating it if stale.

        A `redis.asyncio.Redis`'s connections belong to the loop running when it first
        connects, so a client left over from a now-closed loop (e.g. a second `asyncio.run()`
        call reusing this same `RedisCache`, as `Runner.run_sync` makes easy to hit) would
        crash with "Event loop is closed" instead of reconnecting; same fix as
        `ModelProvider`'s HTTP clients and `db/postgres.py`'s pools.
        """
        loop = asyncio.get_running_loop()
        if self._client is None or self._client[0] is not loop:
            self._client = (loop, redis.from_url(self.url))
        return self._client[1]

    async def get(self, key: str) -> Any:
        """Return the value stored for `key`, or `None` if it's missing or expired."""
        value = await self._connect().get(key)
        return json.loads(value) if value is not None else None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store `value` for `key`, replacing whatever was there.

        `ttl` is seconds until the entry expires; `None` (the default) never expires.
        """
        client = self._connect()
        payload = json.dumps(value)
        if ttl is not None:
            await client.set(key, payload, px=round(ttl * 1000))
        else:
            await client.set(key, payload)

    async def delete(self, key: str) -> None:
        """Remove `key`, if present."""
        await self._connect().delete(key)

    async def clear(self) -> None:
        """Remove every entry in this Redis instance's current database."""
        await self._connect().flushdb()


__all__ = ["DEFAULT_REDIS_URL", "RedisCache"]
