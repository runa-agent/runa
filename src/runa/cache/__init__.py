"""`runa.cache`: `Cache`, a minimal get/set/delete/clear protocol, and its backends.

Deliberately standalone: nothing here imports `agent.py`, `memory/` or `session/`, and nothing in
those wires a `Cache` in automatically. It's plain application-level caching, not a Runa primitive
with an `"auto"`/`"llm"` opt-in story like `Memory`.

Four backends behind one protocol. `MemoryCache` (`cache/memory.py`) is an in-process dict, gone
when the process exits. The other three are persistent, and `runa.db.cache()` picks between them
the same way it picks a session store: `SQLiteCache` locally, `PostgresCache` when
`RUNA_DATABASE_URL` is a `postgresql://` one, and `MemoryCache` under `memory://`, where nothing
else is persisted either. `RedisCache` (`cache/redis.py`) is the explicit opt-in for a deployment
that wants its hot keys off the database's query path.

Values round-trip through `json.dumps`/`json.loads`, not `pickle`, in every persistent backend:
safe to load from a store another process may have written, at the cost of only
JSON-serializable values being cacheable.

`ENTRIES` below is the one table the two SQL backends keep, declared here beside the protocol
they both answer and rendered per dialect by `db/schema.py`, rather than written out once in
each of them.
"""

from typing import Any, Protocol

from runa.db.schema import Column, Table

ENTRIES = Table(
    "cache_entries",
    columns=(
        Column("key", "text", primary_key=True),
        Column("value", "text"),
        Column("expires_at", "float", null=True),
    ),
)


class Cache(Protocol):
    """What application code needs from a cache: get/set/delete/clear, nothing lower.

    No inheritance required, every backend satisfies this by matching shape, the same escape
    hatch as `MemoryStore`/`SessionStore`'s custom-backend story. The two local backends expire
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


# Below `Cache`, for symmetry with the other concern packages. Both local backends are always
# importable; `PostgresCache` and `RedisCache` need their extras, so they stay explicit imports.
from runa.cache.memory import MemoryCache  # noqa: E402
from runa.cache.sqlite import SQLiteCache  # noqa: E402

__all__ = ["Cache", "MemoryCache", "SQLiteCache"]
