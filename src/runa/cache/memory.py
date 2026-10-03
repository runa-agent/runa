"""cache/memory.py: `MemoryCache`, a `Cache` that is a plain dict.

The one backend `runa.db` never picks for you: a cache that dies with the process is a choice,
not a deployment's default, so you ask for it by name. Nothing to do with `runa.memory`, which is
durable facts about a user; this is `ActiveSupport::Cache::MemoryStore`'s namesake.
"""

import time
from typing import Any


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


__all__ = ["MemoryCache"]
