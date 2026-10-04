"""memory/store.py: `MemoryStore`, the vectors a `Memory` searches, and the match it returns.

One interface, three adapters: `memory/sqlite.py`, `memory/postgres.py`, `memory/ephemeral.py`.
Which one a bare `Memory()` gets is `runa.db.memory_store(...)`'s decision, asked once, so nothing
above it names a backend.

Here rather than in `runa.memory` for the same reason as `session/store.py`, `tracing/store.py`
and `eval/store.py`: the three adapters need the contract and the match object, and nothing else
in this package -- not the embedding model, not the extraction prompt, not `Memory` itself. Only
`cache/__init__.py` declares its own interface, because there the store *is* the concern.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class MemoryMatch:
    """One `Memory.search` result: its id, stored text, metadata, and distance to the query."""

    id: int
    text: str
    metadata: dict[str, Any] | None
    distance: float


class MemoryStore(Protocol):
    """The storage a `Memory` needs: add/search/delete already-embedded text, scoped by user.

    The escape hatch for `Memory(store=...)`: any object with these three async methods works,
    no inheritance required. `SQLiteMemoryStore` and `PostgresMemoryStore` satisfy it by matching
    shape; swapping in a hosted vector DB needs no change to `Memory`, `Agent`, or the lifecycle.
    """

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        ...

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first.

        Scoped strictly: an item stored with `user_id=None` is its own scope, not one every user
        can see, in every adapter.
        """
        ...

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists.

        A no-op when no such item exists, including when it belongs to a different `user_id`.
        """
        ...


__all__ = ["MemoryMatch", "MemoryStore"]
