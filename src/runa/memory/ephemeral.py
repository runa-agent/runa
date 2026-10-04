"""memory/ephemeral.py: `EphemeralMemoryStore`, remembered facts that die with the process.

What `runa.db.memory_store()` resolves to under `RUNA_DATABASE_URL=memory://`. The nearest
neighbors are found by brute-force L2 distance over every stored vector, which is the right
algorithm for the only size this store ever reaches and keeps the whole backend dependency-free:
no `sqlite-vec` extension, no `pgvector`, no server.

Scoped by `user_id` the way `memory/sqlite.py`'s `vec0` partition key is, so a search stays one
user's top-k rather than a global one another user's items could crowd out.
"""

import math
from dataclasses import dataclass
from typing import Any

from runa.memory.store import MemoryMatch


@dataclass
class _Item:
    """One remembered fact: who it belongs to, its text, metadata, and its embedding."""

    id: int
    user_id: str | None
    text: str
    metadata: dict[str, Any] | None
    embedding: list[float]


_items: list[_Item] = []


def reset() -> None:
    """Drop every stored item. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _items.clear()


def distance(left: list[float], right: list[float]) -> float:
    """The L2 distance between two vectors, which is what `vec0` and `pgvector` also report.

    Shared with `knowledge/ephemeral.py`: the two stores differ in what they hold, not in how
    close two embeddings are.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class EphemeralMemoryStore:
    """The in-process `MemoryStore`: this process's remembered facts, gone when it exits."""

    def __init__(self, *, dimensions: int) -> None:
        """Store the embedding size this store expects, for parity with the other adapters."""
        self.dimensions = dimensions

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        item_id = len(_items) + 1
        _items.append(
            _Item(
                id=item_id,
                user_id=user_id,
                text=text,
                metadata=metadata,
                embedding=list(embedding),
            )
        )
        return item_id

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        matches = [
            MemoryMatch(
                id=item.id,
                text=item.text,
                metadata=item.metadata,
                distance=distance(embedding, item.embedding),
            )
            for item in _items
            if item.user_id == user_id
        ]
        matches.sort(key=lambda match: match.distance)
        return matches[:k]

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        for index, item in enumerate(_items):
            if item.id == memory_id and item.user_id == user_id:
                del _items[index]
                return


__all__ = ["EphemeralMemoryStore", "distance", "reset"]
