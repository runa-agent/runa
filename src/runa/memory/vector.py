"""memory/vector.py: `MemoryStore` over any `VectorStore`, which is what all three adapters are.

A remembered fact is an embedding, the text it was embedded from, the metadata it arrived with,
and the user it belongs to. `runa.db.vectors` stores exactly that, for either concern, so what
belongs here is only the mapping: which columns a memory occupies, that `user_id` is the partition
rather than a filter, and that a row comes back as a `MemoryMatch`.

The three named adapters (`memory/sqlite.py`, `memory/postgres.py`, `memory/ephemeral.py`) are
this class over the matching `VectorStore`. They stay separate modules because they are separate
public import paths and because the Postgres one must not pull `asyncpg` into an app without the
extra, not because any of them has storage logic of its own.
"""

from typing import Any

from runa.db.vectors import Column, VectorMatch, VectorSpec, VectorStore
from runa.memory.store import MemoryMatch

#: `user_id` first so the generated tables keep the column order they already had on disk.
COLUMNS = {
    "user_id": Column(),
    "text": Column(required=True),
    "metadata": Column(json=True),
}


def spec(dimensions: int) -> VectorSpec:
    """The `memory_items`/`memory_vectors` spec, for vectors of `dimensions` floats."""
    return VectorSpec(
        name="memory",
        dimensions=dimensions,
        columns=COLUMNS,
        partition_by="user_id",
    )


class VectorMemoryStore:
    """A `MemoryStore` reading and writing one `VectorStore`.

    Scoped strictly, in every backend: an item stored with `user_id=None` is its own scope, not
    one every user can see, because `user_id` is the store's partition.
    """

    def __init__(self, vectors: VectorStore) -> None:
        """Hold the vector storage this memory's facts live in."""
        self._vectors = vectors

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        return await self._vectors.add(
            payload={"user_id": user_id, "text": text, "metadata": metadata},
            embedding=embedding,
        )

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        found = await self._vectors.nearest(embedding=embedding, k=k, partition=user_id)
        return [_match(row) for row in found]

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        await self._vectors.delete(item_id=memory_id, partition=user_id)


def _match(row: VectorMatch) -> MemoryMatch:
    """One stored row as the `MemoryMatch` a `Memory` hands back."""
    return MemoryMatch(
        id=row.id,
        text=row.payload["text"],
        metadata=row.payload["metadata"],
        distance=row.distance,
    )


__all__ = ["COLUMNS", "VectorMemoryStore", "spec"]
