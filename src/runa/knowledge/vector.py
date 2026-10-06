"""knowledge/vector.py: `KnowledgeStore` over any `VectorStore`, which is what every adapter is.

`memory/vector.py`'s counterpart, and separate from it for the same reason the two concerns are
separate: these chunks are application-scoped, so the spec declares no partition and nothing here
is keyed by `user_id`. It splits its backends the same way too: `knowledge/sqlite.py` and
`knowledge/postgres.py` are named modules, and in-process is paired in
`runa.db.knowledge_store`.

What belongs here is only the mapping: a chunk occupies a `text` and a `source` column, and a row
comes back as a `KnowledgeMatch`. The storage, the DDL and the distance are `runa.db.vectors`'s.
"""

from runa.db.vectors import Column, VectorMatch, VectorSpec, VectorStore
from runa.knowledge.store import KnowledgeMatch

#: `text` first so the generated table keeps the column order it already had on disk.
COLUMNS = {"text": Column(required=True), "source": Column(required=True)}


def spec(dimensions: int) -> VectorSpec:
    """The `knowledge_items`/`knowledge_vectors` spec, for vectors of `dimensions` floats."""
    return VectorSpec(name="knowledge", dimensions=dimensions, columns=COLUMNS)


class VectorKnowledgeStore:
    """A `KnowledgeStore` reading and writing one `VectorStore`."""

    def __init__(self, vectors: VectorStore) -> None:
        """Hold the vector storage this corpus's chunks live in."""
        self._vectors = vectors

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        return await self._vectors.add(
            payload={"text": text, "source": source}, embedding=embedding
        )

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        found = await self._vectors.nearest(embedding=embedding, k=k)
        return [_match(row) for row in found]

    async def reset(self, *, version: str | None = None) -> None:
        """Empty the store and record `version` as the corpus it is about to hold."""
        await self._vectors.reset(version=version)

    async def version(self) -> str | None:
        """The version the last `reset` recorded, or `None` if it has never been reset."""
        return await self._vectors.version()


def _match(row: VectorMatch) -> KnowledgeMatch:
    """One stored row as the `KnowledgeMatch` a `Knowledge` hands back."""
    return KnowledgeMatch(
        id=row.id,
        text=row.payload["text"],
        source=row.payload["source"],
        distance=row.distance,
    )


__all__ = ["COLUMNS", "VectorKnowledgeStore", "spec"]
