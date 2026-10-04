"""knowledge/ephemeral.py: `EphemeralKnowledgeStore`, ingested chunks that die with the process.

What `runa.db.knowledge_store()` resolves to under `RUNA_DATABASE_URL=memory://`. Brute-force L2
distance over every stored vector, sharing `memory/ephemeral.py`'s `distance`: the two stores
differ in what they hold, not in how close two embeddings are.

Application-scoped rather than `user_id`-scoped, like every other `KnowledgeStore`.
"""

from dataclasses import dataclass

from runa.knowledge import KnowledgeMatch
from runa.memory.ephemeral import distance


@dataclass
class _Chunk:
    """One ingested chunk: its text, the file it came from, and its embedding."""

    id: int
    text: str
    source: str
    embedding: list[float]


_chunks: list[_Chunk] = []


def reset() -> None:
    """Drop every stored chunk. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _chunks.clear()


class EphemeralKnowledgeStore:
    """The in-process `KnowledgeStore`: this process's ingested chunks, gone when it exits."""

    def __init__(self, *, dimensions: int) -> None:
        """Store the embedding size this store expects, for parity with the other adapters."""
        self.dimensions = dimensions

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        chunk_id = len(_chunks) + 1
        _chunks.append(_Chunk(id=chunk_id, text=text, source=source, embedding=list(embedding)))
        return chunk_id

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        matches = [
            KnowledgeMatch(
                id=chunk.id,
                text=chunk.text,
                source=chunk.source,
                distance=distance(embedding, chunk.embedding),
            )
            for chunk in _chunks
        ]
        matches.sort(key=lambda match: match.distance)
        return matches[:k]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        _chunks.clear()


__all__ = ["EphemeralKnowledgeStore", "reset"]
