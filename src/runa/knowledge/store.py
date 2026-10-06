"""knowledge/store.py: `KnowledgeStore`, the chunks a `Knowledge` searches, and what it returns.

One interface, three backends: `knowledge/sqlite.py`, `knowledge/postgres.py`, and the
in-process pairing `runa.db.knowledge_store(...)` holds inline. Which one a bare `Knowledge()`
gets is that function's decision, asked once.

`memory/store.py`'s counterpart, and separate from it for the same reason the two concerns are
separate: these chunks are application-scoped, so nothing here is keyed by `user_id`.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class KnowledgeMatch:
    """One `Knowledge.search` result: its id, stored text, source file, and distance."""

    id: int
    text: str
    source: str
    distance: float


class KnowledgeStore(Protocol):
    """The storage a `Knowledge` needs: add/search already-embedded chunks, and clear them all.

    The escape hatch for `Knowledge(store=...)`: any object with these three async methods works,
    no inheritance required. `SQLiteKnowledgeStore` and `PostgresKnowledgeStore` satisfy it by
    matching shape.
    """

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        ...

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        ...

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`.

        The whole store, not one source: an ingest re-reads the directory, so a file deleted
        since the last one has to stop being retrievable.
        """
        ...


__all__ = ["KnowledgeMatch", "KnowledgeStore"]
