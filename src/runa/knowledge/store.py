"""knowledge/store.py: `KnowledgeStore`, the chunks a `Knowledge` searches, and what it returns.

Also where the corpus's version lives, which is the one piece of ingest state this layer owns
rather than `Knowledge`: see `reset`/`version` below.

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
    """The storage a `Knowledge` needs: add/search already-embedded chunks, and version the set.

    The escape hatch for `Knowledge(store=...)`: any object with these four async methods works,
    no inheritance required. `SQLiteKnowledgeStore` and `PostgresKnowledgeStore` satisfy it by
    matching shape.

    `reset`/`version` is what makes an ingest a decision about the store rather than about one
    `Knowledge` object: the store records the fingerprint of the corpus it holds, so a second
    process -- or the next request's `Agent`, which builds its own `Knowledge` -- can see that the
    chunks it would write are already there. A store that cannot persist a version can return
    `None` from `version()`, at the cost of being re-ingested by every instance that searches it.
    """

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        ...

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        ...

    async def reset(self, *, version: str | None = None) -> None:
        """Empty the store and record `version` as the corpus it is about to hold.

        The whole store, not one source: an ingest re-reads the directory, so a file deleted
        since the last one has to stop being retrievable. Emptying and stamping are one call
        because a version that outlived its chunks would claim a corpus the store doesn't have,
        and `search` would then return nothing, forever, instead of re-ingesting.
        """
        ...

    async def version(self) -> str | None:
        """The version the last `reset` recorded, or `None` if the store has never been reset."""
        ...


__all__ = ["KnowledgeMatch", "KnowledgeStore"]
