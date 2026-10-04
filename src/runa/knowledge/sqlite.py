"""knowledge/sqlite.py: `SQLiteKnowledgeStore`, the local `KnowledgeStore`.

`db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`). The
`vec0` pairing underneath, two tables joined by rowid, is `db/vectors.py`'s; what belongs here is
the payload it carries: a chunk of text, the file it was read from, and the `KnowledgeMatch` a row
comes back as. Unpartitioned, because `Knowledge` is application-scoped, not per user.
"""

from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.vectors import VectorTable
from runa.knowledge.store import KnowledgeMatch


class SQLiteKnowledgeStore:
    """The local `KnowledgeStore`: `db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's chunks/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        self._table = VectorTable(
            self.db_path,
            name="knowledge",
            columns={"text": "TEXT NOT NULL", "source": "TEXT NOT NULL"},
            dimensions=dimensions,
        )

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        return self._table.insert(embedding, text=text, source=source)

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        rows = self._table.nearest(embedding, k=k, columns=("text", "source"))
        return [
            KnowledgeMatch(id=item_id, text=text, source=source, distance=distance)
            for item_id, text, source, distance in rows
        ]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        self._table.clear()


__all__ = ["SQLiteKnowledgeStore"]
