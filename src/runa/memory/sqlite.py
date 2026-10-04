"""memory/sqlite.py: `SQLiteMemoryStore`, the local `MemoryStore`.

`db/runa.db`'s `memory_items`/`memory_vectors` tables, in the same connect-and-create-if-missing
file every other local adapter writes to (`db/sqlite.py`). The `vec0` pairing underneath, two
tables joined by rowid, is `db/vectors.py`'s; what belongs here is the payload it carries: a
user's fact, the metadata it arrived with, and the `MemoryMatch` a row comes back as.
"""

import json
from pathlib import Path
from typing import Any

from runa.db import DEFAULT_DB_PATH
from runa.db.vectors import VectorTable
from runa.memory.store import MemoryMatch


class SQLiteMemoryStore:
    """The local `MemoryStore`: `db/runa.db`'s `memory_items`/`memory_vectors` tables.

    `user_id` is a `vec0` partition key (not just a `WHERE` filter), so nearest-neighbor search
    stays correct per user instead of a global top-k that another user's items could crowd out.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's items/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        self._table = VectorTable(
            self.db_path,
            name="memory",
            columns={"user_id": "TEXT", "text": "TEXT NOT NULL", "metadata": "TEXT"},
            dimensions=dimensions,
            partition_by="user_id",
        )

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        return self._table.insert(
            embedding,
            user_id=user_id,
            text=text,
            metadata=json.dumps(metadata) if metadata is not None else None,
        )

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        rows = self._table.nearest(embedding, k=k, columns=("text", "metadata"), partition=user_id)
        return [
            MemoryMatch(
                id=item_id,
                text=text,
                metadata=json.loads(metadata) if metadata is not None else None,
                distance=distance,
            )
            for item_id, text, metadata, distance in rows
        ]

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        self._table.delete(memory_id, partition=user_id)


__all__ = ["SQLiteMemoryStore"]
