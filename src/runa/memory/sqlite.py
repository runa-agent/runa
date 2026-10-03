"""memory/sqlite.py: `SQLiteMemoryStore`, the local `MemoryStore`.

`db/runa.db`'s `memory_items`/`memory_vectors` tables, in the same connect-and-create-if-missing
file every other local adapter writes to (`db/sqlite.py`). A `vec0` virtual table holds the
embeddings, partitioned by `user_id` for correct per-user nearest-neighbor search; a companion
table holds the text and metadata they came from, joined back to it by rowid.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.db import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import pack_vector as _pack
from runa.memory import MemoryMatch

_ITEMS_TABLE = "memory_items"
_VECTORS_TABLE = "memory_vectors"


def _ddl(dimensions: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {_ITEMS_TABLE} (
        id INTEGER PRIMARY KEY,
        user_id TEXT,
        text TEXT NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS {_VECTORS_TABLE} USING vec0(
        user_id TEXT PARTITION KEY,
        embedding float[{dimensions}]
    );
    """


class SQLiteMemoryStore:
    """The local `MemoryStore`: `db/runa.db`'s `memory_items`/`memory_vectors` tables.

    `user_id` is a `vec0` partition key (not just a `WHERE` filter), so nearest-neighbor search
    stays correct per user instead of a global top-k that another user's items could crowd out.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's items/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, _ddl(self.dimensions), load_vec=True)

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {_ITEMS_TABLE} (user_id, text, metadata) VALUES (?, ?, ?)",
                (user_id, text, json.dumps(metadata) if metadata is not None else None),
            )
            item_id = cursor.lastrowid
            assert item_id is not None
            conn.execute(
                f"INSERT INTO {_VECTORS_TABLE} (rowid, user_id, embedding) VALUES (?, ?, ?)",
                (item_id, user_id, _pack(embedding)),
            )
            conn.commit()
        return item_id

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT items.id, items.text, items.metadata, vectors.distance
                FROM {_VECTORS_TABLE} AS vectors
                JOIN {_ITEMS_TABLE} AS items ON items.id = vectors.rowid
                WHERE vectors.embedding MATCH ? AND vectors.k = ? AND vectors.user_id IS ?
                ORDER BY vectors.distance
                """,
                (_pack(embedding), k, user_id),
            ).fetchall()
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
        with closing(self._connect()) as conn:
            conn.execute(
                f"DELETE FROM {_VECTORS_TABLE} WHERE rowid = ? AND user_id IS ?",
                (memory_id, user_id),
            )
            conn.execute(
                f"DELETE FROM {_ITEMS_TABLE} WHERE id = ? AND user_id IS ?", (memory_id, user_id)
            )
            conn.commit()


__all__ = ["SQLiteMemoryStore"]
