"""knowledge/sqlite.py: `SQLiteKnowledgeStore`, the local `KnowledgeStore`.

`db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`). A `vec0`
virtual table holds the embeddings; a companion table holds the chunk text and the file it came
from, joined back to it by rowid.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import pack_vector as _pack
from runa.knowledge import KnowledgeMatch

_ITEMS_TABLE = "knowledge_items"
_VECTORS_TABLE = "knowledge_vectors"


def _ddl(dimensions: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {_ITEMS_TABLE} (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS {_VECTORS_TABLE} USING vec0(
        embedding float[{dimensions}]
    );
    """


class SQLiteKnowledgeStore:
    """The local `KnowledgeStore`: `db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's chunks/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, _ddl(self.dimensions), load_vec=True)

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {_ITEMS_TABLE} (text, source) VALUES (?, ?)",
                (text, source),
            )
            item_id = cursor.lastrowid
            assert item_id is not None
            conn.execute(
                f"INSERT INTO {_VECTORS_TABLE} (rowid, embedding) VALUES (?, ?)",
                (item_id, _pack(embedding)),
            )
            conn.commit()
        return item_id

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT items.id, items.text, items.source, vectors.distance
                FROM {_VECTORS_TABLE} AS vectors
                JOIN {_ITEMS_TABLE} AS items ON items.id = vectors.rowid
                WHERE vectors.embedding MATCH ? AND vectors.k = ?
                ORDER BY vectors.distance
                """,
                (_pack(embedding), k),
            ).fetchall()
        return [
            KnowledgeMatch(id=item_id, text=text, source=source, distance=distance)
            for item_id, text, source, distance in rows
        ]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {_VECTORS_TABLE}")
            conn.execute(f"DELETE FROM {_ITEMS_TABLE}")
            conn.commit()


__all__ = ["SQLiteKnowledgeStore"]
