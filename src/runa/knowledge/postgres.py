"""knowledge/postgres.py: `PostgresKnowledgeStore`, the shared `KnowledgeStore`.

The `runa-ai[postgres]` extra, not a core dependency. Same tables and query shapes as
`knowledge/sqlite.py`, with a `vector` column (the `pgvector` extension) in place of
`sqlite-vec`'s `vec0` virtual table, so every replica searches one ingested corpus instead of
each ingesting its own. Distance is `<->` (Euclidean/L2), matching `sqlite-vec`'s default.
"""

import asyncpg

from runa.db.pool import connect as _connect
from runa.knowledge import KnowledgeMatch

ITEMS_TABLE = "knowledge_items"


def ddl(dimensions: int) -> str:
    """The `knowledge_items` DDL for vectors of `dimensions` floats."""
    return f"""
    CREATE TABLE IF NOT EXISTS {ITEMS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        embedding vector({dimensions}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """


class PostgresKnowledgeStore:
    """`KnowledgeStore` backed by Postgres and `pgvector`, shared across processes."""

    def __init__(self, url: str, *, dimensions: int) -> None:
        """Store which Postgres database this store's chunks/vectors live in, and vector size."""
        self.url = url
        self.dimensions = dimensions

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.url, ddl(self.dimensions))

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {ITEMS_TABLE} (text, source, embedding)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            text,
            source,
            embedding,
        )
        assert row is not None
        return row["id"]

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        pool = await self._pool()
        rows = await pool.fetch(
            f"""
            SELECT id, text, source, embedding <-> $1 AS distance
            FROM {ITEMS_TABLE}
            ORDER BY embedding <-> $1
            LIMIT $2
            """,
            embedding,
            k,
        )
        return [
            KnowledgeMatch(
                id=row["id"], text=row["text"], source=row["source"], distance=row["distance"]
            )
            for row in rows
        ]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        pool = await self._pool()
        await pool.execute(f"DELETE FROM {ITEMS_TABLE}")


__all__ = ["ITEMS_TABLE", "PostgresKnowledgeStore", "ddl"]
